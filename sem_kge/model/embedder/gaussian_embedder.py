from functools import partial
from types import MethodType

import mdmm

import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import _standard_normal
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from kge.job.train import TrainingJob
from kge.model import KgeEmbedder
from sem_kge import misc
from sem_kge.model import LoggingMixin


def normal_(dist_cls, mu, sigma):
    if dist_cls == Normal:
        return Normal(mu, sigma)
    else:  # dist_cls == MultivariateNormal
        return MultivariateNormal(mu, scale_tril=sigma, validate_args=False)


class GaussianEmbedder(KgeEmbedder, LoggingMixin):

    def __init__(
            self, config, dataset, configuration_key,
            vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        base_dim = self.get_option("dim")
        self.device = self.config.get("job.device")
        self.vocab_size = vocab_size
        self.kl_loss = self.get_option_and_log("kl_loss")
        self.embed_as_dist = False

        self.mode = self.check_option("mode", ["univariate", "multivariate"])
        if self.mode == "univariate":
            self.dist_cls = torch.distributions.normal.Normal
        elif self.mode == "multivariate":
            self.dist_cls = torch.distributions.multivariate_normal.MultivariateNormal
            # create masks
            self.diag_mask = torch.diag(torch.ones(base_dim)).expand((1, base_dim, base_dim)).bool()
            self.bottom_mask = torch.tril(torch.ones((1, base_dim, base_dim))).bool()
        else:
            raise ValueError("Invalid mode!")

        # initialize loc_embedder
        config.set(self.configuration_key + ".loc_embedder.dim", base_dim)
        if self.configuration_key + ".loc_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".loc_embedder.type",
                self.get_option("loc_embedder.type"),
            )
        self.loc_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".loc_embedder", vocab_size
        )

        # initialize scale_embedder
        if self.mode == "univariate":
            scale_dim = base_dim
        else:  # self.mode == "multivariate":
            scale_dim = int(base_dim * (base_dim+1) / 2)
        config.set(self.configuration_key + ".scale_embedder.dim", scale_dim)
        if self.configuration_key + ".scale_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".scale_embedder.type",
                self.get_option("scale_embedder.type"),
            )
        self.scale_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".scale_embedder", vocab_size
        )

        mu = torch.tensor([0.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        if self.mode == "univariate":
            sigma = torch.tensor([1.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        else:
            sigma = torch.eye(base_dim, device=self.device, requires_grad=False).unsqueeze(0)
        self.prior = self.dist_cls(mu, sigma, validate_args=False)

        self.last_kl_divs = []

    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.loc_embedder.prepare_job(job, **kwargs)
        self.scale_embedder.prepare_job(job, **kwargs)

        if self.kl_loss and isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for regularization loss
            max_kl_constraint = mdmm.MaxConstraint(
                lambda: self.last_kl_divs[-1],
                self.get_option_and_log("kl_max_threshold"),
                scale=self.get_option_and_log("kl_max_scale"),
                damping=self.get_option_and_log("kl_max_damping")
            )
            dummy_val = torch.zeros(1, device=self.device)
            kl_max_module = mdmm.MDMM([max_kl_constraint])
            misc.add_constraints_to_job(job, kl_max_module)
            self.kl_max_module = partial(kl_max_module, dummy_val)

        # trace the regularization loss
        def trace_regularization_loss(job):
            last_kl_avg = sum(kl.item() / len(self.last_kl_divs) for kl in self.last_kl_divs)
            self.last_kl_divs = []
            key = f"{self.configuration_key}.kl"
            job.current_trace["batch"][key] = last_kl_avg
            if self.kl_loss and isinstance(job, TrainingJob):
                job.current_trace["batch"][f"{key}_lambda"] = max_kl_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_regularization_loss)

    def dist(self, indexes=None, use_cache=False, cache_action='push'):
        """
        Instantiates `self.dist_cls` using the parameters obtained
        from embedding `indexes` or all indexes if `indexes' is None.
        """

        def mod_rsample(dist_, sample_shape=torch.Size()):
            """Modified rsample that saves samples so we can calculate penalty later."""
            if not use_cache or cache_action == 'push':
                shape = dist_._extended_shape(sample_shape)
                eps = _standard_normal(shape, dtype=dist_.loc.dtype, device=dist_.loc.device)

                if use_cache and cache_action == 'push':
                    self.sample_stack.append(eps.detach())

            if use_cache and cache_action == 'pop':
                eps = self.sample_stack.pop(0)

            if self.mode == "univariate":
                return dist_.loc + eps * dist_.scale
            else:
                return dist_.loc + torch.matmul(dist_.scale_tril, eps.unsqueeze(-1)).squeeze(-1)

        if indexes is None:
            mu = self.loc_embedder.embed_all()
            sigma = self.scale_embedder.embed_all()
        else:
            mu = self.loc_embedder.embed(indexes)
            sigma = self.scale_embedder.embed(indexes)

        if self.mode == "multivariate":
            B, D = mu.shape
            sigma_ = torch.zeros((B, D, D), device=mu.device)
            sigma_[self.bottom_mask.expand_as(sigma_)] = sigma.view(-1)
            mask = self.diag_mask.expand_as(sigma_)
            sigma_[mask] = F.softplus(sigma_[mask])
            sigma = sigma_
        else:
            sigma = F.softplus(sigma)

        dist = normal_(self.dist_cls, mu, sigma)
        dist.rsample = MethodType(mod_rsample, dist)
        return dist

    def log_pdf(self, points, indexes):
        """
        points:  the points at which the pdf is to be evaluated [* x D]
        indexes: the indices of the loc/scale that are to parameterize the
                 distribution [*]
        returns: log of pdf [*]
        """
        dist = self.dist(indexes)
        return dist.log_prob(points).mean(dim=-1)

    def sample(self, indexes=None, use_cache=False, cache_action='push'):
        dist = self.dist(indexes, use_cache, cache_action)
        sample_shape = torch.Size([1 if self.training else 10])
        # TODO set '1' and '10' values in config

        sample = dist.rsample(sample_shape=sample_shape)
        return sample

    def embed(self, indexes):
        if self.embed_as_dist:
            return self.dist(indexes=indexes)
        else:
            return self.sample(indexes).mean(dim=0)

    def embed_all(self):
        if self.embed_as_dist:
            return self.dist()
        else:
            return self.sample().mean(dim=0)

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.loc_embedder.penalty(**kwargs)
        terms += self.scale_embedder.penalty(**kwargs)

        indexes = kwargs['indexes']
        kl_div = kl_divergence(self.dist(indexes), self.prior).mean()
        self.last_kl_divs.append(kl_div)

        if self.kl_loss:
            terms += [(
                f"{self.configuration_key}.kl",
                self.kl_max_module().value
            )]

        return terms
