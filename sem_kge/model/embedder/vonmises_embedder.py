from functools import partial

import mdmm

import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

from kge.job.train import TrainingJob
from kge.model import KgeEmbedder
from sem_kge import misc
from sem_kge.model import LoggingMixin


class VonMisesEmbedder(KgeEmbedder, LoggingMixin):

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
        self.dist_cls = torch.distributions.von_mises.VonMises

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

        scale_dim = base_dim
        config.set(self.configuration_key + ".concentration_embedder.dim", scale_dim)
        if self.configuration_key + ".concentration_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".concentration_embedder.type",
                self.get_option("concentration_embedder.type"),
                )
        self.concentration_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".concentration_embedder", vocab_size
        )

        mu = torch.tensor([0.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        kappa = torch.tensor([1.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        self.prior = self.dist_cls(mu, kappa, validate_args=False)

        self.last_kl_divs = []

    def init_pretrained(self, pretrained_embedder: "KgeEmbedder") -> None:
        pass

    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.loc_embedder.prepare_job(job, **kwargs)
        self.concentration_embedder.prepare_job(job, **kwargs)

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

        if indexes is None:
            mu = self.loc_embedder.embed_all()
            kappa = self.concentration_embedder.embed_all()
        else:
            mu = self.loc_embedder.embed(indexes)
            kappa = self.concentration_embedder.embed(indexes)

        kappa = F.softplus(kappa)

        dist = self.dist_cls(mu, kappa)
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

    def embed(self, indexes):
        if self.embed_as_dist:
            return self.dist(indexes=indexes)
        else:
            raise NotImplementedError()

    def embed_all(self):
        if self.embed_as_dist:
            return self.dist()
        else:
            raise NotImplementedError()

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.loc_embedder.penalty(**kwargs)
        terms += self.concentration_embedder.penalty(**kwargs)

        indexes = kwargs['indexes']
        kl_div = kl_divergence(self.dist(indexes), self.prior).mean()
        self.last_kl_divs.append(kl_div)

        if self.kl_loss:
            terms += [(
                f"{self.configuration_key}.kl",
                self.kl_max_module().value
            )]

        return terms
