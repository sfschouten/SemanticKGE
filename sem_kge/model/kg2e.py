import torch
from kge.job import Job
from torch.distributions.kl import kl_divergence

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from sem_kge.model import GaussianEmbedder

from torch.distributions.normal import Normal


class SymmetricKG2EScorer(RelationalScorer):
    """ """

    def score_emb(self, s_emb: Normal, p_emb: Normal, o_emb: Normal, combine: str):
        n = p_emb.loc.size(0)
        if combine == "spo":
            emb = Normal(s_emb.loc - o_emb.loc - p_emb.loc, s_emb.scale + o_emb.scale + p_emb.scale)
            out = emb.log_prob(torch.zeros_like(emb.loc)).sum(-1)
        elif combine == "sp_":
            e_mu = s_emb.loc.unsqueeze(1) - o_emb.loc.unsqueeze(0) - p_emb.loc.unsqueeze(1)
            e_sigma = s_emb.scale.unsqueeze(1) + o_emb.scale.unsqueeze(0) + p_emb.scale.unsqueeze(1)
            out = Normal(e_mu, e_sigma).log_prob(torch.zeros_like(e_mu)).sum(-1)
        elif combine == "_po":
            e_mu = s_emb.loc.unsqueeze(0) - o_emb.loc.unsqueeze(1) - p_emb.loc.unsqueeze(1)
            e_sigma = s_emb.scale.unsqueeze(0) + o_emb.scale.unsqueeze(1) + p_emb.scale.unsqueeze(1)
            out = Normal(e_mu, e_sigma).log_prob(torch.zeros_like(e_mu)).sum(-1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class AsymmetricKG2EScorer(RelationalScorer):
    """ """

    def score_emb(self, s_emb: Normal, p_emb: Normal, o_emb: Normal, combine: str):
        n = p_emb.loc.size(0)
        if combine == "spo":
            e_emb = Normal(s_emb.loc - o_emb.loc, s_emb.scale + o_emb.scale)
            out = kl_divergence(e_emb, p_emb).sum(-1)
        elif combine == "sp_":
            e_mu = s_emb.loc.unsqueeze(1) - o_emb.loc.unsqueeze(0)
            e_sigma = s_emb.scale.unsqueeze(1) + o_emb.scale.unsqueeze(0)
            p_mu = p_emb.loc.unsqueeze(1)
            p_sigma = p_emb.scale.unsqueeze(1)
            out = kl_divergence(Normal(e_mu, e_sigma), Normal(p_mu, p_sigma)).sum(-1)
        elif combine == "_po":
            e_mu = s_emb.loc.unsqueeze(0) - o_emb.loc.unsqueeze(1)
            e_sigma = s_emb.scale.unsqueeze(0) + o_emb.scale.unsqueeze(1)
            p_mu = p_emb.loc.unsqueeze(1)
            p_sigma = p_emb.scale.unsqueeze(1)
            out = kl_divergence(Normal(e_mu, e_sigma), Normal(p_mu, p_sigma)).sum(-1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class KG2E(KgeModel):
    r"""Implementation of the K2GE model."""

    METRIC = {
        "kl-divergence": AsymmetricKG2EScorer,
        "expected-likelihood": SymmetricKG2EScorer
    }

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)

        distance_metric = self.check_option("distance-metric", list(self.METRIC.keys()))
        scorer = self.METRIC[distance_metric]
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=scorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

        if not isinstance(self.get_s_embedder(), GaussianEmbedder) \
                or not isinstance(self.get_p_embedder(), GaussianEmbedder)\
                or not isinstance(self.get_o_embedder(), GaussianEmbedder):
            raise ValueError("Embedders need to be GaussianEmbedder instances for KG2E.")

        self.get_s_embedder().embed_as_dist = True
        self.get_p_embedder().embed_as_dist = True
        self.get_o_embedder().embed_as_dist = True

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)

        from kge.job import TrainingJobNegativeSampling

        if (
                isinstance(job, TrainingJobNegativeSampling)
                and job.config.get("negative_sampling.implementation") == "auto"
        ):
            # TransE with batch currently tends to run out of memory, so we use triple.
            job.config.set("negative_sampling.implementation", "triple", log=True)
