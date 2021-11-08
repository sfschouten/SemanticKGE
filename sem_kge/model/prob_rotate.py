import torch
from torch.distributions import VonMises, LogNormal
from torch.distributions.kl import kl_divergence, register_kl

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.job import Job

from sem_kge.model import EulerComplexEmbedder


@register_kl(LogNormal, LogNormal)
def _kl_normal_normal(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


@register_kl(VonMises, VonMises)
def _kl_vonmises_vonmises(p, q):
    # k_q - k_p \mu'_p \mu_q + 1/2 log(k_q) - 1/2 log(k_p)
    return q.concentration - p.concentration * torch.cos(q.loc - p.loc) \
           + .5 * torch.log(q.concentration) - .5 * torch.log(p.concentration)


class ProbRotateScorer(RelationalScorer):
    """ """

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb[0].loc.size(0)

        s_angle, s_magnitude = s_emb
        p_angle, p_magnitude = p_emb
        o_angle, o_magnitude = o_emb

        def harmonic_sum(a, b):
            return a * b / (a + b)

        if combine == "spo":
            e_angle = VonMises(
                s_angle.loc - o_angle.loc,
                harmonic_sum(s_angle.concentration, o_angle.concentration)
            )
            e_magnitude = LogNormal(
                s_magnitude.loc - o_magnitude.loc,
                s_magnitude.scale + o_magnitude.scale
            )
            kl_a = kl_divergence(e_angle, p_angle)
            kl_b = kl_divergence(e_magnitude, p_magnitude)
            out = kl_a + kl_b
        elif combine in {"sp_", "_po"}:
            dim1, dim2 = (1, 0) if combine == "sp_" else (0, 1)
            e_angle = VonMises(
                s_angle.loc.unsqueeze(dim1) - o_angle.loc.unsqueeze(dim2),
                harmonic_sum(s_angle.concentration.unsqueeze(dim1), o_angle.concentration.unsqueeze(dim2))
            )
            e_magnitude = LogNormal(
                s_magnitude.loc.unsqueeze(dim1) - o_magnitude.loc.unsqueeze(dim2),
                s_magnitude.scale.unsqueeze(dim1) + o_magnitude.scale.unsqueeze(dim2)
            )
            p_angle.loc.unsqueeze_(dim1)
            p_angle.concentration.unsqueeze_(dim1)
            p_magnitude.loc.unsqueeze_(dim1)
            p_magnitude.scale.unsqueeze_(dim1)
            kl_a = kl_divergence(e_angle, p_angle)
            kl_b = kl_divergence(e_magnitude, p_magnitude)
            out = kl_a + kl_b
        else:
            raise NotImplementedError()
        if len(out.shape) > 1:
            out = out.sum(-1)
        return out.view(n, -1)


class ProbRotatE(KgeModel):

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key=None,
            init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=ProbRotateScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

        if not isinstance(self.get_s_embedder(), EulerComplexEmbedder) \
                or not isinstance(self.get_p_embedder(), EulerComplexEmbedder) \
                or not isinstance(self.get_o_embedder(), EulerComplexEmbedder):
            raise ValueError("Embedders need to be GaussianEmbedder instances for KG2E.")

        self.get_s_embedder().embed_as_dist = True
        self.get_p_embedder().embed_as_dist = True
        self.get_o_embedder().embed_as_dist = True

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)

        from kge.job import TrainingJobNegativeSampling
        if isinstance(job, TrainingJobNegativeSampling) \
                and job.config.get("negative_sampling.implementation") == "auto":
            # TransE with batch currently tends to run out of memory, so we use triple.
            job.config.set("negative_sampling.implementation", "triple", log=True)
