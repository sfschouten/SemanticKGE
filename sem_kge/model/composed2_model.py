import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model.kge_model import KgeModel


class Composed2Model(KgeModel):
    """ """

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key=None,
            init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        configuration_key = self.configuration_key

        base_model1 = self.get_option("base_model1")
        base_model2 = self.get_option("base_model2")

        self.base_models = []
        for idx, base_model in enumerate([base_model1, base_model2]):
            # Initialize base model
            base_model = KgeModel.create(
                config=config,
                dataset=dataset,
                configuration_key=self.configuration_key + f".base_model{idx + 1}",
                init_for_load_only=init_for_load_only,
            )
            self.base_models.append(base_model)

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            configuration_key=configuration_key,
            scorer=None,
            create_embedders=False,
            init_for_load_only=init_for_load_only,
        )

    def prepare_job(self, job: "Job", **kwargs):
        super(KgeModel, self).prepare_job(job, **kwargs)

        for base_model in self.base_models:
            base_model._entity_embedder.prepare_job(job, **kwargs)
            base_model._relation_embedder.prepare_job(job, **kwargs)

        from kge.job import TrainingOrEvaluationJob

        if isinstance(job, TrainingOrEvaluationJob):
            def append_num_parameter(job):
                job.current_trace["epoch"]["num_parameters"] = sum(
                    map(lambda p: p.numel(), job.model.parameters())
                )

            job.post_epoch_hooks.append(append_num_parameter)

    def penalty(self, **kwargs):
        result = []
        for base_model in self.base_models:
            result += base_model.penalty(**kwargs)
        return result

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        total_score = torch.tensor(0)
        for base_model in self.base_models:
            total_score += base_model.score_spo(s, p, o, direction)
        return total_score

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        total_score = torch.tensor(0)
        for base_model in self.base_models:
            total_score += base_model.score_sp(s, p, o)
        return total_score

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        total_score = torch.tensor(0)
        for base_model in self.base_models:
            total_score += base_model.score_po(p, o, s)
        return total_score

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        total_score = torch.tensor(0)
        for base_model in self.base_models:
            total_score += base_model.score_so(s, o, p)
        return total_score

    def score_sp_po(
            self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:

        total_score = torch.tensor(0)
        for base_model in self.base_models:
            total_score += base_model.score_sp_po(s, p, o, entity_subset)
        return total_score
