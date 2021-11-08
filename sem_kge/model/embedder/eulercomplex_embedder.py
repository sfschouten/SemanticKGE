from kge.model import KgeEmbedder

from sem_kge.model import LoggingMixin


class EulerComplexEmbedder(KgeEmbedder, LoggingMixin):

    def __init__(self, config, dataset, configuration_key, vocab_size, init_for_load_only=False):
        super().__init__(config, dataset, configuration_key, init_for_load_only=init_for_load_only)

        base_dim = self.get_option("dim")
        self.device = self.config.get("job.device")
        self.vocab_size = vocab_size
        self.embed_as_dist = False

        config.set(self.configuration_key + ".angle_embedder.dim", base_dim)
        if self.configuration_key + ".angle_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".angle_embedder.type",
                self.get_option("angle_embedder.type"),
            )
        self.angle_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".angle_embedder", vocab_size
        )

        scale_dim = base_dim
        config.set(self.configuration_key + ".magnitude_embedder.dim", scale_dim)
        if self.configuration_key + ".magnitude_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".magnitude_embedder.type",
                self.get_option("magnitude_embedder.type"),
            )
        self.magnitude_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".magnitude_embedder", vocab_size
        )

    def init_pretrained(self, pretrained_embedder: "KgeEmbedder") -> None:
        pass

    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.angle_embedder.prepare_job(job, **kwargs)
        self.magnitude_embedder.prepare_job(job, **kwargs)

    def log_pdf(self, points, indexes):
        """
        points:  the points at which the pdf is to be evaluated [* x D]
        indexes: the indices of the loc/scale that are to parameterize the
                 distribution [*]
        returns: log of pdf [*]
        """
        angle_prob = self.angle_embedder.log_pdf(points, indexes)
        magnitude_prob = self.magnitude_embedder.log_pdf(points, indexes)
        return angle_prob + magnitude_prob

    def embed(self, indexes):
        if self.embed_as_dist:
            angle_dist = self.angle_embedder.dist(indexes=indexes)
            magnitude_dist = self.magnitude_embedder.dist(indexes=indexes)
            return angle_dist, magnitude_dist
        else:
            raise NotImplementedError()

    def embed_all(self):
        if self.embed_as_dist:
            angle_dist = self.angle_embedder.dist()
            magnitude_dist = self.magnitude_embedder.dist()
            return angle_dist, magnitude_dist
        else:
            raise NotImplementedError()

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.angle_embedder.penalty(**kwargs)
        terms += self.magnitude_embedder.penalty(**kwargs)
        return terms
