"""Configuration schema for Emotive AI. Pydantic models defining the config shape."""

from pydantic import BaseModel, Field, model_validator


class RetrievalWeights(BaseModel):
    """Weights for memory retrieval ranking. Must sum to 1.0."""

    semantic: float = Field(default=0.4, ge=0.0, le=1.0)
    recency: float = Field(default=0.25, ge=0.0, le=1.0)
    spreading_activation: float = Field(default=0.2, ge=0.0, le=1.0)
    significance: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RetrievalWeights":
        total = self.semantic + self.recency + self.spreading_activation + self.significance
        if abs(total - 1.0) > 0.001:
            msg = f"Retrieval weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class ConsolidationConfig(BaseModel):
    """Settings for the consolidation engine."""

    significance_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum significance for working memory promotion",
    )
    cluster_min_size: int = Field(
        default=3, ge=2,
        description="Minimum episodic cluster size for semantic extraction",
    )
    cluster_similarity_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Embedding similarity threshold for clustering",
    )
    auto_on_session_end: bool = Field(
        default=True,
        description="Automatically run consolidation when a session ends",
    )


class DecayConfig(BaseModel):
    """Settings for memory decay."""

    episodic_rate: float = Field(default=0.0001, description="Episodic decay rate")
    semantic_rate: float = Field(default=0.00001, description="Semantic decay rate")
    procedural_rate: float = Field(default=0.000001, description="Procedural decay rate")
    archive_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Detail retention below which memories are archived",
    )


class SpreadingActivationConfig(BaseModel):
    """Settings for spreading activation in retrieval."""

    hops: int = Field(default=2, ge=0, le=5)
    decay_per_hop: float = Field(default=0.6, ge=0.0, le=1.0)


class LayerConfig(BaseModel):
    """Which layers are enabled. Phase 0: all false except temperament."""

    temperament: bool = True
    episodes: bool = False
    mood: bool = False
    personality: bool = False
    identity: bool = False


class EpisodeConfig(BaseModel):
    """Settings for emotional episodes (Phase 1+)."""

    base_half_life_minutes: float = Field(default=30.0, ge=1.0)
    formative_intensity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    encoding_strength_weight: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Weight for intensity in encoding_strength formula",
    )


class EmotiveConfig(BaseModel):
    """Root configuration for the Emotive AI system."""

    phase: int = Field(default=0, ge=0, le=5)
    layers: LayerConfig = Field(default_factory=LayerConfig)

    working_memory_capacity: int = Field(default=20, ge=1, le=100)
    retrieval_weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    spreading_activation: SpreadingActivationConfig = Field(
        default_factory=SpreadingActivationConfig,
    )
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    episodes: EpisodeConfig = Field(default_factory=EpisodeConfig)

    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
