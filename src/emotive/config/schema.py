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


class MoodConfig(BaseModel):
    """Settings for mood — neurochemical residue (Phase 2+)."""

    residue_scale: float = Field(
        default=1.0, ge=0.0, le=3.0,
        description="Multiplier for episode residue on mood (1.0 = normal)",
    )
    sensitivity_influence: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="How much mood modulates amygdala sensitivity",
    )


class EpisodeConfig(BaseModel):
    """Settings for emotional episodes (Phase 1+)."""

    base_half_life_minutes: float = Field(default=30.0, ge=1.0)
    formative_intensity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    encoding_strength_weight: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Weight for intensity in encoding_strength formula",
    )


class UnconsciousEncodingConfig(BaseModel):
    """Settings for automatic memory encoding (Phase 1.5+)."""

    enabled: bool = True
    intensity_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Minimum appraisal intensity to trigger encoding",
    )
    max_per_exchange: int = Field(
        default=3, ge=1,
        description="Maximum memories encoded per exchange",
    )
    cooldown_seconds: float = Field(
        default=10.0, ge=0.0,
        description="Minimum seconds between encoding events",
    )


class AutoRecallConfig(BaseModel):
    """Settings for automatic memory retrieval (Phase 1.5+)."""

    enabled: bool = True
    limit: int = Field(default=10, ge=1, description="Max memories to recall")
    include_spreading: bool = True


class GistConfig(BaseModel):
    """Settings for conversation gist compression (Phase 1.5+)."""

    active_buffer_size: int = Field(
        default=6, ge=2, le=20,
        description="Number of turns kept in active buffer",
    )
    primacy_pins: int = Field(
        default=2, ge=0, le=4,
        description="First N turns pinned with decay resistance",
    )


class SelfSchemaConfig(BaseModel):
    """Settings for DMN-analog self-schema generation (Phase 1.5+)."""

    enabled: bool = True
    max_traits: int = Field(default=10, ge=1)
    max_core_facts: int = Field(default=10, ge=1)
    max_values: int = Field(default=5, ge=1)


class LLMProviderConfig(BaseModel):
    """Settings for LLM provider (Phase 1.5+)."""

    provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama' or 'anthropic'",
    )
    host: str = Field(
        default="http://localhost:11434",
        description="Ollama host URL",
    )
    model: str = Field(
        default="qwen2.5:14b",
        description="Model name/ID",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    api_key: str | None = Field(
        default=None,
        description="API key (Anthropic provider only)",
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
    mood: MoodConfig = Field(default_factory=MoodConfig)

    # Phase 1.5: cognitive pipeline config
    unconscious_encoding: UnconsciousEncodingConfig = Field(
        default_factory=UnconsciousEncodingConfig,
    )
    auto_recall: AutoRecallConfig = Field(default_factory=AutoRecallConfig)
    gist: GistConfig = Field(default_factory=GistConfig)
    self_schema: SelfSchemaConfig = Field(default_factory=SelfSchemaConfig)
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)

    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
