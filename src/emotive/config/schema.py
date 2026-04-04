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
    inner_world: bool = False
    anamnesis: bool = False
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
    homeostasis_tick_interval: int = Field(
        default=5, ge=1, le=50,
        description="Run within-session homeostasis every N episodes",
    )
    homeostasis_tick_hours: float = Field(
        default=0.15, ge=0.01, le=1.0,
        description="Simulated hours per homeostasis tick (inter-episode gap)",
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


class EmbodiedStateConfig(BaseModel):
    """Settings for embodied state — energy, cognitive load, comfort (Phase 2.5+)."""

    energy_depletion_base: float = Field(
        default=0.008, ge=0.0, le=0.2,
        description="Base energy cost per exchange (nonlinear: doubles below 0.5)",
    )
    joy_boost: float = Field(
        default=0.03, ge=0.0, le=0.1,
        description="Energy boost from joy/awe at high intensity",
    )
    comfort_decay_rate: float = Field(
        default=0.01, ge=0.0, le=0.1,
        description="Comfort decay rate per exchange toward neutral",
    )


class WorkspaceConfig(BaseModel):
    """Settings for global workspace — attention bottleneck (Phase 2.5+)."""

    max_context_memories: int = Field(
        default=5, ge=1, le=15,
        description="Maximum memories that enter LLM context after filtering",
    )
    max_signals: int = Field(
        default=8, ge=1, le=20,
        description="Maximum total signals broadcast to context",
    )
    identity_threat_override: bool = Field(
        default=True,
        description="Identity threats always broadcast regardless of rank",
    )


class InnerSpeechConfig(BaseModel):
    """Settings for two-tier inner speech (Phase 2.5+)."""

    enabled: bool = Field(default=True, description="Enable inner speech system")
    max_tokens: int = Field(
        default=40, ge=10, le=100,
        description="Max tokens for expanded inner speech LLM call",
    )
    warmth_bypass_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="Social bonding above this skips expanded inner speech",
    )
    system2_intensity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Intensity below this (with warmth bypass) skips System 2",
    )
    system2_prediction_error_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Prediction error above this triggers System 2",
    )
    system2_conflict_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="ACC conflict above this triggers System 2",
    )


class ObsidianExportConfig(BaseModel):
    """Settings for Obsidian auto-export on session end (Phase 2.5+)."""

    auto_export: bool = Field(
        default=False,
        description="Automatically export memories to Obsidian vault on session end",
    )
    vault_path: str | None = Field(
        default=None,
        description="Path to Obsidian vault (falls back to OBSIDIAN_VAULT_PATH env var)",
    )


class DMNEnhancedConfig(BaseModel):
    """Settings for enhanced DMN — spontaneous thoughts + reflection (Phase 2.5+)."""

    flash_probability: float = Field(
        default=0.05, ge=0.0, le=0.5,
        description="Probability of mid-session spontaneous thought per exchange",
    )
    reflection_enabled: bool = Field(
        default=True,
        description="Generate reflection at session end",
    )
    low_energy_suppresses_flash: bool = Field(
        default=True,
        description="Suppress flash when energy is below 0.3",
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

    # Phase 2.5: inner world config
    embodied: EmbodiedStateConfig = Field(default_factory=EmbodiedStateConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    inner_speech: InnerSpeechConfig = Field(default_factory=InnerSpeechConfig)
    dmn_enhanced: DMNEnhancedConfig = Field(default_factory=DMNEnhancedConfig)
    obsidian: ObsidianExportConfig = Field(default_factory=ObsidianExportConfig)

    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
