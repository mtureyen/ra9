"""Tests for Phase 2.5 inner world configuration."""

from emotive.config.schema import (
    DMNEnhancedConfig,
    EmbodiedStateConfig,
    EmotiveConfig,
    InnerSpeechConfig,
    LayerConfig,
    WorkspaceConfig,
)


class TestInnerWorldConfig:
    def test_default_config_loads(self):
        config = EmotiveConfig()
        assert config.embodied is not None
        assert config.workspace is not None
        assert config.inner_speech is not None
        assert config.dmn_enhanced is not None

    def test_inner_world_layer_default_false(self):
        layers = LayerConfig()
        assert layers.inner_world is False

    def test_embodied_defaults(self):
        cfg = EmbodiedStateConfig()
        assert cfg.energy_depletion_base == 0.02
        assert cfg.joy_boost == 0.03
        assert cfg.comfort_decay_rate == 0.01

    def test_workspace_defaults(self):
        cfg = WorkspaceConfig()
        assert cfg.max_context_memories == 5
        assert cfg.max_signals == 8
        assert cfg.identity_threat_override is True

    def test_inner_speech_defaults(self):
        cfg = InnerSpeechConfig()
        assert cfg.enabled is True
        assert cfg.max_tokens == 40
        assert cfg.warmth_bypass_threshold == 0.65
        assert cfg.system2_intensity_threshold == 0.5

    def test_dmn_enhanced_defaults(self):
        cfg = DMNEnhancedConfig()
        assert cfg.flash_probability == 0.05
        assert cfg.reflection_enabled is True
        assert cfg.low_energy_suppresses_flash is True

    def test_config_roundtrip_json(self):
        config = EmotiveConfig()
        json_str = config.model_dump_json()
        restored = EmotiveConfig.model_validate_json(json_str)
        assert restored.embodied.energy_depletion_base == config.embodied.energy_depletion_base
        assert restored.workspace.max_context_memories == config.workspace.max_context_memories
        assert restored.inner_speech.warmth_bypass_threshold == config.inner_speech.warmth_bypass_threshold
        assert restored.dmn_enhanced.flash_probability == config.dmn_enhanced.flash_probability
