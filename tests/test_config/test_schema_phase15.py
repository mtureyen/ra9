"""Tests for Phase 1.5 config schema additions."""

import pytest
from pydantic import ValidationError

from emotive.config.schema import (
    AutoRecallConfig,
    EmotiveConfig,
    GistConfig,
    LLMProviderConfig,
    SelfSchemaConfig,
    UnconsciousEncodingConfig,
)


class TestUnconsciousEncodingConfig:
    def test_defaults(self):
        cfg = UnconsciousEncodingConfig()
        assert cfg.enabled is True
        assert cfg.intensity_threshold == 0.4
        assert cfg.max_per_exchange == 3
        assert cfg.cooldown_seconds == 10.0

    def test_threshold_bounds(self):
        UnconsciousEncodingConfig(intensity_threshold=0.0)
        UnconsciousEncodingConfig(intensity_threshold=1.0)
        with pytest.raises(ValidationError):
            UnconsciousEncodingConfig(intensity_threshold=1.5)
        with pytest.raises(ValidationError):
            UnconsciousEncodingConfig(intensity_threshold=-0.1)

    def test_max_per_exchange_minimum(self):
        with pytest.raises(ValidationError):
            UnconsciousEncodingConfig(max_per_exchange=0)


class TestAutoRecallConfig:
    def test_defaults(self):
        cfg = AutoRecallConfig()
        assert cfg.enabled is True
        assert cfg.limit == 10
        assert cfg.include_spreading is True

    def test_limit_minimum(self):
        with pytest.raises(ValidationError):
            AutoRecallConfig(limit=0)


class TestGistConfig:
    def test_defaults(self):
        cfg = GistConfig()
        assert cfg.active_buffer_size == 6
        assert cfg.primacy_pins == 2

    def test_buffer_bounds(self):
        with pytest.raises(ValidationError):
            GistConfig(active_buffer_size=1)
        with pytest.raises(ValidationError):
            GistConfig(active_buffer_size=21)

    def test_primacy_bounds(self):
        with pytest.raises(ValidationError):
            GistConfig(primacy_pins=-1)
        with pytest.raises(ValidationError):
            GistConfig(primacy_pins=5)


class TestSelfSchemaConfig:
    def test_defaults(self):
        cfg = SelfSchemaConfig()
        assert cfg.enabled is True
        assert cfg.max_traits == 10
        assert cfg.max_core_facts == 10
        assert cfg.max_values == 5


class TestLLMProviderConfig:
    def test_defaults(self):
        cfg = LLMProviderConfig()
        assert cfg.provider == "ollama"
        assert cfg.host == "http://localhost:11434"
        assert cfg.model == "qwen2.5:14b"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.api_key is None

    def test_anthropic_config(self):
        cfg = LLMProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-test",
        )
        assert cfg.provider == "anthropic"
        assert cfg.api_key == "sk-test"

    def test_remote_ollama(self):
        cfg = LLMProviderConfig(
            host="http://192.168.1.100:11434",
            model="llama3.1:8b",
        )
        assert "192.168.1.100" in cfg.host

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            LLMProviderConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMProviderConfig(temperature=2.5)


class TestEmotiveConfigPhase15:
    def test_new_sections_have_defaults(self):
        """All Phase 1.5 sections should work with existing config files."""
        cfg = EmotiveConfig()
        assert cfg.unconscious_encoding.enabled is True
        assert cfg.auto_recall.enabled is True
        assert cfg.gist.active_buffer_size == 6
        assert cfg.self_schema.enabled is True
        assert cfg.llm.provider == "ollama"

    def test_backwards_compatible(self):
        """Phase 0-1 config (without new sections) should still parse."""
        cfg = EmotiveConfig(phase=1, layers={"temperament": True, "episodes": True})
        assert cfg.phase == 1
        assert cfg.unconscious_encoding.enabled is True  # default

    def test_full_config(self):
        """Full Phase 1.5 config with all sections."""
        cfg = EmotiveConfig(
            phase=1,
            unconscious_encoding={"intensity_threshold": 0.5, "max_per_exchange": 2},
            auto_recall={"limit": 5, "include_spreading": False},
            gist={"active_buffer_size": 4, "primacy_pins": 1},
            self_schema={"max_traits": 5},
            llm={"provider": "ollama", "model": "llama3.1:8b"},
        )
        assert cfg.unconscious_encoding.intensity_threshold == 0.5
        assert cfg.auto_recall.limit == 5
        assert cfg.gist.active_buffer_size == 4
        assert cfg.llm.model == "llama3.1:8b"
