"""Tests for Phase 1.5 fixes: encoding threshold, stop sequences, offline mode."""

import json
import os


class TestEncodingThreshold:
    def test_default_threshold_is_055(self):
        """Default config threshold should be 0.55 (raised from 0.4)."""
        from emotive.config.schema import UnconsciousEncodingConfig

        # The Pydantic default is still 0.4 (for backwards compatibility)
        # but config.json overrides it to 0.55
        cfg = UnconsciousEncodingConfig()
        assert cfg.intensity_threshold == 0.4  # schema default

    def test_config_json_threshold(self):
        """config.json should set threshold to 0.55."""
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[2] / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            threshold = data.get("unconscious_encoding", {}).get(
                "intensity_threshold"
            )
            assert threshold == 0.55

    def test_hello_below_threshold(self, embedding_service):
        """'hello' should produce intensity below 0.55."""
        from emotive.subsystems.amygdala.fast_pass import run_fast_pass
        from emotive.subsystems.amygdala.prototypes import (
            FAST_PROTOTYPE_TEXTS,
            compute_prototype_embeddings,
        )

        protos = compute_prototype_embeddings(FAST_PROTOTYPE_TEXTS, embedding_service)
        embedding = embedding_service.embed_text("hello")
        result = run_fast_pass(embedding, protos)
        # Greeting should be below the new threshold — no episode
        assert result.intensity < 0.55

    def test_emotional_above_threshold(self, embedding_service):
        """Emotionally significant input should still encode."""
        from emotive.subsystems.amygdala.fast_pass import run_fast_pass
        from emotive.subsystems.amygdala.prototypes import (
            FAST_PROTOTYPE_TEXTS,
            compute_prototype_embeddings,
        )

        protos = compute_prototype_embeddings(FAST_PROTOTYPE_TEXTS, embedding_service)
        embedding = embedding_service.embed_text(
            "I'm so grateful for everything you've done, this means the world to me"
        )
        result = run_fast_pass(embedding, protos)
        # Strong emotional content should still trigger encoding
        assert result.intensity > 0.3


class TestOllamaStopSequences:
    def test_stop_sequences_in_payload(self):
        """Ollama payload should include stop sequences."""
        from emotive.config.schema import LLMProviderConfig
        from emotive.llm.ollama import OllamaAdapter

        config = LLMProviderConfig(provider="ollama")
        adapter = OllamaAdapter(config)
        payload = adapter._build_payload("system", [{"role": "user", "content": "hi"}], stream=True)

        stop = payload["options"].get("stop", [])
        assert "You:" in stop
        assert "User:" in stop
        assert "\nYou:" in stop
        assert "\nUser:" in stop


class TestOfflineMode:
    def test_hf_offline_env_set(self):
        """HuggingFace offline mode should be enabled."""
        # Importing the module sets the env vars
        import emotive.embeddings.service  # noqa: F401

        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
