"""Tests for config schema validation."""

import pytest
from pydantic import ValidationError

from emotive.config.schema import (
    ConsolidationConfig,
    DecayConfig,
    EmotiveConfig,
    LayerConfig,
    RetrievalWeights,
    SpreadingActivationConfig,
)


def test_default_config_creates_valid_instance():
    cfg = EmotiveConfig()
    assert cfg.phase == 0
    assert cfg.working_memory_capacity == 20


def test_retrieval_weights_sum_to_one():
    w = RetrievalWeights()
    total = w.semantic + w.recency + w.spreading_activation + w.significance
    assert abs(total - 1.0) < 0.001


def test_retrieval_weights_reject_bad_sum():
    with pytest.raises(ValidationError):
        RetrievalWeights(semantic=0.9, recency=0.9, spreading_activation=0.1)


def test_retrieval_weights_individual_bounds():
    with pytest.raises(ValidationError):
        RetrievalWeights(semantic=-0.1, recency=0.6, spreading_activation=0.5)


def test_consolidation_config_defaults():
    c = ConsolidationConfig()
    assert c.significance_threshold == 0.3
    assert c.cluster_min_size == 3
    assert c.auto_on_session_end is True


def test_consolidation_config_cluster_min_size_floor():
    with pytest.raises(ValidationError):
        ConsolidationConfig(cluster_min_size=1)


def test_decay_config_defaults():
    d = DecayConfig()
    assert d.episodic_rate == 0.0001
    assert d.semantic_rate == 0.00001
    assert d.procedural_rate == 0.000001
    assert d.archive_threshold == 0.1


def test_spreading_activation_config_bounds():
    with pytest.raises(ValidationError):
        SpreadingActivationConfig(hops=6)
    with pytest.raises(ValidationError):
        SpreadingActivationConfig(decay_per_hop=1.5)


def test_layer_config_phase0_defaults():
    lc = LayerConfig()
    assert lc.temperament is True
    assert lc.episodes is False
    assert lc.mood is False
    assert lc.personality is False
    assert lc.identity is False


def test_emotive_config_phase_bounds():
    with pytest.raises(ValidationError):
        EmotiveConfig(phase=6)
    with pytest.raises(ValidationError):
        EmotiveConfig(phase=-1)


def test_emotive_config_embedding_model_default():
    cfg = EmotiveConfig()
    assert cfg.embedding_model == "mixedbread-ai/mxbai-embed-large-v1"


def test_episode_config_defaults():
    from emotive.config.schema import EpisodeConfig

    ec = EpisodeConfig()
    assert ec.base_half_life_minutes == 30.0
    assert ec.formative_intensity_threshold == 0.8
    assert ec.encoding_strength_weight == 0.8


def test_emotive_config_has_episode_config():
    cfg = EmotiveConfig()
    assert cfg.episodes.base_half_life_minutes == 30.0
