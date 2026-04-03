"""Tests for the Enhanced DMN subsystem (Phase 2.5 C2)."""

import pytest
from unittest.mock import AsyncMock, patch

from emotive.subsystems.dmn.spontaneous import should_flash, find_cross_memory_connection
from emotive.subsystems.dmn.reflection import build_reflection_prompt, build_spontaneous_thought_prompt
from emotive.subsystems.dmn import DefaultModeNetwork
from emotive.config.schema import DMNEnhancedConfig


# ---------------------------------------------------------------------------
# should_flash
# ---------------------------------------------------------------------------

class TestShouldFlash:
    def test_low_energy_suppresses(self):
        """Energy below threshold always returns False."""
        # Even with probability=1.0, low energy suppresses
        assert should_flash(1.0, energy=0.2) is False

    def test_high_probability_fires(self):
        """Probability=1.0 with sufficient energy always fires."""
        assert should_flash(1.0, energy=0.5) is True

    def test_zero_probability_never_fires(self):
        assert should_flash(0.0, energy=1.0) is False

    def test_custom_suppress_threshold(self):
        assert should_flash(1.0, energy=0.4, suppress_below_energy=0.5) is False
        assert should_flash(1.0, energy=0.6, suppress_below_energy=0.5) is True


# ---------------------------------------------------------------------------
# find_cross_memory_connection
# ---------------------------------------------------------------------------

class TestFindCrossMemoryConnection:
    def test_returns_none_with_fewer_than_two(self):
        assert find_cross_memory_connection([], None) is None
        assert find_cross_memory_connection([{"embedding": [1.0]}], None) is None

    def test_finds_emotionally_similar_topically_distant(self):
        # Two memories with same emotion tag but different embeddings
        mem_a = {
            "content": "A",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "tags": ["joy"],
        }
        mem_b = {
            "content": "B",
            "embedding": [0.0, 0.0, 0.0, 1.0],
            "tags": ["joy"],
        }
        result = find_cross_memory_connection([mem_a, mem_b], None)
        assert result is not None
        assert result == (mem_a, mem_b)

    def test_no_pair_when_no_shared_emotion(self):
        mem_a = {
            "content": "A",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "tags": ["joy"],
        }
        mem_b = {
            "content": "B",
            "embedding": [0.0, 0.0, 0.0, 1.0],
            "tags": ["sadness"],
        }
        result = find_cross_memory_connection([mem_a, mem_b], None)
        assert result is None


# ---------------------------------------------------------------------------
# Reflection prompts
# ---------------------------------------------------------------------------

class TestReflectionPrompts:
    def test_build_reflection_prompt_structure(self):
        system, messages = build_reflection_prompt(
            "Had a deep conversation about music",
            {"social_bonding": 0.7, "caution": 0.5},
            "Traits: curious: 0.8",
        )
        assert "inner reflective voice" in system
        assert len(messages) == 1
        assert "music" in messages[0]["content"]

    def test_build_spontaneous_thought_prompt_structure(self):
        system, messages = build_spontaneous_thought_prompt(
            {"content": "Memory about cats"},
            {"content": "Memory about space"},
        )
        assert "wandering mind" in system
        assert "cats" in messages[0]["content"]
        assert "space" in messages[0]["content"]
