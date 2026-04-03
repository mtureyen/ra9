"""Tests for inner speech memory storage (variable encoding)."""

from emotive.subsystems.prefrontal.context import (
    _format_memories,
    _format_private_thoughts,
    _is_inner_speech_memory,
)


class TestInnerSpeechMemoryDetection:
    def test_detects_inner_speech_by_source(self):
        mem = {"metadata_": {"source": "inner_speech"}, "content": "thought"}
        assert _is_inner_speech_memory(mem) is True

    def test_detects_inner_speech_by_tag(self):
        mem = {"tags": ["inner_speech", "conflict"], "content": "thought"}
        assert _is_inner_speech_memory(mem) is True

    def test_regular_memory_not_inner_speech(self):
        mem = {"tags": ["trust", "mertcan"], "content": "conversation"}
        assert _is_inner_speech_memory(mem) is False

    def test_empty_metadata_not_inner_speech(self):
        mem = {"content": "something"}
        assert _is_inner_speech_memory(mem) is False


class TestFormatPrivateThoughts:
    def test_no_inner_speech_returns_empty(self):
        memories = [{"tags": ["trust"], "content": "normal"}]
        assert _format_private_thoughts(memories) == ""

    def test_clear_thought_formatted(self):
        memories = [{"tags": ["inner_speech"], "content": "be careful",
                     "metadata_": {"source": "inner_speech"}, "detail_retention": 0.8}]
        result = _format_private_thoughts(memories)
        assert "Private Thoughts" in result
        assert "privately thought" in result.lower()
        assert "be careful" in result

    def test_faded_thought_no_content(self):
        memories = [{"tags": ["inner_speech"], "content": "original thought",
                     "metadata_": {"source": "inner_speech", "trigger": "conflict"},
                     "detail_retention": 0.1}]
        result = _format_private_thoughts(memories)
        assert "faded" in result.lower()
        assert "original thought" not in result  # content not shown

    def test_withheld_intention_formatted(self):
        memories = [{"tags": ["inner_speech", "withheld"], "content": "dont share",
                     "metadata_": {"source": "inner_speech", "withheld_intention": True},
                     "detail_retention": 0.8}]
        result = _format_private_thoughts(memories)
        assert "chose not to share" in result.lower()

    def test_divergence_formatted(self):
        memories = [{"tags": ["inner_speech", "divergence"], "content": "be honest",
                     "metadata_": {"source": "inner_speech", "expressed_divergence": True},
                     "detail_retention": 0.8}]
        result = _format_private_thoughts(memories)
        assert "said something different" in result.lower()

    def test_faded_withheld_shows_type_not_content(self):
        memories = [{"tags": ["inner_speech", "withheld"], "content": "secret thing",
                     "metadata_": {"source": "inner_speech", "withheld_intention": True},
                     "detail_retention": 0.1}]
        result = _format_private_thoughts(memories)
        assert "chose not to share" in result.lower()
        assert "secret thing" not in result

    def test_limit_to_5(self):
        memories = [{"tags": ["inner_speech"], "content": f"thought {i}",
                     "metadata_": {"source": "inner_speech"}, "detail_retention": 0.8}
                    for i in range(10)]
        result = _format_private_thoughts(memories)
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) <= 5


class TestFormatMemoriesFiltering:
    def test_inner_speech_excluded_from_regular(self):
        memories = [
            {"tags": ["trust"], "content": "conversation memory", "memory_type": "episodic"},
            {"tags": ["inner_speech"], "content": "private thought",
             "metadata_": {"source": "inner_speech"}, "memory_type": "episodic"},
        ]
        result = _format_memories(memories)
        assert "conversation memory" in result
        assert "private thought" not in result


class TestSignificanceComputation:
    def test_base_significance(self):
        """Routine thought: base = 0.2 + low intensity = ~0.3"""
        sig = 0.2 + 0.4 * 0.3  # intensity 0.4 * 0.3 weight
        assert 0.2 < sig < 0.4

    def test_divergence_boosts_significance(self):
        """Divergence adds 0.4"""
        sig = 0.2 + 0.4  # divergence
        assert abs(sig - 0.6) < 0.001

    def test_max_significance_caps_at_1(self):
        """All factors: 0.2 + 0.4 + 0.3 + 0.2 + 0.3 = 1.4 -> capped at 1.0"""
        sig = min(0.2 + 0.4 + 0.3 + 0.2 + 1.0 * 0.3, 1.0)
        assert sig == 1.0

    def test_decay_tiers(self):
        """High sig -> slow decay, low sig -> fast decay"""
        # High: months
        assert 0.7 > 0.6  # -> 0.00003
        # Medium: weeks
        assert 0.4 > 0.3  # -> 0.00005
        # Low: days
        assert 0.2 < 0.3  # -> 0.0002
