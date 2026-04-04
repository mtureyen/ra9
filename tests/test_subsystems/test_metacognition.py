"""Tests for the Metacognition subsystem (Phase 2.5 D)."""

import pytest

from emotive.subsystems.prefrontal.metacognition.markers import MetacognitiveMarkers
from emotive.subsystems.prefrontal.metacognition import (
    Metacognition,
    _compute_memory_confidence,
    _compute_emotional_clarity,
    _compute_knowledge_confidence,
)
from emotive.subsystems.workspace.signals import WorkspaceOutput


class TestMetacognitiveMarkers:
    def test_low_memory_confidence_felt(self):
        m = MetacognitiveMarkers(memory_confidence=0.2, emotional_clarity=0.8, knowledge_confidence=0.8)
        assert "not sure I remember" in m.to_felt_description()

    def test_low_emotional_clarity_felt(self):
        m = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.2, knowledge_confidence=0.8)
        assert "feelings about this are mixed" in m.to_felt_description()

    def test_low_knowledge_confidence_felt(self):
        m = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.1)
        assert "unfamiliar territory" in m.to_felt_description()

    def test_all_clear_returns_empty(self):
        m = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        assert m.to_felt_description() == ""

    def test_multiple_concerns_joined(self):
        m = MetacognitiveMarkers(memory_confidence=0.2, emotional_clarity=0.2, knowledge_confidence=0.1)
        desc = m.to_felt_description()
        assert "not sure I remember" in desc
        assert "feelings" in desc
        assert "unfamiliar" in desc


class TestComputeMemoryConfidence:
    def test_no_memories_returns_zero(self):
        assert _compute_memory_confidence([]) == 0.0

    def test_averages_retrieval_scores(self):
        mems = [
            {"retrieval_score": 0.8},
            {"retrieval_score": 0.6},
        ]
        assert _compute_memory_confidence(mems) == pytest.approx(0.7)


class TestComputeKnowledgeConfidence:
    def test_no_memories_returns_zero(self):
        assert _compute_knowledge_confidence([]) == 0.0

    def test_many_high_quality_memories_high_confidence(self):
        mems = [{"retrieval_score": 0.9}] * 5
        conf = _compute_knowledge_confidence(mems)
        assert conf > 0.7


class TestMetacognitionSubsystem:
    def _make_appraisal(self, intensity=0.5, secondary=None):
        class FakeAppraisal:
            pass
        a = FakeAppraisal()
        a.intensity = intensity
        a.secondary_emotions = secondary or []
        return a

    def test_evaluate_returns_markers(self, app_context, event_bus):
        meta = Metacognition(app_context, event_bus)
        markers = meta.evaluate(
            recalled_memories=[{"retrieval_score": 0.7}],
            appraisal=self._make_appraisal(intensity=0.6),
            workspace_output=WorkspaceOutput(),
        )
        assert isinstance(markers, MetacognitiveMarkers)
        assert 0.0 <= markers.memory_confidence <= 1.0
        assert 0.0 <= markers.emotional_clarity <= 1.0
        assert 0.0 <= markers.knowledge_confidence <= 1.0

    def test_publishes_metacognition_complete(self, app_context, event_bus):
        from emotive.runtime.event_bus import METACOGNITION_COMPLETE
        received = []
        event_bus.subscribe(METACOGNITION_COMPLETE, lambda et, d: received.append(d))

        meta = Metacognition(app_context, event_bus)
        meta.evaluate(
            recalled_memories=[],
            appraisal=self._make_appraisal(),
            workspace_output=WorkspaceOutput(),
        )
        assert len(received) == 1
        assert "memory_confidence" in received[0]
