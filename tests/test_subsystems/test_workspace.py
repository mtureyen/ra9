"""Tests for the Global Workspace subsystem (Phase 2.5 C1)."""

import pytest

from emotive.subsystems.workspace.signals import WorkspaceSignal, WorkspaceOutput
from emotive.subsystems.workspace.salience import compute_salience, rank_and_select
from emotive.subsystems.workspace import GlobalWorkspace
from emotive.config.schema import WorkspaceConfig


# ---------------------------------------------------------------------------
# Pure function tests: compute_salience
# ---------------------------------------------------------------------------

class TestComputeSalience:
    def test_emotion_salience_equals_intensity(self):
        assert compute_salience("emotion", "joy", emotion_intensity=0.7) == pytest.approx(0.7)

    def test_emotion_salience_clamped(self):
        assert compute_salience("emotion", "anger", emotion_intensity=1.5) == 1.0

    def test_memory_salience_from_retrieval_score(self):
        mem = {"content": "test", "retrieval_score": 0.8}
        assert compute_salience("memory", mem) == pytest.approx(0.8)

    def test_memory_salience_missing_score(self):
        mem = {"content": "test"}
        assert compute_salience("memory", mem) == 0.0

    def test_conflict_salience_boosted(self):
        # 0.4 * 1.5 = 0.6
        assert compute_salience("conflict", 0.4) == pytest.approx(0.6)

    def test_conflict_salience_clamped_at_one(self):
        # 0.8 * 1.5 = 1.2 -> clamped to 1.0
        assert compute_salience("conflict", 0.8) == 1.0

    def test_prediction_salience_equals_error(self):
        assert compute_salience("prediction", 0.6, prediction_error=0.6) == pytest.approx(0.6)

    def test_mood_shift_salience_from_deviation(self):
        mood = {"social_bonding": 0.9, "caution": 0.5}  # 0.4 deviation
        assert compute_salience("mood_shift", mood) == pytest.approx(0.4)

    def test_unknown_type_returns_zero(self):
        assert compute_salience("unknown", "data") == 0.0


# ---------------------------------------------------------------------------
# Pure function tests: rank_and_select
# ---------------------------------------------------------------------------

class TestRankAndSelect:
    def test_empty_signals_returns_empty(self):
        output = rank_and_select([])
        assert output.broadcast == []
        assert output.unconscious == []
        assert output.broadcast_memories == []

    def test_top_signals_broadcast(self):
        signals = [
            WorkspaceSignal("a", "x", 0.9, "emotion"),
            WorkspaceSignal("b", "y", 0.1, "emotion"),
        ]
        output = rank_and_select(signals, max_signals=1)
        assert len(output.broadcast) == 1
        assert output.broadcast[0].salience == 0.9
        assert len(output.unconscious) == 1

    def test_identity_threat_always_broadcast(self):
        signals = [
            WorkspaceSignal("conflict", 0.8, 0.8, "conflict"),
            WorkspaceSignal("a", "x", 0.9, "emotion"),
        ]
        output = rank_and_select(signals, max_signals=1, identity_override=True)
        # Conflict forced + one more slot, but max_signals=1 so only conflict
        assert any(s.signal_type == "conflict" for s in output.broadcast)

    def test_identity_override_false_no_force(self):
        signals = [
            WorkspaceSignal("conflict", 0.1, 0.1, "conflict"),
            WorkspaceSignal("a", "x", 0.9, "emotion"),
        ]
        output = rank_and_select(signals, max_signals=1, identity_override=False)
        assert output.broadcast[0].salience == 0.9

    def test_memories_extracted_to_broadcast_memories(self):
        mem = {"content": "hello", "retrieval_score": 0.8}
        signals = [
            WorkspaceSignal("memory", mem, 0.8, "memory"),
        ]
        output = rank_and_select(signals, max_memories=5, max_signals=5)
        assert len(output.broadcast_memories) == 1
        assert output.broadcast_memories[0] == mem

    def test_max_memories_respected(self):
        signals = [
            WorkspaceSignal("memory", {"content": f"m{i}", "retrieval_score": 0.5}, 0.5, "memory")
            for i in range(10)
        ]
        output = rank_and_select(signals, max_memories=3, max_signals=10)
        assert len(output.broadcast_memories) == 3


# ---------------------------------------------------------------------------
# Subsystem tests
# ---------------------------------------------------------------------------

class TestGlobalWorkspace:
    def _make_appraisal(self, emotion="joy", intensity=0.5, user_state=None):
        class FakeAppraisal:
            pass
        a = FakeAppraisal()
        a.primary_emotion = emotion
        a.intensity = intensity
        a.user_state = user_state
        return a

    def test_broadcast_returns_workspace_output(self, app_context, event_bus):
        ws = GlobalWorkspace(app_context, event_bus)
        config = WorkspaceConfig()
        appraisal = self._make_appraisal()
        output = ws.broadcast(
            recalled_memories=[],
            appraisal=appraisal,
            prediction_error=0.0,
            mood={"social_bonding": 0.5},
            conflict_score=0.0,
            embodied_state={"energy": 1.0},
            config=config,
        )
        assert isinstance(output, WorkspaceOutput)

    def test_publishes_workspace_broadcast_event(self, app_context, event_bus):
        from emotive.runtime.event_bus import WORKSPACE_BROADCAST
        received = []
        event_bus.subscribe(WORKSPACE_BROADCAST, lambda et, d: received.append(d))

        ws = GlobalWorkspace(app_context, event_bus)
        config = WorkspaceConfig()
        ws.broadcast(
            recalled_memories=[],
            appraisal=self._make_appraisal(),
            prediction_error=0.0,
            mood={"social_bonding": 0.5},
            conflict_score=0.0,
            embodied_state={"energy": 1.0},
            config=config,
        )
        assert len(received) == 1
        assert "broadcast_count" in received[0]
