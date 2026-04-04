"""Tests for ACC repetition monitor + novelty nudge."""

from emotive.subsystems.acc.repetition import RepetitionMonitor


class TestRepetitionMonitor:
    def test_no_stuck_on_first_exchange(self):
        monitor = RepetitionMonitor()
        stuck = monitor.update([0.1] * 100, novelty=0.5)
        assert not stuck
        assert not monitor.is_stuck

    def test_no_stuck_on_different_responses(self):
        monitor = RepetitionMonitor()
        # Two very different embeddings
        monitor.update([1.0] + [0.0] * 99, novelty=0.5)
        stuck = monitor.update([0.0] * 99 + [1.0], novelty=0.5)
        assert not stuck

    def test_stuck_on_similar_responses(self):
        monitor = RepetitionMonitor()
        # Two nearly identical embeddings
        emb = [0.5] * 100
        monitor.update(emb, novelty=0.5)
        stuck = monitor.update(emb, novelty=0.5)
        assert stuck
        assert monitor.is_stuck

    def test_stuck_on_slightly_varied_responses(self):
        monitor = RepetitionMonitor()
        emb1 = [0.5] * 100
        emb2 = [0.5] * 99 + [0.51]  # Tiny variation
        monitor.update(emb1, novelty=0.5)
        stuck = monitor.update(emb2, novelty=0.5)
        # Should be stuck — nearly identical
        assert stuck

    def test_stuck_on_declining_novelty(self):
        monitor = RepetitionMonitor()
        # Different responses but declining novelty
        monitor.update([1.0] + [0.0] * 99, novelty=0.8)
        monitor.update([0.0, 1.0] + [0.0] * 98, novelty=0.6)
        stuck = monitor.update([0.0, 0.0, 1.0] + [0.0] * 97, novelty=0.4)
        # 0.8 → 0.6 → 0.4 = three consecutive drops
        assert stuck

    def test_not_stuck_with_rising_novelty(self):
        monitor = RepetitionMonitor()
        monitor.update([1.0] + [0.0] * 99, novelty=0.3)
        monitor.update([0.0, 1.0] + [0.0] * 98, novelty=0.5)
        stuck = monitor.update([0.0, 0.0, 1.0] + [0.0] * 97, novelty=0.7)
        # 0.3 → 0.5 → 0.7 = rising, not stuck
        assert not stuck

    def test_not_stuck_with_stable_novelty(self):
        monitor = RepetitionMonitor()
        monitor.update([1.0] + [0.0] * 99, novelty=0.5)
        monitor.update([0.0, 1.0] + [0.0] * 98, novelty=0.5)
        stuck = monitor.update([0.0, 0.0, 1.0] + [0.0] * 97, novelty=0.5)
        # Stable novelty, different responses — not stuck
        assert not stuck

    def test_cancel_nudge_high_novelty(self):
        monitor = RepetitionMonitor()
        emb = [0.5] * 100
        monitor.update(emb, novelty=0.5)
        monitor.update(emb, novelty=0.5)
        assert monitor.is_stuck

        # User introduces something new
        cancelled = monitor.cancel_nudge(input_novelty=0.7)
        assert cancelled
        assert not monitor.is_stuck

    def test_cancel_nudge_low_novelty_stays_stuck(self):
        monitor = RepetitionMonitor()
        emb = [0.5] * 100
        monitor.update(emb, novelty=0.5)
        monitor.update(emb, novelty=0.5)
        assert monitor.is_stuck

        # User didn't introduce anything new
        cancelled = monitor.cancel_nudge(input_novelty=0.3)
        assert not cancelled
        assert monitor.is_stuck

    def test_reset(self):
        monitor = RepetitionMonitor()
        emb = [0.5] * 100
        monitor.update(emb, novelty=0.5)
        monitor.update(emb, novelty=0.5)
        assert monitor.is_stuck

        monitor.reset()
        assert not monitor.is_stuck
        assert len(monitor._response_embeddings) == 0
        assert len(monitor._novelty_scores) == 0

    def test_max_history(self):
        monitor = RepetitionMonitor(max_history=2)
        monitor.update([1.0] + [0.0] * 99, novelty=0.5)
        monitor.update([0.0, 1.0] + [0.0] * 98, novelty=0.5)
        monitor.update([0.0, 0.0, 1.0] + [0.0] * 97, novelty=0.5)
        # Only last 2 should be in history
        assert len(monitor._response_embeddings) == 2


class TestNoveltyNudge:
    def test_nudge_formatted_in_context(self):
        from emotive.subsystems.prefrontal.context import _format_memories

        memories = [
            {"content": "Normal memory", "memory_type": "semantic"},
            {"content": "A forgotten angle", "memory_type": "episodic", "_novelty_nudge": True},
        ]
        result = _format_memories(memories)
        assert "[nudge]" in result
        assert "Something you haven't thought about recently" in result
        assert "[semantic]" in result

    def test_nudge_not_in_normal_memories(self):
        from emotive.subsystems.prefrontal.context import _format_memories

        memories = [
            {"content": "Normal memory", "memory_type": "semantic"},
        ]
        result = _format_memories(memories)
        assert "[nudge]" not in result


class TestRepetitionInThalamus:
    def test_thalamus_has_monitor(self, app_context):
        from emotive.thalamus.dispatcher import Thalamus

        thalamus = Thalamus(app_context)
        assert hasattr(thalamus, "_repetition_monitor")
        assert isinstance(thalamus._repetition_monitor, RepetitionMonitor)

    def test_session_resets_monitor(self, app_context):
        from emotive.thalamus.dispatcher import Thalamus
        from emotive.thalamus.session import boot_session, end_session

        thalamus = Thalamus(app_context)

        # Simulate stuck state
        emb = [0.5] * 100
        thalamus._repetition_monitor.update(emb, novelty=0.5)
        thalamus._repetition_monitor.update(emb, novelty=0.5)
        assert thalamus._repetition_monitor.is_stuck

        # Boot should reset
        boot_session(thalamus)
        assert not thalamus._repetition_monitor.is_stuck

        # End should reset
        thalamus._repetition_monitor.update(emb, novelty=0.5)
        thalamus._repetition_monitor.update(emb, novelty=0.5)
        end_session(thalamus)
        assert not thalamus._repetition_monitor.is_stuck

    def test_debug_includes_loop_detected(self, app_context):
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector
        from emotive.thalamus.dispatcher import Thalamus
        from emotive.thalamus.session import boot_session

        thalamus = Thalamus(app_context)
        boot_session(thalamus)

        fast = AppraisalResult(
            vector=AppraisalVector(0.5, 0.5, 0.5, 0.5, 0.5),
            primary_emotion="surprise",
            secondary_emotions=[],
            intensity=0.3,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )

        debug = thalamus._post_process("hello", "hi there", fast)
        assert "loop_detected" in debug


class TestBrainLogLoopDetection:
    def test_loop_shown_in_brain_log(self, tmp_path):
        from emotive.chat.terminal import _write_brain_status

        log = tmp_path / "brain.log"
        _write_brain_status(log, {
            "fast_emotion": "joy", "fast_intensity": 0.5,
            "reappraised": False, "final_emotion": "joy", "final_intensity": 0.5,
            "recalled_count": 5, "recalled_top": "some memory",
            "encoded": False, "intent_detected": False, "gist_compressed": 0,
            "loop_detected": True,
        })
        content = log.read_text()
        assert "loop" in content
        assert "novelty nudge" in content

    def test_no_loop_not_shown(self, tmp_path):
        from emotive.chat.terminal import _write_brain_status

        log = tmp_path / "brain.log"
        _write_brain_status(log, {
            "fast_emotion": "joy", "fast_intensity": 0.5,
            "reappraised": False, "final_emotion": "joy", "final_intensity": 0.5,
            "recalled_count": 5, "recalled_top": "some memory",
            "encoded": False, "intent_detected": False, "gist_compressed": 0,
            "loop_detected": False,
        })
        content = log.read_text()
        assert "loop" not in content
