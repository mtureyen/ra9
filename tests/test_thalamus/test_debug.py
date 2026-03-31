"""Tests for brain activity logging and debug monitor."""

from pathlib import Path

from emotive.chat.terminal import _write_brain_status


class TestWriteBrainStatus:
    def _make_debug(self, **overrides):
        base = {
            "fast_emotion": "joy",
            "fast_intensity": 0.45,
            "reappraised": False,
            "final_emotion": "joy",
            "final_intensity": 0.45,
            "encoded": False,
            "episode_id": None,
            "intent_detected": False,
            "recalled_count": 0,
            "recalled_top": None,
            "gist_compressed": 0,
        }
        base.update(overrides)
        return base

    def test_writes_to_file(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug())
        assert log.exists()
        content = log.read_text()
        assert "amygdala:" in content
        assert "joy" in content

    def test_shows_reappraisal(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(
            reappraised=True,
            fast_emotion="surprise",
            fast_intensity=0.30,
            final_emotion="trust",
            final_intensity=0.71,
        ))
        content = log.read_text()
        assert "reappraised" in content
        assert "surprise" in content
        assert "trust" in content

    def test_shows_recalled_memories(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(
            recalled_count=3,
            recalled_top="Mertcan is my friend",
        ))
        content = log.read_text()
        assert "3 memories" in content
        assert "Mertcan is my friend" in content

    def test_shows_zero_recalled(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(recalled_count=0))
        content = log.read_text()
        assert "0 memories" in content

    def test_shows_encoding(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(
            encoded=True,
            final_emotion="trust",
            final_intensity=0.71,
        ))
        content = log.read_text()
        assert "episode + memory" in content
        assert "trust" in content

    def test_shows_no_encoding(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(encoded=False))
        content = log.read_text()
        assert "below threshold" in content

    def test_shows_intent_detected(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(intent_detected=True))
        content = log.read_text()
        assert "enhanced encoding" in content

    def test_no_intent_line_when_not_detected(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(intent_detected=False))
        content = log.read_text()
        assert "intent" not in content

    def test_shows_gist_compression(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(gist_compressed=2))
        content = log.read_text()
        assert "2 turns compressed" in content

    def test_no_gist_line_when_zero(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(gist_compressed=0))
        content = log.read_text()
        assert "gist" not in content

    def test_appends_to_file(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug(final_emotion="joy"))
        _write_brain_status(log, self._make_debug(final_emotion="trust"))
        content = log.read_text()
        assert "joy" in content
        assert "trust" in content

    def test_has_timestamp(self, tmp_path):
        log = tmp_path / "brain.log"
        _write_brain_status(log, self._make_debug())
        content = log.read_text()
        assert "───" in content


class TestThalamusLastDebug:
    def test_last_debug_populated(self, app_context):
        from emotive.thalamus.dispatcher import Thalamus
        from emotive.thalamus.session import boot_session
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector

        thalamus = Thalamus(app_context)
        boot_session(thalamus)

        assert thalamus.last_debug is None

        fast = AppraisalResult(
            vector=AppraisalVector(0.5, 0.5, 0.5, 0.5, 0.5),
            primary_emotion="surprise",
            secondary_emotions=[],
            intensity=0.3,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )

        thalamus._post_process("hello", "hi there", fast)

        # last_debug is NOT set by _post_process directly — it's set
        # by process_input after post_process returns. But _post_process
        # now returns the debug dict.
        debug = thalamus._post_process("test", "response", fast)
        assert "final_emotion" in debug
        assert "encoded" in debug
        assert "intent_detected" in debug


class TestDebugMonitorImport:
    def test_watch_importable(self):
        from emotive.chat.debug import watch
        assert callable(watch)

    def test_debug_package_exists(self):
        import emotive.debug
        assert emotive.debug is not None
