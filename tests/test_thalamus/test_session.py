"""Tests for Thalamus session lifecycle (boot/end)."""

from emotive.thalamus.dispatcher import Thalamus
from emotive.thalamus.session import boot_session, end_session


class TestSessionBoot:
    def test_boot_creates_conversation(self, app_context):
        thalamus = Thalamus(app_context)
        conv_id = boot_session(thalamus)

        assert conv_id is not None
        assert thalamus.conversation_id == conv_id

    def test_boot_regenerates_schema(self, app_context):
        thalamus = Thalamus(app_context)
        boot_session(thalamus)

        # DMN should have a schema after boot
        assert thalamus.dmn.current is not None

    def test_boot_publishes_event(self, app_context, event_bus):
        events = []
        event_bus.subscribe("session_started", lambda t, d: events.append(d))

        thalamus = Thalamus(app_context)
        boot_session(thalamus)

        assert len(events) == 1
        assert "self_schema" in events[0]


class TestSessionEnd:
    def test_end_session(self, app_context):
        thalamus = Thalamus(app_context)
        boot_session(thalamus)

        result = end_session(thalamus)
        assert result is not None
        assert "episodes_archived" in result
        assert thalamus.conversation_id is None

    def test_end_publishes_event(self, app_context, event_bus):
        events = []
        event_bus.subscribe("session_ended", lambda t, d: events.append(d))

        thalamus = Thalamus(app_context)
        boot_session(thalamus)
        end_session(thalamus)

        assert len(events) == 1

    def test_end_clears_pfc_buffer(self, app_context):
        thalamus = Thalamus(app_context)
        boot_session(thalamus)
        thalamus.prefrontal.add_turn("user", "test message")

        end_session(thalamus)
        _, messages = thalamus.prefrontal.build_context()
        assert len(messages) == 0


class TestThalamusInit:
    def test_all_subsystems_initialized(self, app_context):
        thalamus = Thalamus(app_context)
        assert thalamus.amygdala is not None
        assert thalamus.association_cortex is not None
        assert thalamus.prefrontal is not None
        assert thalamus.hippocampus is not None
        assert thalamus.dmn is not None
        assert thalamus.llm is not None

    def test_subsystem_names(self, app_context):
        thalamus = Thalamus(app_context)
        assert thalamus.amygdala.name == "amygdala"
        assert thalamus.association_cortex.name == "association_cortex"
        assert thalamus.prefrontal.name == "prefrontal_cortex"
        assert thalamus.hippocampus.name == "hippocampus"
        assert thalamus.dmn.name == "dmn"


class TestThalamusPostProcess:
    def test_post_process_does_not_crash(self, app_context):
        """Post-processing should never crash, even with errors."""
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector

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

        # Should not raise even with normal input
        thalamus._post_process("hello", "hi there", fast)

    def test_post_process_publishes_response_generated(self, app_context, event_bus):
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector

        events = []
        event_bus.subscribe("response_generated", lambda t, d: events.append(d))

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

        thalamus._post_process("hello", "hi there", fast)
        assert len(events) == 1
        # Emotion may change from fast pass due to slow pass reappraisal
        assert "emotion" in events[0]
        assert "intensity" in events[0]

    def test_post_process_with_intent_detection(self, app_context):
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector

        thalamus = Thalamus(app_context)
        boot_session(thalamus)

        fast = AppraisalResult(
            vector=AppraisalVector(0.5, 0.5, 0.8, 0.5, 0.5),
            primary_emotion="joy",
            secondary_emotions=[],
            intensity=0.5,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )

        # Response with encoding intent
        thalamus._post_process(
            "you mean so much to me",
            "I want to remember this moment. Thank you.",
            fast,
        )
        # Should not crash — intent detection + encoding runs
