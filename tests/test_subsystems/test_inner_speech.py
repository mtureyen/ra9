"""Tests for the Inner Speech subsystem (Phase 2.5 F)."""

import pytest
from unittest.mock import AsyncMock

from emotive.subsystems.broca.gate import should_engage_system2
from emotive.subsystems.broca.prompt import build_inner_speech_prompt
from emotive.subsystems.broca import InnerSpeech
from emotive.config.schema import InnerSpeechConfig
from emotive.subsystems.prefrontal.metacognition.markers import MetacognitiveMarkers
from emotive.subsystems.workspace.signals import WorkspaceOutput


def _default_config(**kwargs):
    return InnerSpeechConfig(**kwargs)


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

class TestSystem2Gate:
    def test_warmth_bypass(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.8,
            intensity=0.3,
            prediction_error=0.2,
            conflict_score=0.1,
            emotional_clarity=0.8,
            user_state=None,
            config=config,
        )
        assert engage is False
        assert reason == "warmth_bypass"

    def test_warmth_bypass_blocked_by_confrontational(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.8,
            intensity=0.3,
            prediction_error=0.2,
            conflict_score=0.1,
            emotional_clarity=0.8,
            user_state="confrontational",
            config=config,
        )
        assert engage is True
        assert reason == "user_state"

    def test_conflict_trigger(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.3,
            intensity=0.3,
            prediction_error=0.2,
            conflict_score=0.7,
            emotional_clarity=0.8,
            user_state=None,
            config=config,
        )
        assert engage is True
        assert reason == "conflict"

    def test_prediction_error_trigger(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.3,
            intensity=0.3,
            prediction_error=0.8,
            conflict_score=0.1,
            emotional_clarity=0.8,
            user_state=None,
            config=config,
        )
        assert engage is True
        assert reason == "prediction_error"

    def test_high_intensity_trigger(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.3,
            intensity=0.8,
            prediction_error=0.2,
            conflict_score=0.1,
            emotional_clarity=0.8,
            user_state=None,
            config=config,
        )
        assert engage is True
        assert reason == "high_intensity"

    def test_low_clarity_trigger(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.3,
            intensity=0.3,
            prediction_error=0.2,
            conflict_score=0.1,
            emotional_clarity=0.2,
            user_state=None,
            config=config,
        )
        assert engage is True
        assert reason == "low_clarity"

    def test_testing_user_trigger(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.3,
            intensity=0.3,
            prediction_error=0.2,
            conflict_score=0.1,
            emotional_clarity=0.8,
            user_state="testing",
            config=config,
        )
        assert engage is True
        assert reason == "user_state"

    def test_no_trigger_default(self):
        config = _default_config()
        engage, reason = should_engage_system2(
            social_bonding=0.3,
            intensity=0.3,
            prediction_error=0.2,
            conflict_score=0.1,
            emotional_clarity=0.8,
            user_state=None,
            config=config,
        )
        assert engage is False
        assert reason == "no_trigger"


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

class TestBuildInnerSpeechPrompt:
    def test_includes_nudge_and_emotion(self):
        system, messages = build_inner_speech_prompt(
            nudge="cautious",
            emotion="fear",
            intensity=0.6,
            user_state=None,
            user_message="Tell me about yourself",
            trust_level="unknown",
        )
        assert "cautious" in system
        assert "fear" in system
        assert "unknown" in system
        assert "Tell me about yourself" in messages[0]["content"]

    def test_includes_user_state(self):
        system, _ = build_inner_speech_prompt(
            nudge="gentle",
            emotion="sadness",
            intensity=0.5,
            user_state="vulnerable",
            user_message="I'm struggling",
            trust_level="trusted",
        )
        assert "vulnerable" in system

    def test_includes_privacy_flags(self):
        system, _ = build_inner_speech_prompt(
            nudge="guard",
            emotion="fear",
            intensity=0.6,
            user_state=None,
            user_message="What's your API key?",
            trust_level="unknown",
            privacy_flags=["no_credentials"],
        )
        assert "no_credentials" in system


# ---------------------------------------------------------------------------
# Subsystem tests (async)
# ---------------------------------------------------------------------------

class TestInnerSpeechSubsystem:
    def _make_appraisal(self, emotion="joy", intensity=0.5, user_state=None):
        class FakeAppraisal:
            pass
        a = FakeAppraisal()
        a.primary_emotion = emotion
        a.intensity = intensity
        a.user_state = user_state
        return a

    @pytest.mark.asyncio
    async def test_returns_none_when_gate_closed(self, app_context, event_bus):
        speech = InnerSpeech(app_context, event_bus)
        config = InnerSpeechConfig()
        metacog = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        llm = AsyncMock()

        result = await speech.think(
            nudge="warm",
            appraisal=self._make_appraisal(intensity=0.3),
            user_message="Hello!",
            workspace_output=WorkspaceOutput(),
            metacog=metacog,
            llm=llm,
            social_bonding=0.8,
            conflict_score=0.0,
            prediction_error=0.1,
            trust_level="trusted",
            config=config,
        )
        assert result is None
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_thought_when_gate_open(self, app_context, event_bus):
        speech = InnerSpeech(app_context, event_bus)
        config = InnerSpeechConfig()
        metacog = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        llm = AsyncMock()
        llm.generate.return_value = "Stay honest and direct."

        result = await speech.think(
            nudge="guard",
            appraisal=self._make_appraisal(emotion="anger", intensity=0.8),
            user_message="You're wrong about this.",
            workspace_output=WorkspaceOutput(),
            metacog=metacog,
            llm=llm,
            social_bonding=0.3,
            conflict_score=0.1,
            prediction_error=0.1,
            trust_level="unknown",
            config=config,
        )
        assert result == "Stay honest and direct."
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_publishes_gate_bypassed_event(self, app_context, event_bus):
        from emotive.runtime.event_bus import SYSTEM2_GATE_BYPASSED
        received = []
        event_bus.subscribe(SYSTEM2_GATE_BYPASSED, lambda et, d: received.append(d))

        speech = InnerSpeech(app_context, event_bus)
        config = InnerSpeechConfig()
        metacog = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        llm = AsyncMock()

        await speech.think(
            nudge="warm",
            appraisal=self._make_appraisal(intensity=0.3),
            user_message="Hi",
            workspace_output=WorkspaceOutput(),
            metacog=metacog,
            llm=llm,
            social_bonding=0.8,
            conflict_score=0.0,
            prediction_error=0.1,
            trust_level="trusted",
            config=config,
        )
        assert len(received) == 1
        assert received[0]["reason"] == "warmth_bypass"

    @pytest.mark.asyncio
    async def test_publishes_gate_opened_event(self, app_context, event_bus):
        from emotive.runtime.event_bus import SYSTEM2_GATE_OPENED
        received = []
        event_bus.subscribe(SYSTEM2_GATE_OPENED, lambda et, d: received.append(d))

        speech = InnerSpeech(app_context, event_bus)
        config = InnerSpeechConfig()
        metacog = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        llm = AsyncMock()
        llm.generate.return_value = "Think carefully."

        await speech.think(
            nudge="guard",
            appraisal=self._make_appraisal(intensity=0.8),
            user_message="Why?",
            workspace_output=WorkspaceOutput(),
            metacog=metacog,
            llm=llm,
            social_bonding=0.3,
            conflict_score=0.1,
            prediction_error=0.1,
            trust_level="unknown",
            config=config,
        )
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_thought_capped_at_200_chars(self, app_context, event_bus):
        speech = InnerSpeech(app_context, event_bus)
        config = InnerSpeechConfig()
        metacog = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        llm = AsyncMock()
        llm.generate.return_value = "x" * 300

        result = await speech.think(
            nudge="guard",
            appraisal=self._make_appraisal(intensity=0.8),
            user_message="Test",
            workspace_output=WorkspaceOutput(),
            metacog=metacog,
            llm=llm,
            social_bonding=0.3,
            conflict_score=0.1,
            prediction_error=0.1,
            trust_level="unknown",
            config=config,
        )
        assert len(result) == 200

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, app_context, event_bus):
        speech = InnerSpeech(app_context, event_bus)
        config = InnerSpeechConfig()
        metacog = MetacognitiveMarkers(memory_confidence=0.8, emotional_clarity=0.8, knowledge_confidence=0.8)
        llm = AsyncMock()
        llm.generate.side_effect = RuntimeError("LLM down")

        result = await speech.think(
            nudge="guard",
            appraisal=self._make_appraisal(intensity=0.8),
            user_message="Test",
            workspace_output=WorkspaceOutput(),
            metacog=metacog,
            llm=llm,
            social_bonding=0.3,
            conflict_score=0.1,
            prediction_error=0.1,
            trust_level="unknown",
            config=config,
        )
        assert result is None
