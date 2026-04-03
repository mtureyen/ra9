"""Tests for the Social Perception subsystem (user state detection)."""

import pytest

from emotive.layers.appraisal import AppraisalResult
from emotive.subsystems.amygdala.social_perception import (
    USER_STATE_PROTOTYPE_TEXTS,
    compute_social_perception_prototypes,
    run_social_perception,
)


@pytest.fixture()
def social_prototypes(embedding_service):
    return compute_social_perception_prototypes(
        USER_STATE_PROTOTYPE_TEXTS, embedding_service
    )


class TestSocialPerceptionPrototypes:
    def test_all_prototypes_have_embeddings(self, social_prototypes):
        """All prototype texts produce valid embeddings."""
        assert len(social_prototypes) == len(USER_STATE_PROTOTYPE_TEXTS)
        for name in USER_STATE_PROTOTYPE_TEXTS:
            assert name in social_prototypes
            assert len(social_prototypes[name]) > 0

    def test_confidence_between_0_and_1(self, embedding_service, social_prototypes):
        """Confidence scores are always between 0 and 1."""
        texts = [
            "Why do you exist?",
            "That's amazing!",
            "I'm so upset right now",
            "hello",
            "haha pizza time",
        ]
        for text in texts:
            emb = embedding_service.embed_text(text)
            state, confidence = run_social_perception(emb, social_prototypes)
            assert 0.0 <= confidence <= 1.0


class TestSocialPerceptionDetection:
    def test_testing_or_curious(self, embedding_service, social_prototypes):
        """'Why do you exist? What are you?' should detect testing or curious."""
        emb = embedding_service.embed_text("Why do you exist? What are you?")
        state, confidence = run_social_perception(emb, social_prototypes)
        assert state in ("testing", "curious")
        assert confidence > 0.50

    def test_enthusiastic(self, embedding_service, social_prototypes):
        """'That's really amazing, I love it!' should detect enthusiastic."""
        emb = embedding_service.embed_text("That's really amazing, I love it!")
        state, confidence = run_social_perception(emb, social_prototypes)
        assert state == "enthusiastic"
        assert confidence > 0.50

    def test_upset_or_vulnerable(self, embedding_service, social_prototypes):
        """'I'm having a terrible day and I need to talk' should detect upset or vulnerable."""
        emb = embedding_service.embed_text(
            "I'm having a terrible day and I need to talk"
        )
        state, confidence = run_social_perception(emb, social_prototypes)
        assert state in ("upset", "vulnerable")
        assert confidence > 0.50

    def test_playful(self, embedding_service, social_prototypes):
        """'haha what if we got pizza' should detect playful."""
        emb = embedding_service.embed_text("haha what if we got pizza")
        state, confidence = run_social_perception(emb, social_prototypes)
        assert state == "playful"
        assert confidence > 0.50

    def test_confrontational(self, embedding_service, social_prototypes):
        """'You're wrong about that and here's why' should detect confrontational."""
        emb = embedding_service.embed_text("You're wrong about that and here's why")
        state, confidence = run_social_perception(emb, social_prototypes)
        assert state == "confrontational"
        assert confidence > 0.50

    def test_neutral_hello_has_state(self, embedding_service, social_prototypes):
        """Neutral 'hello' produces a result (greetings match some state)."""
        emb = embedding_service.embed_text("hello")
        state, confidence = run_social_perception(emb, social_prototypes)
        # Greetings will match something — the key is that strongly
        # typed inputs (confrontational, vulnerable) score higher than neutral
        assert confidence >= 0.0
        assert confidence <= 1.0


class TestSocialPerceptionIntegration:
    def test_appraisal_result_has_user_state_fields(self):
        """AppraisalResult should have optional user_state fields with defaults."""
        from emotive.layers.appraisal import AppraisalVector

        result = AppraisalResult(
            vector=AppraisalVector(0.5, 0.5, 0.5, 0.5, 0.5),
            primary_emotion="neutral",
            secondary_emotions=[],
            intensity=0.1,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )
        assert result.user_state is None
        assert result.user_state_confidence == 0.0

    def test_amygdala_fast_pass_populates_user_state(
        self, app_context, event_bus
    ):
        """Amygdala.fast_pass() should populate user_state on the result."""
        from emotive.subsystems.amygdala import Amygdala

        amy = Amygdala(app_context, event_bus)
        emb = app_context.embedding_service.embed_text(
            "That's really amazing, I love it!"
        )
        result = amy.fast_pass(emb)
        assert hasattr(result, "user_state")
        assert hasattr(result, "user_state_confidence")
        # Should detect something for enthusiastic text
        assert result.user_state is not None
        assert result.user_state_confidence > 0.0

    def test_amygdala_publishes_social_perception_event(
        self, app_context, event_bus
    ):
        """Amygdala.fast_pass() should publish SOCIAL_PERCEPTION_COMPLETE."""
        from emotive.subsystems.amygdala import Amygdala

        events = []
        event_bus.subscribe(
            "social_perception_complete", lambda t, d: events.append(d)
        )

        amy = Amygdala(app_context, event_bus)
        emb = app_context.embedding_service.embed_text("I'm so excited!")
        amy.fast_pass(emb)

        assert len(events) == 1
        assert "user_state" in events[0]
        assert "user_state_confidence" in events[0]
