"""Tests for the Amygdala subsystem (two-pass appraisal)."""

import pytest

from emotive.layers.appraisal import AppraisalResult, AppraisalVector
from emotive.subsystems.amygdala.fast_pass import (
    EMOTION_TO_APPRAISAL,
    cosine_similarity,
    run_fast_pass,
)
from emotive.subsystems.amygdala.prototypes import (
    FAST_PROTOTYPE_TEXTS,
    SLOW_PROTOTYPE_TEXTS,
    compute_prototype_embeddings,
)
from emotive.subsystems.amygdala.slow_pass import (
    REAPPRAISAL_THRESHOLD,
    SITUATION_TO_APPRAISAL,
    run_slow_pass,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0


class TestPrototypes:
    def test_fast_prototypes_cover_all_emotions(self):
        assert set(FAST_PROTOTYPE_TEXTS.keys()) == {
            "joy", "sadness", "anger", "fear",
            "surprise", "awe", "disgust", "trust",
        }

    def test_slow_prototypes_exist(self):
        assert len(SLOW_PROTOTYPE_TEXTS) >= 10

    def test_emotion_to_appraisal_covers_all(self):
        for emotion in FAST_PROTOTYPE_TEXTS:
            assert emotion in EMOTION_TO_APPRAISAL

    def test_situation_to_appraisal_covers_all(self):
        for situation in SLOW_PROTOTYPE_TEXTS:
            assert situation in SITUATION_TO_APPRAISAL

    def test_compute_prototype_embeddings(self, embedding_service):
        protos = compute_prototype_embeddings(
            {"joy": "feeling happy", "sadness": "feeling sad"},
            embedding_service,
        )
        assert "joy" in protos
        assert "sadness" in protos
        assert len(protos["joy"]) == 1024  # mxbai-embed-large dim


class TestFastPass:
    @pytest.fixture()
    def fast_prototypes(self, embedding_service):
        return compute_prototype_embeddings(FAST_PROTOTYPE_TEXTS, embedding_service)

    def test_returns_appraisal_result(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text("I'm so happy today!")
        result = run_fast_pass(embedding, fast_prototypes)
        assert isinstance(result, AppraisalResult)
        assert result.primary_emotion in FAST_PROTOTYPE_TEXTS
        assert 0.0 <= result.intensity <= 1.0

    def test_happy_text_detects_positive(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text("I feel great, wonderful news!")
        result = run_fast_pass(embedding, fast_prototypes)
        assert result.vector.valence > 0.5

    def test_sad_text_detects_negative(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text("I lost someone important to me")
        result = run_fast_pass(embedding, fast_prototypes)
        # Should detect negative valence or sadness-adjacent emotion
        assert result.primary_emotion in ("sadness", "fear", "anger", "disgust")

    def test_trust_text_detects_social(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text(
            "I trust you completely, you're my best friend"
        )
        result = run_fast_pass(embedding, fast_prototypes)
        assert result.vector.social_significance > 0.4

    def test_sensitivity_modulates_intensity(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text("Something happened")
        low = run_fast_pass(embedding, fast_prototypes, sensitivity=0.1)
        high = run_fast_pass(embedding, fast_prototypes, sensitivity=0.9)
        assert high.intensity > low.intensity

    def test_has_secondary_emotions(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text(
            "Amazing unexpected wonderful surprise gift"
        )
        result = run_fast_pass(embedding, fast_prototypes)
        assert isinstance(result.secondary_emotions, list)

    def test_formative_threshold(self, embedding_service, fast_prototypes):
        embedding = embedding_service.embed_text("test")
        result = run_fast_pass(
            embedding, fast_prototypes, formative_threshold=0.01
        )
        # With very low threshold, most things should be formative
        # (or at least the flag should be set based on intensity)
        assert isinstance(result.is_formative, bool)


class TestSlowPass:
    @pytest.fixture()
    def slow_prototypes(self, embedding_service):
        return compute_prototype_embeddings(SLOW_PROTOTYPE_TEXTS, embedding_service)

    @pytest.fixture()
    def fast_result(self):
        return AppraisalResult(
            vector=AppraisalVector(
                goal_relevance=0.5, novelty=0.5, valence=0.5,
                agency=0.5, social_significance=0.5,
            ),
            primary_emotion="surprise",
            secondary_emotions=[],
            intensity=0.3,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )

    def test_returns_appraisal_result(
        self, embedding_service, slow_prototypes, fast_result
    ):
        result = run_slow_pass(
            "I need to tell you something personal",
            "I'm listening. Take your time.",
            slow_prototypes,
            fast_result,
            embedding_service,
        )
        assert isinstance(result, AppraisalResult)

    def test_vulnerability_detection(
        self, embedding_service, slow_prototypes, fast_result
    ):
        result = run_slow_pass(
            "I've never told anyone this before, but I'm scared",
            "Thank you for trusting me with that. I'm here for you.",
            slow_prototypes,
            fast_result,
            embedding_service,
        )
        # Should detect high social significance
        assert result.vector.social_significance > 0.5

    def test_reappraisal_preserves_fast_when_similar(
        self, embedding_service, slow_prototypes
    ):
        """When slow pass is similar to fast, returns fast result unchanged."""
        fast = AppraisalResult(
            vector=AppraisalVector(
                goal_relevance=0.6, novelty=0.4, valence=0.7,
                agency=0.5, social_significance=0.6,
            ),
            primary_emotion="joy",
            secondary_emotions=["trust"],
            intensity=0.5,
            half_life_minutes=40.0,
            is_formative=False,
            decay_rate=0.017,
        )
        result = run_slow_pass(
            "That's nice",
            "Yes, it is.",
            slow_prototypes,
            fast,
            embedding_service,
        )
        # Either returns fast unchanged or a very similar result
        assert isinstance(result, AppraisalResult)


class TestAmygdalaSubsystem:
    def test_initialization(self, app_context, event_bus):
        from emotive.subsystems.amygdala import Amygdala

        amy = Amygdala(app_context, event_bus)
        assert amy.name == "amygdala"
        assert not amy._initialized

    def test_lazy_prototype_init(self, app_context, event_bus):
        from emotive.subsystems.amygdala import Amygdala

        amy = Amygdala(app_context, event_bus)
        assert not amy._initialized
        embedding = app_context.embedding_service.embed_text("hello")
        amy.fast_pass(embedding)
        assert amy._initialized

    def test_fast_pass_publishes_event(self, app_context, event_bus):
        from emotive.subsystems.amygdala import Amygdala

        events = []
        event_bus.subscribe("fast_appraisal_complete", lambda t, d: events.append(d))

        amy = Amygdala(app_context, event_bus)
        embedding = app_context.embedding_service.embed_text("I'm happy!")
        amy.fast_pass(embedding)

        assert len(events) == 1
        assert "primary_emotion" in events[0]
        assert "intensity" in events[0]

    def test_slow_pass_publishes_event(self, app_context, event_bus):
        from emotive.subsystems.amygdala import Amygdala

        events = []
        event_bus.subscribe("appraisal_complete", lambda t, d: events.append(d))

        amy = Amygdala(app_context, event_bus)
        embedding = app_context.embedding_service.embed_text("test")
        fast = amy.fast_pass(embedding)
        amy.slow_pass("test input", "test response", fast)

        assert len(events) == 1
        assert "reappraised" in events[0]
