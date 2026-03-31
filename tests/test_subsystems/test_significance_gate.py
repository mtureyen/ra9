"""Tests for emotional significance gate (Design Decision #11)."""

from emotive.subsystems.amygdala.prototypes import SLOW_PROTOTYPE_TEXTS
from emotive.subsystems.amygdala.slow_pass import SITUATION_TO_APPRAISAL


class TestNeutralPrototypes:
    def test_conversational_redirect_exists(self):
        assert "conversational_redirect" in SLOW_PROTOTYPE_TEXTS
        assert "conversational_redirect" in SITUATION_TO_APPRAISAL

    def test_constructive_feedback_exists(self):
        assert "constructive_feedback" in SLOW_PROTOTYPE_TEXTS
        assert "constructive_feedback" in SITUATION_TO_APPRAISAL

    def test_neutral_exchange_exists(self):
        assert "neutral_exchange" in SLOW_PROTOTYPE_TEXTS
        assert "neutral_exchange" in SITUATION_TO_APPRAISAL

    def test_neutral_prototypes_have_neutral_valence(self):
        for name in ["conversational_redirect", "neutral_exchange"]:
            assert SITUATION_TO_APPRAISAL[name]["valence"] == 0.5

    def test_neutral_prototypes_have_low_goal_relevance(self):
        for name in ["conversational_redirect", "neutral_exchange"]:
            assert SITUATION_TO_APPRAISAL[name]["goal_relevance"] <= 0.3


class TestSignificanceGate:
    def test_significance_gate_returns_neutral_for_low_match(self):
        """When best prototype match is below 0.55, should return neutral."""
        from emotive.subsystems.amygdala.fast_pass import run_fast_pass

        # Use fake prototypes that won't match well with a random embedding
        fake_protos = {
            "joy": [1.0] + [0.0] * 1023,
            "sadness": [0.0, 1.0] + [0.0] * 1022,
        }
        # Random-ish embedding that doesn't match either
        random_emb = [0.01] * 1024
        result = run_fast_pass(random_emb, fake_protos)
        assert result.primary_emotion == "neutral"
        assert result.intensity < 0.1

    def test_emotional_content_still_works(self, embedding_service):
        from emotive.subsystems.amygdala.fast_pass import run_fast_pass
        from emotive.subsystems.amygdala.prototypes import (
            FAST_PROTOTYPE_TEXTS,
            compute_prototype_embeddings,
        )

        protos = compute_prototype_embeddings(FAST_PROTOTYPE_TEXTS, embedding_service)

        # Strongly emotional content
        embedding = embedding_service.embed_text(
            "I am so incredibly happy and grateful for everything"
        )
        result = run_fast_pass(embedding, protos)
        assert result.intensity > 0.2
        assert result.primary_emotion != "neutral"

    def test_redirect_not_anger(self, embedding_service):
        """Conversational redirects should not classify as anger in slow pass."""
        from emotive.subsystems.amygdala.prototypes import (
            SLOW_PROTOTYPE_TEXTS,
            compute_prototype_embeddings,
        )
        from emotive.subsystems.amygdala.slow_pass import run_slow_pass
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector

        protos = compute_prototype_embeddings(SLOW_PROTOTYPE_TEXTS, embedding_service)
        fast = AppraisalResult(
            vector=AppraisalVector(0.5, 0.5, 0.5, 0.5, 0.5),
            primary_emotion="anger",
            secondary_emotions=[],
            intensity=0.45,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )

        result = run_slow_pass(
            "no not that I meant the other thing",
            "Oh I see, let me address that instead.",
            protos, fast, embedding_service,
        )
        # Should reappraise away from anger toward neutral territory
        assert result.vector.valence >= 0.4  # not strongly negative
