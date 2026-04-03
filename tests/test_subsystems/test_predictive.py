"""Tests for the Predictive Processing subsystem (Phase 2.5 B3)."""

import pytest

from emotive.subsystems.predictive.expectations import (
    compute_prediction_error,
    generate_expectation_embedding,
)
from emotive.subsystems.predictive import PredictiveProcessor


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestComputePredictionError:
    def test_none_expected_returns_neutral(self):
        actual = [0.1] * 1024
        assert compute_prediction_error(None, actual) == 0.5

    def test_identical_vectors_zero_error(self):
        vec = [0.3, 0.5, -0.1] * 341 + [0.3]
        error = compute_prediction_error(vec, vec)
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_high_error(self):
        vec = [1.0, 0.0, -1.0] * 341 + [1.0]
        opposite = [-v for v in vec]
        error = compute_prediction_error(vec, opposite)
        assert error == pytest.approx(2.0, abs=1e-6) or error == 1.0  # clamped
        # Since cosine_sim of opposite = -1, error = 1 - (-1) = 2 -> clamped to 1.0
        assert error <= 1.0

    def test_error_clamped_to_unit_range(self):
        vec = [1.0, 0.0, -1.0] * 341 + [1.0]
        opposite = [-v for v in vec]
        error = compute_prediction_error(vec, opposite)
        assert 0.0 <= error <= 1.0


class TestGenerateExpectationEmbedding:
    def test_produces_correct_dimensionality(self):
        response = [0.1] * 1024
        context = [[0.2] * 1024, [0.3] * 1024]
        result = generate_expectation_embedding(response, context)
        assert len(result) == 1024

    def test_empty_context_returns_response(self):
        response = [0.5] * 1024
        result = generate_expectation_embedding(response, [])
        assert result == response

    def test_blends_response_and_context(self):
        response = [1.0] * 4
        context = [[0.0] * 4]
        result = generate_expectation_embedding(response, context)
        # 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert result == pytest.approx([0.5] * 4, abs=1e-6)


# ---------------------------------------------------------------------------
# Subsystem integration tests (using real embeddings)
# ---------------------------------------------------------------------------

class TestPredictiveProcessor:
    def test_first_message_neutral_error(self, app_context, event_bus):
        proc = PredictiveProcessor(app_context, event_bus)
        dummy = [0.1] * 1024
        error = proc.compute_error(dummy)
        assert error == 0.5

    def test_similar_followup_low_error(self, embedding_service, app_context, event_bus):
        proc = PredictiveProcessor(app_context, event_bus)

        emb1 = embedding_service.embed_text("I love programming in Python")
        proc.store_expectation(emb1)

        emb2 = embedding_service.embed_text("Python programming is great fun")
        error = proc.compute_error(emb2)
        assert error < 0.4

    def test_topic_switch_high_error(self, embedding_service, app_context, event_bus):
        proc = PredictiveProcessor(app_context, event_bus)

        emb1 = embedding_service.embed_text("I love programming in Python")
        proc.store_expectation(emb1)

        emb2 = embedding_service.embed_text("The recipe for chocolate cake requires eggs and flour")
        error = proc.compute_error(emb2)
        assert error > 0.3  # different topic -> higher error

    def test_expectation_stored_after_exchange(self, app_context, event_bus):
        proc = PredictiveProcessor(app_context, event_bus)
        assert proc._expected_embedding is None

        emb = [0.2] * 1024
        proc.store_expectation(emb)
        assert proc._expected_embedding is not None

    def test_recent_embeddings_limited_to_three(self, app_context, event_bus):
        proc = PredictiveProcessor(app_context, event_bus)
        for i in range(5):
            proc.store_expectation([float(i)] * 1024)
        assert len(proc._recent_embeddings) == 3

    def test_reset_clears_state(self, app_context, event_bus):
        proc = PredictiveProcessor(app_context, event_bus)
        proc.store_expectation([0.1] * 1024)
        assert proc._expected_embedding is not None
        assert len(proc._recent_embeddings) > 0

        proc.reset()
        assert proc._expected_embedding is None
        assert proc._recent_embeddings == []

    def test_publishes_prediction_error_event(self, app_context, event_bus):
        from emotive.runtime.event_bus import PREDICTION_ERROR_COMPUTED

        received = []
        event_bus.subscribe(PREDICTION_ERROR_COMPUTED, lambda et, d: received.append(d))

        proc = PredictiveProcessor(app_context, event_bus)
        proc.compute_error([0.1] * 1024)
        assert len(received) == 1
        assert "error" in received[0]
        assert received[0]["error"] == 0.5

    def test_publishes_expectation_stored_event(self, app_context, event_bus):
        from emotive.runtime.event_bus import EXPECTATION_STORED

        received = []
        event_bus.subscribe(EXPECTATION_STORED, lambda et, d: received.append(d))

        proc = PredictiveProcessor(app_context, event_bus)
        proc.store_expectation([0.1] * 1024)
        assert len(received) == 1
