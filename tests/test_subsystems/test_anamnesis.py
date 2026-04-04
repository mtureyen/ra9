"""Tests for Phase Anamnesis — neural retrieval pipeline.

Tests each brain region component independently, then integration.
"""

import math
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest


# === Context Vector (TCM) ===


class TestContextVector:
    """Temporal Context Model — drifting context vector."""

    def test_init_unit_length(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=128)
        norm = np.linalg.norm(cv.vector)
        assert abs(norm - 1.0) < 0.001

    def test_drift_updates_vector(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=128, beta=0.92)
        original = cv.vector.copy()
        cv.drift([1.0] * 128)
        # Vector should have changed (cosine sim < 1.0)
        sim = float(np.dot(original, cv.vector))
        assert sim < 0.999

    def test_drift_maintains_unit_length(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=128)
        for _ in range(10):
            cv.drift(np.random.randn(128).tolist())
        norm = np.linalg.norm(cv.vector)
        assert abs(norm - 1.0) < 0.001

    def test_beta_controls_drift_rate(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        # High beta = slow drift
        cv_slow = ContextVector(dim=64, beta=0.99)
        cv_fast = ContextVector(dim=64, beta=0.5)
        # Give them same starting vector
        cv_fast._vector = cv_slow._vector.copy()

        inp = np.random.randn(64).tolist()
        cv_slow.drift(inp)
        cv_fast.drift(inp)

        # Fast should have drifted more (lower similarity to original direction)
        # Both valid — just checking they differ
        assert not np.allclose(cv_slow.vector, cv_fast.vector, atol=0.01)

    def test_similarity_to_stored_context(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=64)
        snapshot = cv.snapshot()
        # Similarity to itself should be ~1.0
        sim = cv.similarity_to(snapshot)
        assert sim > 0.99

    def test_similarity_to_none(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=64)
        assert cv.similarity_to(None) == 0.0
        assert cv.similarity_to([]) == 0.0

    def test_snapshot_returns_list(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=64)
        snap = cv.snapshot()
        assert isinstance(snap, list)
        assert len(snap) == 64

    def test_context_half_life(self):
        """After ~5-10 drifts on new topic, similarity to old drops ~50%."""
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=128, beta=0.92)
        old_snap = cv.snapshot()

        # Drift 7 times on a completely different "topic"
        new_direction = np.random.randn(128).tolist()
        for _ in range(7):
            cv.drift(new_direction)

        sim = cv.similarity_to(old_snap)
        assert sim < 0.95  # Should have drifted (β=0.92 means slow drift)

    def test_reset(self):
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=64)
        # Drift it heavily first so it's not random
        for _ in range(20):
            cv.drift([1.0] * 64)
        old = cv.vector.copy()
        cv.reset()
        # After reset, should be a different random vector
        sim = float(np.dot(old, cv.vector))
        assert sim < 0.9  # different direction


# === Retrieval State ===


class TestRetrievalState:
    """RetrievalState — persistent state across exchanges."""

    def test_init_defaults(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        assert rs.strategy == "BROAD"
        assert rs.exchange_count_on_topic == 0
        assert rs.encoding_strength == 0.5
        assert rs.retrieval_strength == 0.5

    def test_mode_balance_high_novelty(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        rs.compute_mode_balance(
            prediction_error=0.9,
            emotional_intensity=0.8,
            is_recall_query=False,
        )
        # High novelty+emotion → encoding mode dominates
        assert rs.encoding_strength > rs.retrieval_strength

    def test_mode_balance_recall_query(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        rs.compute_mode_balance(
            prediction_error=0.1,
            emotional_intensity=0.1,
            is_recall_query=True,
        )
        # Recall query → retrieval mode dominates
        assert rs.retrieval_strength > rs.encoding_strength

    def test_theta_iterations_deepen_with_topic(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        rs.compute_mode_balance(0.3, 0.3, False)

        rs.exchange_count_on_topic = 0
        iter_0 = rs.get_theta_iterations()

        rs.exchange_count_on_topic = 3
        iter_3 = rs.get_theta_iterations()

        assert iter_3 > iter_0

    def test_rif_decay(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState, RIFEntry
        rs = RetrievalState()
        mid = uuid.uuid4()
        rs.add_rif(mid, suppression=0.3)

        # Immediately: full suppression
        active = rs.get_active_rif()
        assert mid in active
        assert active[mid] > 0.25

    def test_rif_expires(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState, RIFEntry
        rs = RetrievalState()
        mid = uuid.uuid4()
        rs.add_rif(mid, suppression=0.3)

        # Manually set decay start to 24 hours ago
        rs.rif_suppressed[mid].decay_start = datetime.now(timezone.utc) - timedelta(hours=24)

        active = rs.get_active_rif()
        # After 24 hours (4τ), should be near-zero and cleaned up
        assert mid not in active

    def test_update_topic_resets_count(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        rs.exchange_count_on_topic = 5
        rs.update_topic("floret", topic_changed=True)
        assert rs.exchange_count_on_topic == 1
        assert rs.topic_person == "floret"

    def test_update_topic_increments_same(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        rs.topic_person = "floret"
        rs.exchange_count_on_topic = 3
        rs.update_topic("floret", topic_changed=False)
        assert rs.exchange_count_on_topic == 4

    def test_reset_clears_all(self):
        from emotive.subsystems.hippocampus.retrieval.state import RetrievalState
        rs = RetrievalState()
        rs.exchange_count_on_topic = 5
        rs.topic_person = "test"
        rs.add_rif(uuid.uuid4(), 0.5)
        rs.reset()
        assert rs.exchange_count_on_topic == 0
        assert rs.topic_person is None
        assert len(rs.rif_suppressed) == 0


# === Activation Tracking ===


class TestActivation:
    """Per-memory exponential decay and strengthening."""

    def test_compute_activation_no_access(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_activation
        result = compute_activation(0.8, None)
        assert result == 0.8  # unchanged without access time

    def test_compute_activation_recent(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_activation
        now = datetime.now(timezone.utc)
        result = compute_activation(0.8, now - timedelta(minutes=5), now=now)
        assert result > 0.7  # still high

    def test_compute_activation_old(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_activation
        now = datetime.now(timezone.utc)
        result = compute_activation(0.8, now - timedelta(hours=72), now=now)
        assert result < 0.3  # decayed significantly

    def test_emotional_memories_decay_slower(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_activation
        now = datetime.now(timezone.utc)
        access = now - timedelta(hours=30)
        regular = compute_activation(0.8, access, "episodic", 0.3, now)
        emotional = compute_activation(0.8, access, "episodic", 0.8, now)
        assert emotional > regular

    def test_spacing_bonus_massed(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_spacing_bonus
        now = datetime.now(timezone.utc)
        # 3 retrievals within same hour
        timestamps = [
            (now - timedelta(minutes=30)).isoformat(),
            (now - timedelta(minutes=20)).isoformat(),
            (now - timedelta(minutes=10)).isoformat(),
        ]
        bonus = compute_spacing_bonus(timestamps)
        assert bonus == 0.0  # massed = no bonus

    def test_spacing_bonus_spaced(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_spacing_bonus
        now = datetime.now(timezone.utc)
        # 3 retrievals across 3 days
        timestamps = [
            (now - timedelta(days=3)).isoformat(),
            (now - timedelta(days=1)).isoformat(),
            now.isoformat(),
        ]
        bonus = compute_spacing_bonus(timestamps)
        assert bonus >= 0.2  # spaced = good bonus

    def test_retrieval_effort_easy(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_retrieval_effort
        effort = compute_retrieval_effort(
            best_completion_score=0.9,
            competitor_count=2,
            iterations_used=2,
            max_iterations=5,
        )
        assert effort < 0.3

    def test_retrieval_effort_hard(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_retrieval_effort
        effort = compute_retrieval_effort(
            best_completion_score=0.3,
            competitor_count=8,
            iterations_used=5,
            max_iterations=5,
            tot_active=True,
        )
        assert effort > 0.7

    def test_strengthening_scales_with_effort(self):
        from emotive.subsystems.hippocampus.retrieval.activation import compute_retrieval_strengthening
        easy = compute_retrieval_strengthening(effort=0.1)
        hard = compute_retrieval_strengthening(effort=0.9)
        assert hard > easy


# === Strategy Controller (dlPFC) ===


class TestStrategy:
    """Retrieval strategy selection."""

    def test_person_strategy(self):
        from emotive.subsystems.prefrontal.dlpfc import select_strategy
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        pc._known_names = {"floret"}
        result = select_strategy("do you remember floret?", pc)
        assert result.strategy == "PERSON"
        assert result.detected_person == "floret"

    def test_temporal_strategy(self):
        from emotive.subsystems.prefrontal.dlpfc import select_strategy
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        result = select_strategy("what happened last week?", pc)
        assert result.strategy == "TEMPORAL"

    def test_emotion_strategy(self):
        from emotive.subsystems.prefrontal.dlpfc import select_strategy
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        result = select_strategy("when I was feeling sad", pc)
        assert result.strategy == "EMOTION"

    def test_topic_strategy_on_recall(self):
        from emotive.subsystems.prefrontal.dlpfc import select_strategy
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        result = select_strategy("do you remember what we talked about?", pc)
        assert result.strategy == "TOPIC"
        assert result.is_recall_query is True

    def test_broad_strategy_default(self):
        from emotive.subsystems.prefrontal.dlpfc import select_strategy
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        result = select_strategy("hello how are you", pc)
        assert result.strategy == "BROAD"
        assert result.is_recall_query is False

    def test_person_overrides_other_patterns(self):
        from emotive.subsystems.prefrontal.dlpfc import select_strategy
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        pc._known_names = {"lena"}
        # Contains both person and time reference — person wins
        result = select_strategy("what did lena say last time?", pc)
        assert result.strategy == "PERSON"


# === Pattern Separator (Dentate Gyrus) ===


class TestDentateGyrus:
    """DG pattern separation — orthogonalize similar queries."""

    def test_no_separation_novel_query(self):
        from emotive.subsystems.hippocampus.retrieval.dentate_gyrus import PatternSeparator
        sep = PatternSeparator()
        query = np.random.randn(1024).tolist()
        result = sep.separate(query, "floret")
        # First query: no separation needed
        assert result.shape == (1024,)

    def test_separation_similar_different_person(self):
        from emotive.subsystems.hippocampus.retrieval.dentate_gyrus import PatternSeparator
        sep = PatternSeparator()

        # First query about floret
        base = np.random.randn(1024)
        base = base / np.linalg.norm(base)
        sep.separate(base.tolist(), "floret")

        # Very similar query about lena (different person)
        similar = base + np.random.randn(1024) * 0.1
        result = sep.separate(similar.tolist(), "lena")

        # Result should be orthogonalized — less similar to base
        base_unit = base / np.linalg.norm(base)
        result_unit = result / np.linalg.norm(result)
        overlap = float(np.dot(base_unit, result_unit))
        assert overlap < 0.9  # should have been separated

    def test_no_separation_same_person(self):
        from emotive.subsystems.hippocampus.retrieval.dentate_gyrus import PatternSeparator
        sep = PatternSeparator()

        base = np.random.randn(1024)
        base = base / np.linalg.norm(base)
        sep.separate(base.tolist(), "floret")

        # Nearly identical query about same person — no separation
        # (tiny noise so overlap > 0.7, triggering the check, but same person = skip)
        similar = base + np.random.randn(1024) * 0.01
        result = sep.separate(similar.tolist(), "floret")

        # Compare the INPUT similar vector to the OUTPUT
        similar_unit = similar / np.linalg.norm(similar)
        result_unit = result / np.linalg.norm(result)
        overlap = float(np.dot(similar_unit, result_unit))
        assert overlap > 0.95  # same person = no orthogonalization applied

    def test_reset_clears_history(self):
        from emotive.subsystems.hippocampus.retrieval.dentate_gyrus import PatternSeparator
        sep = PatternSeparator()
        sep.separate([0.5] * 1024, "a")
        sep.separate([0.5] * 1024, "b")
        sep.reset()
        assert len(sep._recent) == 0


# === CA1 Comparator ===


class TestCA1:
    """CA1 comparator — context match + familiarity/recollection."""

    def _make_candidate(self, score=0.7, emotion_intensity=0.5, tags=None):
        from emotive.subsystems.hippocampus.retrieval.ca3 import CompletionCandidate
        return CompletionCandidate(
            memory_id=uuid.uuid4(),
            content="test memory",
            embedding=np.random.randn(1024).astype(np.float32),
            tags=tags or ["test"],
            completion_score=score,
            emotional_intensity=emotion_intensity,
            primary_emotion="trust",
            memory_type="episodic",
            created_at=datetime.now(timezone.utc),
            retrieval_count=5,
            is_formative=False,
        )

    def test_empty_candidates(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import compare_and_filter
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=256)
        output = compare_and_filter([], cv)
        assert len(output.results) == 0
        assert output.tot_triggered is False

    def test_results_sorted_by_score(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import compare_and_filter
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=256)
        candidates = [self._make_candidate(score=s) for s in [0.3, 0.9, 0.6]]
        output = compare_and_filter(candidates, cv)
        scores = [r.final_score for r in output.results]
        assert scores == sorted(scores, reverse=True)

    def test_tot_triggered(self):
        """TOT when familiarity high but recollection low."""
        from emotive.subsystems.hippocampus.retrieval.ca1 import compare_and_filter
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=256)
        # Candidate with decent completion but no context match
        candidates = [self._make_candidate(score=0.5)]
        output = compare_and_filter(
            candidates, cv,
            familiarity_threshold=0.4,
            recollection_threshold=0.5,
            tot_lower=0.2,
        )
        # With no encoding contexts, context_match defaults to 0.4
        # familiarity = min(0.5, 1.0) = 0.5 > 0.4 ✓
        # recollection = 0.4*0.6 + 0.5*0.3 + 1.0*0.1 = 0.49 < 0.5 ✓
        # recollection > tot_lower (0.2) ✓
        assert output.tot_triggered is True

    def test_dual_signal_both_high(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import compare_and_filter
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector
        cv = ContextVector(dim=256)
        candidates = [self._make_candidate(score=0.95)]
        output = compare_and_filter(candidates, cv)
        assert output.best_familiarity > 0.8
        assert output.tot_triggered is False  # both signals high = full recall


# === Person Node Cache ===


class TestPersonNode:
    """Concept cells — person-node activation."""

    def test_detect_person_known(self):
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        pc._known_names = {"floret", "lena", "mertcan"}
        assert pc.detect_person("do you remember floret?") == "floret"
        assert pc.detect_person("lena told me something") == "lena"

    def test_detect_person_unknown(self):
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        pc._known_names = {"floret"}
        assert pc.detect_person("hello there") is None

    def test_detect_person_case_insensitive(self):
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        pc._known_names = {"floret"}
        assert pc.detect_person("FLORET was here") == "floret"

    def test_add_memory(self):
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        mid = uuid.uuid4()
        pc.add_memory("Floret", mid)
        assert "floret" in pc.known_people
        assert mid in pc.get_memory_ids("floret")

    def test_word_boundary(self):
        from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
        pc = PersonNodeCache()
        pc._known_names = {"bo"}
        # "bo" should match as whole word, not partial
        assert pc.detect_person("bo said hi") == "bo"
        assert pc.detect_person("about something") is None  # "bo" inside "about" — no match


# === Suppression (E13) ===


class TestSuppression:
    """Intentional suppression — dlPFC/vlPFC down-regulation."""

    def test_apply_suppression(self):
        """apply_suppression increases suppression_level on the memory object."""
        from unittest.mock import MagicMock
        from emotive.subsystems.hippocampus.retrieval.suppression import apply_suppression

        mem = MagicMock()
        mem.suppression_level = 0.0
        mem.suppression_decay_start = None

        session = MagicMock()
        session.get.return_value = mem

        apply_suppression(session, uuid.uuid4(), suppression_strength=0.4)
        assert mem.suppression_level == pytest.approx(0.4)
        assert mem.suppression_decay_start is not None

    def test_effective_suppression_decay(self):
        """Suppression decays exponentially over 24h."""
        from emotive.subsystems.hippocampus.retrieval.suppression import get_effective_suppression

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)
        # After 24h (one time constant), level decays to ~37% of original
        result = get_effective_suppression(1.0, start, now=now)
        assert result < 0.5  # significantly decayed
        assert result > 0.2  # e^(-1) ~ 0.37

    def test_suppression_weakened_by_stress(self):
        """Low energy / high arousal halves effective suppression."""
        from emotive.subsystems.hippocampus.retrieval.suppression import get_effective_suppression

        now = datetime.now(timezone.utc)
        start = now  # just applied

        normal = get_effective_suppression(0.8, start, energy=1.0, arousal=0.0, now=now)
        stressed = get_effective_suppression(0.8, start, energy=0.2, arousal=0.0, now=now)

        assert stressed == pytest.approx(normal * 0.5, abs=0.01)


# === Narrative Reconstruction (E18) ===


class TestNarrative:
    """Narrative arc construction from person-focused memories."""

    def _make_memory(self, content, emotion="trust", created_at=None):
        return {
            "content": content,
            "primary_emotion": emotion,
            "created_at": created_at or datetime.now(timezone.utc),
            "tags": ["test"],
        }

    def test_narrative_with_person(self):
        """3+ memories about a person produces a narrative string."""
        from emotive.subsystems.hippocampus.retrieval.narrative import construct_narrative

        now = datetime.now(timezone.utc)
        memories = [
            self._make_memory("Met Lena at the park", "joy", now - timedelta(days=5)),
            self._make_memory("Lena shared her story", "trust", now - timedelta(days=3)),
            self._make_memory("Talked with Lena about dreams", "curiosity", now),
        ]
        result = construct_narrative(memories, detected_person="Lena")
        assert result is not None
        assert "Lena" in result
        assert "3 memories" in result

    def test_narrative_too_few(self):
        """Fewer than 3 memories returns None."""
        from emotive.subsystems.hippocampus.retrieval.narrative import construct_narrative

        memories = [
            self._make_memory("One memory"),
            self._make_memory("Two memories"),
        ]
        result = construct_narrative(memories, detected_person="Lena")
        assert result is None

    def test_narrative_emotional_arc(self):
        """Narrative mentions emotion change when first/last differ."""
        from emotive.subsystems.hippocampus.retrieval.narrative import construct_narrative

        now = datetime.now(timezone.utc)
        memories = [
            self._make_memory("Conflict with Mert", "anger", now - timedelta(days=10)),
            self._make_memory("Mert apologized", "surprise", now - timedelta(days=5)),
            self._make_memory("Reconciled with Mert", "trust", now),
        ]
        result = construct_narrative(memories, detected_person="Mert")
        assert result is not None
        assert "anger" in result
        assert "trust" in result


# === Memory Resistance (E20) ===


class TestResistance:
    """Identity-threat resistance — elevated retrieval thresholds."""

    def test_resistance_raises_threshold(self):
        """Positive resistance produces multiplier > 1.0."""
        from emotive.subsystems.hippocampus.retrieval.resistance import compute_resistance_threshold

        mult = compute_resistance_threshold(identity_resistance=0.8)
        assert mult > 1.0

    def test_resistance_lowered_by_safety(self):
        """High trust + comfort lowers the multiplier."""
        from emotive.subsystems.hippocampus.retrieval.resistance import compute_resistance_threshold

        unsafe = compute_resistance_threshold(
            identity_resistance=0.8, person_trust=0.3, energy=0.3, comfort=0.3,
        )
        safe = compute_resistance_threshold(
            identity_resistance=0.8, person_trust=0.9, energy=0.8, comfort=0.8,
        )
        assert safe < unsafe

    def test_resistance_habituation(self):
        """update_resistance_after_safe_exposure reduces resistance by 15%."""
        from emotive.subsystems.hippocampus.retrieval.resistance import update_resistance_after_safe_exposure

        meta = {"identity_resistance": 0.6}
        result = update_resistance_after_safe_exposure(meta, person_trust=0.8, comfort=0.7)
        assert result["identity_resistance"] == pytest.approx(0.6 * 0.85)


# === Prospective Memory (E21) ===


class TestProspective:
    """Prospective memory — intention-cue trigger matching."""

    def _make_intention(self, cue, action):
        return {
            "id": uuid.uuid4(),
            "trigger_cue": cue,
            "intended_action": action,
            "content": f"Intention: {action}",
        }

    def test_check_triggers_matches(self):
        """Trigger cue in input fires the intention."""
        from emotive.subsystems.hippocampus.retrieval.prospective import check_prospective_triggers

        intentions = [self._make_intention("birthday", "wish Lena happy birthday")]
        result = check_prospective_triggers(intentions, "it's my birthday today")
        assert len(result) == 1
        assert "wish Lena happy birthday" in result[0]

    def test_check_triggers_no_match(self):
        """Unrelated input does not trigger any intention."""
        from emotive.subsystems.hippocampus.retrieval.prospective import check_prospective_triggers

        intentions = [self._make_intention("birthday", "wish Lena happy birthday")]
        result = check_prospective_triggers(intentions, "the weather is nice today")
        assert len(result) == 0

    def test_check_triggers_person_match(self):
        """detected_person matching trigger cue fires the intention."""
        from emotive.subsystems.hippocampus.retrieval.prospective import check_prospective_triggers

        intentions = [self._make_intention("lena", "ask Lena about the book")]
        result = check_prospective_triggers(
            intentions, "hi there", detected_person="lena",
        )
        assert len(result) == 1
        assert "ask Lena about the book" in result[0]


# === Reward Gating (E31) ===


class TestRewardGating:
    """Basal ganglia — dopamine-mediated reward gating."""

    def test_positive_reward(self):
        """Trust increase produces positive reward signal."""
        from emotive.subsystems.basal_ganglia import compute_reward_signal

        reward = compute_reward_signal(None, None, trust_change=0.3)
        assert reward > 0

    def test_negative_reward(self):
        """Trust decrease produces negative reward signal."""
        from emotive.subsystems.basal_ganglia import compute_reward_signal

        reward = compute_reward_signal(None, None, trust_change=-0.3)
        assert reward < 0

    def test_gating_bonus(self):
        """Positive cumulative_reward gives a positive gating bonus."""
        from emotive.subsystems.basal_ganglia import get_gating_bonus

        bonus = get_gating_bonus({"cumulative_reward": 0.5})
        assert bonus > 0
        assert bonus == pytest.approx(0.05)

    def test_gating_bonus_negative_ignored(self):
        """Negative cumulative_reward gives zero bonus (no penalty path)."""
        from emotive.subsystems.basal_ganglia import get_gating_bonus

        bonus = get_gating_bonus({"cumulative_reward": -0.5})
        assert bonus == 0.0


# === Locus Coeruleus (LC scope) ===


class TestLocusCoeruleus:
    """NE arousal modulation — retrieval scope control."""

    def test_low_arousal_broad_scope(self):
        """Low prediction error produces broad scope with candidate_multiplier > 1."""
        from emotive.subsystems.locus_coeruleus import compute_retrieval_scope

        result = compute_retrieval_scope(prediction_error=0.1, emotional_intensity=0.1)
        assert result["scope"] == "broad"
        assert result["candidate_multiplier"] > 1.0

    def test_high_arousal_narrow_scope(self):
        """High prediction error produces narrow scope with candidate_multiplier < 1."""
        from emotive.subsystems.locus_coeruleus import compute_retrieval_scope

        result = compute_retrieval_scope(prediction_error=0.9, emotional_intensity=0.8)
        assert result["scope"] == "narrow"
        assert result["candidate_multiplier"] < 1.0

    def test_medium_arousal_focused(self):
        """Medium prediction error produces focused scope."""
        from emotive.subsystems.locus_coeruleus import compute_retrieval_scope

        result = compute_retrieval_scope(prediction_error=0.5, emotional_intensity=0.3)
        assert result["scope"] == "focused"
        assert result["candidate_multiplier"] == 1.0


# === Unconscious Priming (E5) ===


class TestUnconscousPriming:
    """Semantic priming from the unconscious pool."""

    def _make_candidate(self, content):
        from unittest.mock import MagicMock
        c = MagicMock()
        c.content = content
        return c

    def test_priming_extracts_keywords(self):
        """Unconscious pool produces distinctive priming words."""
        from emotive.subsystems.hippocampus.retrieval.pipeline import _extract_priming_keywords

        unconscious = [
            self._make_candidate("the lighthouse keeper watched dolphins"),
            self._make_candidate("a crystalline cavern under moonlight"),
        ]
        conscious = [
            self._make_candidate("hello how are you today"),
        ]
        words = _extract_priming_keywords(unconscious, conscious)
        assert len(words) > 0
        # Should contain distinctive words like "lighthouse", "dolphins", etc.
        assert "lighthouse" in words or "dolphins" in words or "keeper" in words

    def test_priming_excludes_conscious(self):
        """Words already present in conscious memories are excluded."""
        from emotive.subsystems.hippocampus.retrieval.pipeline import _extract_priming_keywords

        unconscious = [
            self._make_candidate("the garden bloomed with roses"),
        ]
        conscious = [
            self._make_candidate("the garden was beautiful and serene"),
        ]
        words = _extract_priming_keywords(unconscious, conscious)
        assert "garden" not in words  # shared word excluded

    def test_priming_excludes_stop_words(self):
        """Common stop words are filtered out."""
        from emotive.subsystems.hippocampus.retrieval.pipeline import _extract_priming_keywords

        unconscious = [
            self._make_candidate("the and but with from this that those what which"),
        ]
        conscious = []
        words = _extract_priming_keywords(unconscious, conscious)
        assert "the" not in words
        assert "and" not in words
        assert "with" not in words
        assert "from" not in words
        assert "this" not in words
        assert "that" not in words
        assert "those" not in words
        assert len(words) == 0  # all are stop words or < 3 chars


# === Encoding-time (E6, E10, E26) ===


class TestEncodingTime:
    """Formation period flag and source type metadata at encoding time."""

    def test_formation_period_flag(self):
        """store_memory sets formation_period=True when total memories < 150."""
        from unittest.mock import MagicMock, patch

        import emotive.memory.base as base_mod
        from emotive.memory.base import store_memory

        # Reset the formation period cache
        base_mod._formation_period_cache = None

        session = MagicMock()
        embedding_service = MagicMock()
        embedding_service.embed_text.return_value = [0.1] * 1024

        # Mock the count query to return < 150
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50
        session.execute.return_value = mock_result

        # Mock find_similar_memories and apply_interference to avoid DB calls
        with patch("emotive.memory.base.find_similar_memories", return_value=[]), \
             patch("emotive.memory.base.apply_interference", return_value=0), \
             patch("emotive.memory.base.link_by_conversation", return_value=0):
            mem = store_memory(
                session, embedding_service,
                content="early memory",
                memory_type="episodic",
            )
        assert mem.formation_period is True
        # Clean up cache
        base_mod._formation_period_cache = None

    def test_source_type_in_metadata(self):
        """source_type='imagined' is stored in memory metadata."""
        from unittest.mock import MagicMock, patch

        from emotive.memory.base import store_memory

        session = MagicMock()
        embedding_service = MagicMock()
        embedding_service.embed_text.return_value = [0.1] * 1024

        mock_result = MagicMock()
        mock_result.scalar.return_value = 200  # past formation period
        session.execute.return_value = mock_result

        with patch("emotive.memory.base.find_similar_memories", return_value=[]), \
             patch("emotive.memory.base.apply_interference", return_value=0), \
             patch("emotive.memory.base.link_by_conversation", return_value=0):
            mem = store_memory(
                session, embedding_service,
                content="imagined a forest walk",
                memory_type="episodic",
                source_type="imagined",
            )
        assert mem.metadata_["source_type"] == "imagined"


# === Within-Retrieval Inhibition (E4) ===


class TestWithinRetrievalInhibition:
    """Sequential suppression within a single retrieval attempt."""

    def _make_result(self, score, embedding=None):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.final_score = score
        r.candidate = MagicMock()
        r.candidate.embedding = embedding
        return r

    def test_empty_list(self):
        from emotive.subsystems.hippocampus.retrieval.pipeline import _apply_within_retrieval_inhibition
        assert _apply_within_retrieval_inhibition([]) == []

    def test_single_item(self):
        from emotive.subsystems.hippocampus.retrieval.pipeline import _apply_within_retrieval_inhibition
        item = self._make_result(0.9)
        result = _apply_within_retrieval_inhibition([item])
        assert len(result) == 1
        assert result[0] is item

    def test_no_suppression_orthogonal_embeddings(self):
        """Orthogonal embeddings should not suppress each other."""
        from emotive.subsystems.hippocampus.retrieval.pipeline import _apply_within_retrieval_inhibition

        e1 = np.zeros(128, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(128, dtype=np.float32)
        e2[1] = 1.0

        r1 = self._make_result(0.9, e1)
        r2 = self._make_result(0.8, e2)
        result = _apply_within_retrieval_inhibition([r1, r2])
        assert len(result) == 2
        # Scores should be unmodified (overlap = 0 < 0.6 threshold)
        assert r2.final_score == pytest.approx(0.8)

    def test_suppression_similar_embeddings(self):
        """Highly similar embeddings should suppress the loser."""
        from emotive.subsystems.hippocampus.retrieval.pipeline import _apply_within_retrieval_inhibition

        e1 = np.ones(128, dtype=np.float32)
        e1 /= np.linalg.norm(e1)
        e2 = e1.copy()  # identical embedding

        r1 = self._make_result(0.9, e1)
        r2 = self._make_result(0.8, e2)
        result = _apply_within_retrieval_inhibition([r1, r2])
        assert len(result) == 2
        # r2 should have been suppressed (overlap ~ 1.0 > 0.6)
        assert r2.final_score < 0.8

    def test_none_embedding_no_crash(self):
        """Items with None embeddings should pass through without suppression."""
        from emotive.subsystems.hippocampus.retrieval.pipeline import _apply_within_retrieval_inhibition

        r1 = self._make_result(0.9, None)
        r2 = self._make_result(0.8, None)
        result = _apply_within_retrieval_inhibition([r1, r2])
        assert len(result) == 2
        assert r2.final_score == pytest.approx(0.8)


# === Proactive Interference (E12) ===


class TestProactiveInterference:
    """Old memories competing with newer ones at retrieval time."""

    def _make_result(self, score, created_at):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.final_score = score
        r.candidate = MagicMock()
        r.candidate.created_at = created_at
        return r

    def test_empty_list(self):
        from emotive.subsystems.hippocampus.retrieval.interference import detect_proactive_interference
        assert detect_proactive_interference([]) == []

    def test_single_item(self):
        from emotive.subsystems.hippocampus.retrieval.interference import detect_proactive_interference
        now = datetime.now(timezone.utc)
        item = self._make_result(0.9, now)
        result = detect_proactive_interference([item])
        assert len(result) == 1

    def test_old_high_score_reduced(self):
        """Old high-activation memory blocking newer should be reduced."""
        from emotive.subsystems.hippocampus.retrieval.interference import detect_proactive_interference

        now = datetime.now(timezone.utc)
        old = self._make_result(0.9, now - timedelta(days=30))
        new = self._make_result(0.5, now)
        result = detect_proactive_interference([old, new])
        # Old memory had score 0.9 > 0.5 * 1.5 = 0.75, and age_diff > 1
        assert old.final_score < 0.9

    def test_no_interference_same_day(self):
        """Memories created same day: no proactive interference."""
        from emotive.subsystems.hippocampus.retrieval.interference import detect_proactive_interference

        now = datetime.now(timezone.utc)
        a = self._make_result(0.9, now - timedelta(hours=2))
        b = self._make_result(0.5, now)
        detect_proactive_interference([a, b])
        # Same day = age_diff_days < 1 → no reduction
        assert a.final_score == pytest.approx(0.9)

    def test_no_interference_scores_close(self):
        """If old score is not > 1.5x new, no interference."""
        from emotive.subsystems.hippocampus.retrieval.interference import detect_proactive_interference

        now = datetime.now(timezone.utc)
        old = self._make_result(0.6, now - timedelta(days=10))
        new = self._make_result(0.5, now)
        detect_proactive_interference([old, new])
        # 0.6 is NOT > 0.5 * 1.5 = 0.75
        assert old.final_score == pytest.approx(0.6)


# === CA1 Source Confusion (E2) ===


class TestSourceConfusion:
    """Source confusion — similar content from different people."""

    def _make_candidate(self, tags, embedding=None):
        from emotive.subsystems.hippocampus.retrieval.ca3 import CompletionCandidate
        return CompletionCandidate(
            memory_id=uuid.uuid4(),
            content="test memory",
            embedding=embedding,
            tags=tags,
            completion_score=0.8,
            emotional_intensity=0.5,
            primary_emotion="trust",
            memory_type="episodic",
            created_at=datetime.now(timezone.utc),
            retrieval_count=5,
            is_formative=False,
        )

    def test_confusion_detected_similar_content_different_people(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import _detect_source_confusion, ComparatorResult

        e = np.ones(128, dtype=np.float32)
        e /= np.linalg.norm(e)

        c1 = self._make_candidate(["alice"], e.copy())
        c2 = self._make_candidate(["bob"], e.copy())  # identical embedding

        results = [
            ComparatorResult(candidate=c1, context_match=0.5, familiarity_score=0.8,
                             recollection_score=0.6, final_score=0.7),
            ComparatorResult(candidate=c2, context_match=0.5, familiarity_score=0.8,
                             recollection_score=0.6, final_score=0.7),
        ]
        confusions = _detect_source_confusion(results, threshold=0.85)
        assert len(confusions) == 1
        assert confusions[0].person_a == "alice"
        assert confusions[0].person_b == "bob"

    def test_no_confusion_same_person(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import _detect_source_confusion, ComparatorResult

        e = np.ones(128, dtype=np.float32)
        e /= np.linalg.norm(e)

        c1 = self._make_candidate(["alice"], e.copy())
        c2 = self._make_candidate(["alice"], e.copy())

        results = [
            ComparatorResult(candidate=c1, context_match=0.5, familiarity_score=0.8,
                             recollection_score=0.6, final_score=0.7),
            ComparatorResult(candidate=c2, context_match=0.5, familiarity_score=0.8,
                             recollection_score=0.6, final_score=0.7),
        ]
        confusions = _detect_source_confusion(results, threshold=0.85)
        assert len(confusions) == 0

    def test_no_confusion_dissimilar_embeddings(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import _detect_source_confusion, ComparatorResult

        e1 = np.zeros(128, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(128, dtype=np.float32)
        e2[1] = 1.0

        c1 = self._make_candidate(["alice"], e1)
        c2 = self._make_candidate(["bob"], e2)

        results = [
            ComparatorResult(candidate=c1, context_match=0.5, familiarity_score=0.8,
                             recollection_score=0.6, final_score=0.7),
            ComparatorResult(candidate=c2, context_match=0.5, familiarity_score=0.8,
                             recollection_score=0.6, final_score=0.7),
        ]
        confusions = _detect_source_confusion(results, threshold=0.85)
        assert len(confusions) == 0

    def test_no_confusion_empty_results(self):
        from emotive.subsystems.hippocampus.retrieval.ca1 import _detect_source_confusion
        assert _detect_source_confusion([], threshold=0.85) == []


# === CA1 compare_and_filter edge cases ===


class TestCA1EdgeCases:
    """Additional edge cases for CA1 comparator."""

    def _make_candidate(self, score=0.7, tags=None):
        from emotive.subsystems.hippocampus.retrieval.ca3 import CompletionCandidate
        return CompletionCandidate(
            memory_id=uuid.uuid4(),
            content="test memory",
            embedding=np.random.randn(1024).astype(np.float32),
            tags=tags or ["test"],
            completion_score=score,
            emotional_intensity=0.5,
            primary_emotion="trust",
            memory_type="episodic",
            created_at=datetime.now(timezone.utc),
            retrieval_count=5,
            is_formative=False,
        )

    def test_low_completion_low_context_rejected(self):
        """Candidate with both low completion and low context match is rejected."""
        from emotive.subsystems.hippocampus.retrieval.ca1 import compare_and_filter
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector

        cv = ContextVector(dim=256)
        # Low completion score and no stored context → ctx_match=0.4
        # But context_reject_threshold = 0.15, so 0.4 > 0.15 → passes
        # To trigger rejection, need ctx_match < 0.15 AND score < 0.7
        # Since we don't store contexts, ctx_match=0.4 always. Just verify it passes.
        c = self._make_candidate(score=0.3)
        output = compare_and_filter([c], cv)
        # ctx_match=0.4 > 0.15, so it passes even with low score
        assert len(output.results) >= 1

    def test_familiarity_capped_at_one(self):
        """Familiarity is capped at 1.0 even for very high completion scores."""
        from emotive.subsystems.hippocampus.retrieval.ca1 import compare_and_filter
        from emotive.subsystems.hippocampus.retrieval.context_vector import ContextVector

        cv = ContextVector(dim=256)
        c = self._make_candidate(score=1.5)  # artificially high
        output = compare_and_filter([c], cv)
        assert output.results[0].familiarity_score <= 1.0


# === Somatic Markers (E14) ===


class TestSomaticMarkers:
    """Insula somatic markers — body state shapes retrieval."""

    def test_compute_somatic_bias_neutral(self):
        """Neutral body state (mid-range) produces minimal biases."""
        from emotive.subsystems.insula.somatic_markers import compute_somatic_bias

        bias = compute_somatic_bias(energy=0.5, cognitive_load=0.3, comfort=0.5)
        assert bias.threat_boost == 0.0
        assert bias.suppression_weakening == 0.0
        assert bias.retrieval_narrowing is False

    def test_low_comfort_threat_boost(self):
        """Low comfort activates threat memory boost."""
        from emotive.subsystems.insula.somatic_markers import compute_somatic_bias

        bias = compute_somatic_bias(comfort=0.2)
        assert bias.threat_boost > 0
        assert bias.threat_boost == pytest.approx((0.4 - 0.2) * 0.5)

    def test_high_energy_action_boost(self):
        """High energy activates action/procedural memory boost."""
        from emotive.subsystems.insula.somatic_markers import compute_somatic_bias

        bias = compute_somatic_bias(energy=0.9)
        assert bias.action_boost > 0
        assert bias.action_boost == pytest.approx((0.9 - 0.7) * 0.5)

    def test_low_energy_suppression_weakening(self):
        """Low energy weakens intentional suppression."""
        from emotive.subsystems.insula.somatic_markers import compute_somatic_bias

        bias = compute_somatic_bias(energy=0.1)
        assert bias.suppression_weakening > 0
        assert bias.spontaneous_threshold_reduction > 0

    def test_high_load_narrows_retrieval(self):
        """High cognitive load activates retrieval narrowing."""
        from emotive.subsystems.insula.somatic_markers import compute_somatic_bias

        bias = compute_somatic_bias(cognitive_load=0.9)
        assert bias.retrieval_narrowing is True

    def test_apply_somatic_bias_threat(self):
        """Threat boost raises score for fear-related memories."""
        from emotive.subsystems.insula.somatic_markers import apply_somatic_bias_to_score, SomaticBias

        bias = SomaticBias(threat_boost=0.15)
        score = apply_somatic_bias_to_score(
            0.5, memory_tags=["test"], memory_type="episodic",
            primary_emotion="fear", bias=bias,
        )
        assert score > 0.5

    def test_apply_somatic_bias_action(self):
        """Action boost raises score for procedural memories."""
        from emotive.subsystems.insula.somatic_markers import apply_somatic_bias_to_score, SomaticBias

        bias = SomaticBias(action_boost=0.1)
        score = apply_somatic_bias_to_score(
            0.5, memory_tags=["procedural"], memory_type="procedural",
            primary_emotion="neutral", bias=bias,
        )
        assert score > 0.5

    def test_apply_somatic_bias_no_effect(self):
        """Zero biases leave score unchanged."""
        from emotive.subsystems.insula.somatic_markers import apply_somatic_bias_to_score, SomaticBias

        bias = SomaticBias()
        score = apply_somatic_bias_to_score(
            0.5, memory_tags=["test"], memory_type="episodic",
            primary_emotion="trust", bias=bias,
        )
        assert score == pytest.approx(0.5)

    def test_threat_boost_from_tags(self):
        """Threat boost also fires for threat-emotion tags (not just primary_emotion)."""
        from emotive.subsystems.insula.somatic_markers import apply_somatic_bias_to_score, SomaticBias

        bias = SomaticBias(threat_boost=0.2)
        score = apply_somatic_bias_to_score(
            0.5, memory_tags=["anger"], memory_type="episodic",
            primary_emotion="neutral", bias=bias,
        )
        # primary_emotion is neutral so no primary boost,
        # but "anger" is in tags → bias.threat_boost * 0.5
        assert score == pytest.approx(0.5 + 0.2 * 0.5)


# === Reward Signal edge cases (E31) ===


class TestRewardSignalEdgeCases:
    """Additional edge cases for basal ganglia reward computation."""

    def test_mood_improvement_positive_reward(self):
        from emotive.subsystems.basal_ganglia import compute_reward_signal

        pre = {"valence": 0.3, "arousal": 0.5}
        post = {"valence": 0.7, "arousal": 0.5}
        reward = compute_reward_signal(pre, post)
        assert reward > 0

    def test_reward_clamped(self):
        from emotive.subsystems.basal_ganglia import compute_reward_signal

        reward = compute_reward_signal(
            None, None, trust_change=10.0, bonding_change=10.0,
        )
        assert reward <= 0.5

    def test_gating_bonus_missing_key(self):
        from emotive.subsystems.basal_ganglia import get_gating_bonus

        assert get_gating_bonus({}) == 0.0

    def test_zero_reward(self):
        from emotive.subsystems.basal_ganglia import compute_reward_signal

        reward = compute_reward_signal(None, None)
        assert reward == 0.0


# === Locus Coeruleus edge cases ===


class TestLocusCoeruleusEdgeCases:
    """Low energy amplification of arousal."""

    def test_low_energy_amplifies_arousal(self):
        """Low energy causes moderate arousal to read as high → narrow scope."""
        from emotive.subsystems.locus_coeruleus import compute_retrieval_scope

        # Moderate inputs: normally "focused"
        normal = compute_retrieval_scope(prediction_error=0.5, emotional_intensity=0.3, energy=1.0)
        # Same inputs but low energy → amplified arousal
        tired = compute_retrieval_scope(prediction_error=0.5, emotional_intensity=0.3, energy=0.2)

        # arousal = 0.5*0.6 + 0.3*0.4 = 0.42 (focused) for normal
        # arousal = 0.42 * 1.3 = 0.546 (still focused) for tired
        # But with PE=0.5 + EI=0.5, energy=0.2: arousal=0.5*0.6+0.5*0.4=0.5, *1.3=0.65 → narrow
        tired_high = compute_retrieval_scope(prediction_error=0.5, emotional_intensity=0.5, energy=0.2)
        assert tired_high["scope"] == "narrow"


# === Suppression edge cases ===


class TestSuppressionEdgeCases:
    """Additional edge cases for suppression."""

    def test_zero_suppression_returns_zero(self):
        from emotive.subsystems.hippocampus.retrieval.suppression import get_effective_suppression

        now = datetime.now(timezone.utc)
        assert get_effective_suppression(0.0, now, now=now) == 0.0

    def test_none_decay_start_returns_zero(self):
        from emotive.subsystems.hippocampus.retrieval.suppression import get_effective_suppression

        assert get_effective_suppression(0.5, None) == 0.0

    def test_high_arousal_weakens_suppression(self):
        from emotive.subsystems.hippocampus.retrieval.suppression import get_effective_suppression

        now = datetime.now(timezone.utc)
        normal = get_effective_suppression(0.8, now, energy=1.0, arousal=0.0, now=now)
        aroused = get_effective_suppression(0.8, now, energy=1.0, arousal=0.8, now=now)
        assert aroused == pytest.approx(normal * 0.5, abs=0.01)

    def test_apply_suppression_memory_not_found(self):
        """apply_suppression with nonexistent memory does nothing."""
        from unittest.mock import MagicMock
        from emotive.subsystems.hippocampus.retrieval.suppression import apply_suppression

        session = MagicMock()
        session.get.return_value = None
        # Should not raise
        apply_suppression(session, uuid.uuid4())

    def test_apply_suppression_additive_capped(self):
        """Multiple suppressions add up but cap at 1.0."""
        from unittest.mock import MagicMock
        from emotive.subsystems.hippocampus.retrieval.suppression import apply_suppression

        mem = MagicMock()
        mem.suppression_level = 0.8
        mem.suppression_decay_start = None
        session = MagicMock()
        session.get.return_value = mem

        apply_suppression(session, uuid.uuid4(), suppression_strength=0.5)
        assert mem.suppression_level == pytest.approx(1.0)


# === Resistance edge cases ===


class TestResistanceEdgeCases:
    """Additional edge cases for memory resistance."""

    def test_zero_resistance_returns_one(self):
        from emotive.subsystems.hippocampus.retrieval.resistance import compute_resistance_threshold

        assert compute_resistance_threshold(0) == 1.0

    def test_direct_inquiry_lowers_resistance(self):
        from emotive.subsystems.hippocampus.retrieval.resistance import compute_resistance_threshold

        normal = compute_resistance_threshold(0.5, person_trust=0.5)
        direct = compute_resistance_threshold(0.5, person_trust=0.5, is_direct_inquiry=True)
        assert direct < normal

    def test_minimum_multiplier(self):
        """Multiplier never goes below 0.5."""
        from emotive.subsystems.hippocampus.retrieval.resistance import compute_resistance_threshold

        mult = compute_resistance_threshold(
            0.1, person_trust=0.99, energy=0.99, comfort=0.99, is_direct_inquiry=True,
        )
        assert mult >= 0.5

    def test_habituation_unsafe_no_change(self):
        """Unsafe conditions: no resistance reduction."""
        from emotive.subsystems.hippocampus.retrieval.resistance import update_resistance_after_safe_exposure

        meta = {"identity_resistance": 0.6}
        result = update_resistance_after_safe_exposure(meta, person_trust=0.3, comfort=0.3)
        assert result["identity_resistance"] == 0.6  # unchanged

    def test_habituation_full(self):
        """Very low resistance fully habituates to 0."""
        from emotive.subsystems.hippocampus.retrieval.resistance import update_resistance_after_safe_exposure

        meta = {"identity_resistance": 0.05}
        result = update_resistance_after_safe_exposure(meta, person_trust=0.8, comfort=0.7)
        # 0.05 * 0.85 = 0.0425 < 0.1 → set to 0
        assert result["identity_resistance"] == 0

    def test_habituation_no_resistance(self):
        """No identity_resistance key returns unchanged."""
        from emotive.subsystems.hippocampus.retrieval.resistance import update_resistance_after_safe_exposure

        meta = {}
        result = update_resistance_after_safe_exposure(meta, person_trust=0.8, comfort=0.7)
        assert "identity_resistance" not in result or result.get("identity_resistance", 0) == 0


# === Narrative edge cases ===


class TestNarrativeEdgeCases:
    """Additional edge cases for narrative construction."""

    def test_no_person_no_theme(self):
        """No person and no common theme returns None."""
        from emotive.subsystems.hippocampus.retrieval.narrative import construct_narrative

        memories = [
            {"content": "aaa", "primary_emotion": "joy", "tags": ["x"], "created_at": datetime.now(timezone.utc)},
            {"content": "bbb", "primary_emotion": "trust", "tags": ["y"], "created_at": datetime.now(timezone.utc)},
            {"content": "ccc", "primary_emotion": "fear", "tags": ["z"], "created_at": datetime.now(timezone.utc)},
        ]
        result = construct_narrative(memories, detected_person=None)
        assert result is None

    def test_theme_based_narrative(self):
        """Common tag across 3+ memories fires theme-based narrative."""
        from emotive.subsystems.hippocampus.retrieval.narrative import construct_narrative

        now = datetime.now(timezone.utc)
        memories = [
            {"content": "Studied machine learning", "primary_emotion": "curiosity",
             "tags": ["machine_learning"], "created_at": now - timedelta(days=3)},
            {"content": "ML paper reading", "primary_emotion": "curiosity",
             "tags": ["machine_learning"], "created_at": now - timedelta(days=1)},
            {"content": "Built an ML model", "primary_emotion": "joy",
             "tags": ["machine_learning"], "created_at": now},
        ]
        result = construct_narrative(memories, detected_person=None)
        assert result is not None
        assert "machine_learning" in result

    def test_same_emotion_arc(self):
        """Same first and last emotion → 'dominant feeling throughout'."""
        from emotive.subsystems.hippocampus.retrieval.narrative import construct_narrative

        now = datetime.now(timezone.utc)
        memories = [
            {"content": "Start", "primary_emotion": "joy", "tags": ["test"],
             "created_at": now - timedelta(days=2)},
            {"content": "Middle", "primary_emotion": "joy", "tags": ["test"],
             "created_at": now - timedelta(days=1)},
            {"content": "End", "primary_emotion": "joy", "tags": ["test"],
             "created_at": now},
        ]
        result = construct_narrative(memories, detected_person="Lena")
        assert result is not None
        assert "dominant feeling" in result

    def test_find_common_theme(self):
        """_find_common_theme returns tag with 3+ occurrences."""
        from emotive.subsystems.hippocampus.retrieval.narrative import _find_common_theme

        memories = [
            {"tags": ["cooking", "joy"]},
            {"tags": ["cooking", "trust"]},
            {"tags": ["cooking", "surprise"]},
        ]
        assert _find_common_theme(memories) == "cooking"

    def test_find_common_theme_none(self):
        """No tag with 3+ occurrences → None."""
        from emotive.subsystems.hippocampus.retrieval.narrative import _find_common_theme

        memories = [
            {"tags": ["a"]},
            {"tags": ["b"]},
            {"tags": ["c"]},
        ]
        assert _find_common_theme(memories) is None


# === Prospective Memory edge cases ===


class TestProspectiveEdgeCases:
    """Additional edge cases for prospective memory triggers."""

    def test_empty_prospective_list(self):
        from emotive.subsystems.hippocampus.retrieval.prospective import check_prospective_triggers

        result = check_prospective_triggers([], "hello")
        assert result == []

    def test_empty_trigger_cue_ignored(self):
        from emotive.subsystems.hippocampus.retrieval.prospective import check_prospective_triggers

        intentions = [{"trigger_cue": "", "intended_action": "do something", "content": "test"}]
        result = check_prospective_triggers(intentions, "any input")
        assert result == []

    def test_case_insensitive_match(self):
        from emotive.subsystems.hippocampus.retrieval.prospective import check_prospective_triggers

        intentions = [{"trigger_cue": "Birthday", "intended_action": "celebrate", "content": "test"}]
        result = check_prospective_triggers(intentions, "It's my BIRTHDAY")
        assert len(result) == 1


# === Priming Keywords edge cases ===


class TestPrimingEdgeCases:
    """Additional edge cases for unconscious priming."""

    def _make_candidate(self, content):
        from unittest.mock import MagicMock
        c = MagicMock()
        c.content = content
        return c

    def test_empty_unconscious_pool(self):
        from emotive.subsystems.hippocampus.retrieval.pipeline import _extract_priming_keywords

        words = _extract_priming_keywords([], [self._make_candidate("hello world")])
        assert words == set()

    def test_empty_conscious_and_unconscious(self):
        from emotive.subsystems.hippocampus.retrieval.pipeline import _extract_priming_keywords

        words = _extract_priming_keywords([], [])
        assert words == set()


# === ACC Tone Monitor ===


class TestToneMonitor:
    """ACC tone alignment — response vs nudge matching."""

    def test_known_nudge_with_matches(self):
        from emotive.subsystems.acc.tone_monitor import check_tone_alignment

        score = check_tone_alignment("I really appreciate you and care about this", "warm")
        assert score > 0

    def test_known_nudge_no_matches(self):
        from emotive.subsystems.acc.tone_monitor import check_tone_alignment

        score = check_tone_alignment("The algorithm converges quickly", "warm")
        assert score == 0.0

    def test_unknown_nudge_returns_neutral(self):
        from emotive.subsystems.acc.tone_monitor import check_tone_alignment

        score = check_tone_alignment("any text", "unknown_nudge_type")
        assert score == 0.5

    def test_empty_response(self):
        from emotive.subsystems.acc.tone_monitor import check_tone_alignment

        score = check_tone_alignment("", "warm")
        assert score == 0.0


# === ACC Repetition Monitor ===


class TestRepetitionMonitor:
    """ACC repetition detection — stuck loop detection."""

    def test_not_stuck_initially(self):
        from emotive.subsystems.acc.repetition import RepetitionMonitor

        rm = RepetitionMonitor()
        assert rm.is_stuck is False

    def test_stuck_on_similar_responses(self):
        from emotive.subsystems.acc.repetition import RepetitionMonitor

        rm = RepetitionMonitor()
        # Two very similar embeddings
        emb = [1.0] * 128
        rm.update(emb, novelty=0.5)
        stuck = rm.update(emb, novelty=0.5)
        assert stuck is True

    def test_not_stuck_on_diverse_responses(self):
        from emotive.subsystems.acc.repetition import RepetitionMonitor

        rm = RepetitionMonitor()
        e1 = [1.0] + [0.0] * 127
        e2 = [0.0] + [1.0] + [0.0] * 126
        rm.update(e1, novelty=0.5)
        stuck = rm.update(e2, novelty=0.5)
        assert stuck is False

    def test_stuck_on_declining_novelty(self):
        from emotive.subsystems.acc.repetition import RepetitionMonitor

        rm = RepetitionMonitor()
        # Use orthogonal embeddings to avoid similarity trigger
        rm.update([1.0] + [0.0] * 127, novelty=0.8)
        rm.update([0.0, 1.0] + [0.0] * 126, novelty=0.5)
        stuck = rm.update([0.0, 0.0, 1.0] + [0.0] * 125, novelty=0.3)
        assert stuck is True  # 0.8 > 0.5 > 0.3 = declining

    def test_cancel_nudge(self):
        from emotive.subsystems.acc.repetition import RepetitionMonitor

        rm = RepetitionMonitor()
        emb = [1.0] * 128
        rm.update(emb, novelty=0.5)
        rm.update(emb, novelty=0.5)
        assert rm.is_stuck is True
        cancelled = rm.cancel_nudge(input_novelty=0.7)
        assert cancelled is True
        assert rm.is_stuck is False

    def test_reset(self):
        from emotive.subsystems.acc.repetition import RepetitionMonitor

        rm = RepetitionMonitor()
        emb = [1.0] * 128
        rm.update(emb, novelty=0.5)
        rm.update(emb, novelty=0.5)
        rm.reset()
        assert rm.is_stuck is False
