"""Tests for raw SQL memory queries."""

import uuid
from datetime import datetime, timedelta, timezone

from emotive.db.models.memory import Memory, MemoryLink
from emotive.db.queries.memory_queries import (
    apply_decay,
    apply_interference,
    compute_recency_weight,
    create_memory_link,
    find_similar_memories,
    get_linked_memories,
    link_by_conversation,
    rank_memories,
    search_by_embedding,
    strengthen_link,
)


def _make_memory(session, embedding_service, content, **kwargs):
    m = Memory(
        memory_type=kwargs.get("memory_type", "episodic"),
        content=content,
        embedding=embedding_service.embed_text(content),
        is_archived=kwargs.get("is_archived", False),
    )
    session.add(m)
    session.flush()
    return m


def test_search_by_embedding_returns_results(db_session, embedding_service):
    m = _make_memory(db_session, embedding_service, "dogs playing in the park")
    emb = embedding_service.embed_text("dogs at the park")
    results = search_by_embedding(db_session, emb, limit=5)
    ids = [r["id"] for r in results]
    assert m.id in ids
    assert results[0]["similarity"] > 0.5


def test_search_by_embedding_excludes_archived(db_session, embedding_service):
    m = _make_memory(db_session, embedding_service, "archived memory", is_archived=True)
    emb = embedding_service.embed_text("archived memory")
    results = search_by_embedding(db_session, emb, limit=5)
    ids = [r["id"] for r in results]
    assert m.id not in ids


def test_search_by_embedding_filters_by_type(db_session, embedding_service):
    _make_memory(db_session, embedding_service, "episodic content", memory_type="episodic")
    _make_memory(db_session, embedding_service, "semantic pattern", memory_type="semantic")
    emb = embedding_service.embed_text("semantic pattern")
    results = search_by_embedding(db_session, emb, limit=5, memory_type="semantic")
    types = [r["memory_type"] for r in results]
    assert all(t == "semantic" for t in types)


def test_search_by_embedding_respects_limit(db_session, embedding_service):
    for i in range(5):
        _make_memory(db_session, embedding_service, f"memory number {i}")
    emb = embedding_service.embed_text("memory")
    results = search_by_embedding(db_session, emb, limit=2)
    assert len(results) <= 2


def test_get_linked_memories_single_hop(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "memory A")
    b = _make_memory(db_session, embedding_service, "memory B")
    create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.8)
    db_session.flush()

    activations = get_linked_memories(db_session, [a.id], hops=1, decay_per_hop=0.6)
    assert b.id in activations
    assert activations[b.id] == pytest.approx(0.8 * 0.6, abs=0.01)


def test_get_linked_memories_two_hops(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "hop A")
    b = _make_memory(db_session, embedding_service, "hop B")
    c = _make_memory(db_session, embedding_service, "hop C")
    create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.8)
    create_memory_link(db_session, b.id, c.id, "semantic_similarity", 0.7)
    db_session.flush()

    activations = get_linked_memories(db_session, [a.id], hops=2, decay_per_hop=0.6)
    assert b.id in activations
    assert c.id in activations
    assert activations[c.id] < activations[b.id]


def test_get_linked_memories_excludes_seeds(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "seed mem")
    b = _make_memory(db_session, embedding_service, "linked mem")
    create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.8)
    db_session.flush()

    activations = get_linked_memories(db_session, [a.id], hops=1)
    assert a.id not in activations


def test_get_linked_memories_empty_input(db_session):
    assert get_linked_memories(db_session, []) == {}


def test_compute_recency_weight_recent():
    w = compute_recency_weight(datetime.now(timezone.utc))
    assert w > 0.9


def test_compute_recency_weight_old():
    old = datetime.now(timezone.utc) - timedelta(days=100)
    w = compute_recency_weight(old)
    assert w < 0.1


def test_rank_memories_significance_boosts_ranking():
    now = datetime.now(timezone.utc)
    candidates = [
        {
            "id": uuid.uuid4(), "similarity": 0.7, "created_at": now,
            "metadata": {"significance": 0.9},
        },
        {
            "id": uuid.uuid4(), "similarity": 0.7, "created_at": now,
            "metadata": {"significance": 0.1},
        },
    ]
    ranked = rank_memories(candidates, {})
    # Same similarity, but high significance should rank first
    assert ranked[0]["significance"] == 0.9
    assert ranked[1]["significance"] == 0.1
    assert ranked[0]["final_rank"] > ranked[1]["final_rank"]


def test_rank_memories_default_significance_when_missing():
    now = datetime.now(timezone.utc)
    candidates = [
        {"id": uuid.uuid4(), "similarity": 0.7, "created_at": now, "metadata": {}},
    ]
    ranked = rank_memories(candidates, {})
    assert ranked[0]["significance"] == 0.5  # default


def test_rank_memories_orders_by_final_rank():
    candidates = [
        {"id": uuid.uuid4(), "similarity": 0.5, "created_at": datetime.now(timezone.utc)},
        {"id": uuid.uuid4(), "similarity": 0.9, "created_at": datetime.now(timezone.utc)},
        {"id": uuid.uuid4(), "similarity": 0.3, "created_at": datetime.now(timezone.utc)},
    ]
    ranked = rank_memories(candidates, {})
    scores = [r["final_rank"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_apply_decay_updates_detail_retention(db_session, embedding_service):
    _make_memory(db_session, embedding_service, "decay test")
    db_session.flush()
    result = apply_decay(db_session, archive_threshold=0.01)
    assert result["memories_decayed"] >= 1


def test_create_memory_link_success(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "link source")
    b = _make_memory(db_session, embedding_service, "link target")
    link_id = create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.7)
    assert link_id is not None


def test_create_memory_link_duplicate_returns_none(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "dup source")
    b = _make_memory(db_session, embedding_service, "dup target")
    create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.7)
    db_session.flush()
    result = create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.9)
    assert result is None


def test_find_similar_memories_above_threshold(db_session, embedding_service):
    _make_memory(db_session, embedding_service, "cats playing with yarn")
    _make_memory(db_session, embedding_service, "kittens playing with string")
    db_session.flush()
    emb = embedding_service.embed_text("cats playing with yarn")
    results = find_similar_memories(db_session, emb, threshold=0.5)
    assert len(results) >= 1


import pytest  # noqa: E402 (needed for approx)

# --- Phase 0.5: Brain-closer query tests ---


def test_strengthen_link_creates_if_not_exists(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "strengthen source")
    b = _make_memory(db_session, embedding_service, "strengthen target")
    db_session.flush()

    strengthen_link(db_session, a.id, b.id, "semantic_similarity", boost=0.3)
    db_session.flush()

    link = db_session.query(MemoryLink).filter(
        MemoryLink.source_memory_id == a.id,
        MemoryLink.target_memory_id == b.id,
    ).first()
    assert link is not None
    assert link.strength == pytest.approx(0.3, abs=0.01)


def test_strengthen_link_increments_existing(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "boost source")
    b = _make_memory(db_session, embedding_service, "boost target")
    create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.5)
    db_session.flush()

    strengthen_link(db_session, a.id, b.id, "semantic_similarity", boost=0.1)
    db_session.flush()

    link = db_session.query(MemoryLink).filter(
        MemoryLink.source_memory_id == a.id,
        MemoryLink.target_memory_id == b.id,
    ).first()
    assert link.strength == pytest.approx(0.6, abs=0.01)


def test_strengthen_link_caps_at_one(db_session, embedding_service):
    a = _make_memory(db_session, embedding_service, "cap source")
    b = _make_memory(db_session, embedding_service, "cap target")
    create_memory_link(db_session, a.id, b.id, "semantic_similarity", 0.95)
    db_session.flush()

    strengthen_link(db_session, a.id, b.id, "semantic_similarity", boost=0.2)
    db_session.flush()

    link = db_session.query(MemoryLink).filter(
        MemoryLink.source_memory_id == a.id,
        MemoryLink.target_memory_id == b.id,
    ).first()
    assert link.strength == pytest.approx(1.0, abs=0.01)


def test_apply_interference_weakens_similar(db_session, embedding_service):
    old = _make_memory(db_session, embedding_service, "cats playing with toys")
    db_session.flush()
    assert old.detail_retention == 1.0

    # Store something very similar
    new = _make_memory(db_session, embedding_service, "cats playing with toys in the garden")
    db_session.flush()

    interfered = apply_interference(
        db_session, list(new.embedding), new.id,
        similarity_threshold=0.7, interference_strength=0.05,
    )
    db_session.flush()
    db_session.refresh(old)

    assert interfered >= 1
    assert old.detail_retention < 1.0


def test_apply_interference_ignores_dissimilar(db_session, embedding_service):
    old = _make_memory(db_session, embedding_service, "quantum physics lecture")
    db_session.flush()

    new = _make_memory(db_session, embedding_service, "chocolate cake recipe")
    db_session.flush()

    apply_interference(
        db_session, list(new.embedding), new.id,
        similarity_threshold=0.85, interference_strength=0.05,
    )
    db_session.flush()
    db_session.refresh(old)

    assert old.detail_retention == 1.0


def test_link_by_conversation_links_same_session(db_session, embedding_service):
    conv_id = uuid.uuid4()
    a = _make_memory(db_session, embedding_service, "conv mem 1")
    a.conversation_id = conv_id
    b = _make_memory(db_session, embedding_service, "conv mem 2")
    b.conversation_id = conv_id
    c = _make_memory(db_session, embedding_service, "conv mem 3")
    c.conversation_id = conv_id
    db_session.flush()

    created = link_by_conversation(db_session, conv_id, strength=0.4)
    db_session.flush()

    assert created >= 2  # at least 1-2, 2-3 links


def test_link_by_conversation_no_links_for_single_memory(db_session, embedding_service):
    conv_id = uuid.uuid4()
    a = _make_memory(db_session, embedding_service, "lonely conv mem")
    a.conversation_id = conv_id
    db_session.flush()

    created = link_by_conversation(db_session, conv_id)
    assert created == 0
