"""Tests for semantic memory."""

from emotive.db.models.memory import MemoryLink
from emotive.memory.episodic import store_episodic
from emotive.memory.semantic import (
    SEMANTIC_DECAY_RATE,
    _find_common_tags,
    extract_semantic_from_cluster,
    store_semantic,
)


def test_store_semantic_sets_type(db_session, embedding_service):
    mem = store_semantic(db_session, embedding_service, content="pattern test")
    assert mem.memory_type == "semantic"


def test_store_semantic_sets_decay_rate(db_session, embedding_service):
    mem = store_semantic(db_session, embedding_service, content="decay test")
    assert mem.decay_rate == SEMANTIC_DECAY_RATE


def test_store_semantic_creates_links_to_sources(db_session, embedding_service):
    src1 = store_episodic(db_session, embedding_service, content="source one")
    src2 = store_episodic(db_session, embedding_service, content="source two")
    db_session.flush()

    mem = store_semantic(
        db_session, embedding_service,
        content="pattern from sources",
        source_memory_ids=[src1.id, src2.id],
    )
    db_session.flush()

    links = db_session.query(MemoryLink).filter(
        MemoryLink.target_memory_id == mem.id
    ).all()
    assert len(links) == 2


def test_store_semantic_sets_confidence(db_session, embedding_service):
    mem = store_semantic(
        db_session, embedding_service,
        content="confidence test", confidence=0.85,
    )
    assert mem.confidence == 0.85


def test_extract_semantic_from_cluster_creates_pattern(db_session, embedding_service):
    mems = []
    for i in range(3):
        m = store_episodic(
            db_session, embedding_service,
            content=f"Cooking Italian pasta recipe variation {i}",
        )
        mems.append(m)
    db_session.flush()

    result = extract_semantic_from_cluster(db_session, embedding_service, mems)
    assert result is not None
    assert result.memory_type == "semantic"
    assert "Pattern from 3" in result.content


def test_extract_semantic_from_cluster_too_small(db_session, embedding_service):
    m = store_episodic(db_session, embedding_service, content="lonely memory")
    result = extract_semantic_from_cluster(db_session, embedding_service, [m])
    assert result is None


def test_find_common_tags_majority():
    # Create mock-like objects with tags
    class FakeMem:
        def __init__(self, tags):
            self.tags = tags

    mems = [FakeMem(["a", "b"]), FakeMem(["a", "c"]), FakeMem(["a", "b"])]
    common = _find_common_tags(mems)
    assert "a" in common  # appears in 3/3
    assert "b" in common  # appears in 2/3
    assert "c" not in common  # appears in 1/3


def test_find_common_tags_empty():
    assert _find_common_tags([]) == []
