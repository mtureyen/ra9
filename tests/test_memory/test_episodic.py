"""Tests for episodic memory."""

import uuid

from emotive.memory.episodic import EPISODIC_DECAY_RATE, store_episodic


def test_store_episodic_sets_type(db_session, embedding_service):
    mem = store_episodic(db_session, embedding_service, content="episode test")
    assert mem.memory_type == "episodic"


def test_store_episodic_sets_decay_rate(db_session, embedding_service):
    mem = store_episodic(db_session, embedding_service, content="decay rate test")
    assert mem.decay_rate == EPISODIC_DECAY_RATE


def test_store_episodic_with_context(db_session, embedding_service):
    mem = store_episodic(
        db_session, embedding_service,
        content="context test", context={"participant": "user123"},
    )
    assert mem.metadata_["participant"] == "user123"


def test_store_episodic_with_tags(db_session, embedding_service):
    mem = store_episodic(
        db_session, embedding_service,
        content="tags test", tags=["important", "social"],
    )
    assert "important" in mem.tags
    assert "social" in mem.tags


def test_store_episodic_with_conversation_id(db_session, embedding_service):
    cid = uuid.uuid4()
    mem = store_episodic(
        db_session, embedding_service,
        content="conv test", conversation_id=cid,
    )
    assert mem.conversation_id == cid
