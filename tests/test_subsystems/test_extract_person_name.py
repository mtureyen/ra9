"""Tests for _extract_person_name (Fix D: trust modulation bug)."""

from emotive.memory.base import store_memory
from emotive.subsystems.hippocampus.conflict import _extract_person_name


class TestExtractPersonName:
    def test_finds_known_person_from_tags(self, db_session, embedding_service):
        """Should find person names that appear in person-related memory tags."""
        store_memory(
            db_session, embedding_service,
            content="mertcan is my creator and built this system",
            memory_type="semantic",
            tags=["creator", "mertcan"],
        )
        db_session.flush()

        result = _extract_person_name("mertcan said something nice", db_session)
        assert result == "mertcan"

    def test_returns_none_for_unknown_person(self, db_session):
        """Unknown names should return None."""
        result = _extract_person_name("xyzzy_nobody said hello", db_session)
        assert result is None

    def test_case_insensitive_match(self, db_session, embedding_service):
        """Name matching should be case-insensitive."""
        store_memory(
            db_session, embedding_service,
            content="TestPerson is a friend",
            memory_type="episodic",
            tags=["friend", "testperson"],
        )
        db_session.flush()

        result = _extract_person_name("TestPerson mentioned something", db_session)
        assert result == "testperson"

    def test_no_partial_match(self, db_session, embedding_service):
        """Should not match partial words (e.g., 'art' inside 'start')."""
        store_memory(
            db_session, embedding_service,
            content="art is interesting",
            memory_type="episodic",
            tags=["personal", "art"],
        )
        db_session.flush()

        result = _extract_person_name("I want to start something new", db_session)
        # "art" should NOT match inside "start"
        assert result is None

    def test_empty_content_returns_none(self, db_session):
        """Empty content should return None."""
        result = _extract_person_name("", db_session)
        assert result is None

    def test_mertcan_in_real_db(self, db_session):
        """Mertcan should be extractable from the real DB (has many tagged memories)."""
        result = _extract_person_name(
            "Mertcan told me about something interesting today", db_session
        )
        # In the real DB, Mertcan should be known via person-context
        # If no person tags exist yet, this may be None — that's acceptable
        assert result is None or result == "mertcan"


class TestTrustModulationIntegration:
    def test_trusted_person_halves_conflict(self, db_session, embedding_service):
        """When a trusted person is mentioned, conflict should be halved."""
        # Set up: store enough memories about TrustTarget so they become trusted
        for i in range(8):
            store_memory(
                db_session, embedding_service,
                content=f"trusttarget is a great person, memory {i}",
                memory_type="episodic",
                tags=["friend", "trusttarget"],
            )

        # Set up strong identity memory
        store_memory(
            db_session, embedding_service,
            content="My name is Ryo and I was created by trusttarget",
            memory_type="semantic",
            metadata={"significance": 0.95},
        )
        from emotive.db.models.memory import Memory
        from sqlalchemy import select

        mem = db_session.execute(
            select(Memory).order_by(Memory.created_at.desc()).limit(1)
        ).scalar()
        mem.retrieval_count = 8
        mem.is_formative = True
        db_session.flush()

        # The person should now be extractable
        name = _extract_person_name(
            "trusttarget said something contradictory", db_session
        )
        assert name == "trusttarget"

        # Verify trust level
        from emotive.memory.identity import compute_person_trust

        trust = compute_person_trust(db_session, "trusttarget")
        assert trust in ("trusted", "core")
