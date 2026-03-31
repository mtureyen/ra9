"""Tests for simple person trust (Design Decision #13)."""

from emotive.memory.base import store_memory
from emotive.memory.identity import compute_person_trust


class TestPersonTrust:
    def test_unknown_person(self, db_session):
        trust = compute_person_trust(db_session, "xyzzy_nobody")
        assert trust == "unknown"

    def test_familiar_person(self, db_session, embedding_service):
        for i in range(3):
            store_memory(
                db_session, embedding_service,
                content=f"TestPerson123 said something interesting number {i}",
                memory_type="episodic",
            )
        db_session.flush()
        trust = compute_person_trust(db_session, "testperson123")
        assert trust in ("familiar", "trusted")

    def test_trusted_person_many_mentions(self, db_session, embedding_service):
        for i in range(8):
            store_memory(
                db_session, embedding_service,
                content=f"TrustTestPerson is a great friend, memory {i}",
                memory_type="episodic",
            )
        db_session.flush()
        trust = compute_person_trust(db_session, "trusttestperson")
        assert trust in ("trusted", "core")

    def test_mertcan_is_trusted_in_real_db(self, db_session):
        """Mertcan should be trusted or core in the real DB."""
        trust = compute_person_trust(db_session, "mertcan")
        assert trust in ("trusted", "core")

    def test_case_insensitive(self, db_session, embedding_service):
        store_memory(
            db_session, embedding_service,
            content="CasePerson is mentioned here",
            memory_type="episodic",
        )
        db_session.flush()
        trust_lower = compute_person_trust(db_session, "caseperson")
        trust_upper = compute_person_trust(db_session, "CasePerson")
        assert trust_lower == trust_upper
