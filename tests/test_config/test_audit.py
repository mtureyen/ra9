"""Tests for config audit trail."""

from emotive.config.audit import _to_jsonb, audit_config_change


def test_audit_config_change_creates_row(db_session):
    entry = audit_config_change(db_session, "test.key", "old", "new")
    db_session.flush()
    assert entry.config_key == "test.key"
    assert entry.new_value == {"value": "new"}


def test_audit_config_change_with_reason(db_session):
    entry = audit_config_change(db_session, "k", "a", "b", reason="testing")
    db_session.flush()
    assert entry.reason == "testing"


def test_audit_config_change_none_old_value(db_session):
    entry = audit_config_change(db_session, "k", None, "new")
    db_session.flush()
    assert entry.old_value is None


def test_to_jsonb_wraps_scalars():
    assert _to_jsonb(0.5) == {"value": 0.5}
    assert _to_jsonb("hello") == {"value": "hello"}


def test_to_jsonb_passes_dicts():
    assert _to_jsonb({"a": 1}) == {"a": 1}


def test_to_jsonb_none():
    assert _to_jsonb(None) is None
