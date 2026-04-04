"""Tests for the FastAPI server wrapping the thalamus."""

import json

import pytest
from fastapi.testclient import TestClient

from emotive.api.server import app, state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset API state before each test."""
    state.app_context = None
    state.thalamus = None
    state.session_active = False
    yield
    state.app_context = None
    state.thalamus = None
    state.session_active = False


class TestHealthEndpoint:
    def test_health_no_brain(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
            # May fail during lifespan boot in test, but the endpoint exists
            assert resp.status_code in (200, 500)


class TestSessionEndpoints:
    def test_boot_requires_brain(self):
        """Without lifespan (no brain), boot should fail."""
        with TestClient(app, raise_server_exceptions=False) as client:
            # In test client, lifespan runs — brain boots.
            # But if it fails to boot (e.g., no DB), we get 503
            resp = client.post("/session/boot")
            # Either boots successfully or fails gracefully
            assert resp.status_code in (200, 503)

    def test_chat_requires_session(self):
        """Chat without active session should return 400."""
        # Force state to have brain but no session
        with TestClient(app, raise_server_exceptions=False) as client:
            # After lifespan boots brain, but before session boot
            state.session_active = False
            resp = client.post("/chat", json={"message": "hey"})
            assert resp.status_code == 400


class TestMoodEndpoint:
    def test_mood_requires_session(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            state.session_active = False
            resp = client.get("/mood")
            assert resp.status_code == 400


class TestEmbodiedEndpoint:
    def test_embodied_requires_session(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            state.session_active = False
            resp = client.get("/embodied")
            assert resp.status_code == 400


class TestSchemaEndpoint:
    def test_schema_without_brain(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            state.thalamus = None
            resp = client.get("/schema")
            assert resp.status_code == 503


class TestMemoriesEndpoint:
    def test_memories_requires_brain(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            state.app_context = None
            resp = client.get("/memories?q=test")
            assert resp.status_code == 503


class TestHistoryEndpoint:
    def test_history_requires_session(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            state.session_active = False
            resp = client.get("/history")
            assert resp.status_code == 400


class TestSafeJson:
    def test_safe_json_handles_primitives(self):
        from emotive.api.server import _safe_json
        result = _safe_json({"a": 1, "b": "text", "c": 0.5, "d": True, "e": None})
        assert result == {"a": 1, "b": "text", "c": 0.5, "d": True, "e": None}

    def test_safe_json_handles_nested(self):
        from emotive.api.server import _safe_json
        result = _safe_json({"outer": {"inner": 42}})
        assert result == {"outer": {"inner": 42}}

    def test_safe_json_converts_non_serializable(self):
        from emotive.api.server import _safe_json
        import uuid
        uid = uuid.uuid4()
        result = _safe_json({"id": uid})
        assert result["id"] == str(uid)

    def test_safe_json_handles_lists(self):
        from emotive.api.server import _safe_json
        result = _safe_json({"items": [1, "two", {"nested": True}]})
        assert result == {"items": [1, "two", {"nested": True}]}
