"""Tests for experience_event MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_experience_event_with_appraisal(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        conv_id = r["data"]["conversation_id"]

        r = (await client.call_tool("experience_event_tool", {
            "event": "User shared a personal story",
            "source": "user_message",
            "conversation_id": conv_id,
            "appraisal": {
                "goal_relevance": 0.8,
                "novelty": 0.6,
                "valence": 0.75,
                "agency": 0.2,
                "social_significance": 0.9,
            },
        })).data
        assert r["status"] == "ok"
        assert "episode" in r["data"]
        assert "memory_id" in r["data"]
        ep = r["data"]["episode"]
        assert ep["primary_emotion"]
        assert 0 < ep["intensity"] <= 1


@pytest.mark.asyncio
async def test_experience_event_without_appraisal_uses_fallback(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        conv_id = r["data"]["conversation_id"]

        r = (await client.call_tool("experience_event_tool", {
            "event": "A beautiful and surprising moment of trust",
            "source": "user_message",
            "conversation_id": conv_id,
        })).data
        assert r["status"] == "ok"
        assert r["data"]["episode"]["primary_emotion"]


@pytest.mark.asyncio
async def test_experience_event_formative_detection(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("experience_event_tool", {
            "event": "Formative test",
            "source": "user_message",
            "appraisal": {
                "goal_relevance": 0.95,
                "novelty": 0.95,
                "valence": 0.95,
                "agency": 0.9,
                "social_significance": 0.95,
            },
        })).data
        assert r["status"] == "ok"
        # High appraisal should produce high intensity, possibly formative
        assert r["data"]["episode"]["intensity"] > 0.7


@pytest.mark.asyncio
async def test_experience_event_returns_no_mood_update(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("experience_event_tool", {
            "event": "Test event",
            "source": "user_message",
            "appraisal": {
                "goal_relevance": 0.5,
                "novelty": 0.5,
                "valence": 0.5,
                "agency": 0.5,
                "social_significance": 0.5,
            },
        })).data
        assert r["data"]["mood_update"] is None
        assert "Phase 2" in r["data"]["note"]
