"""Tests for get_history MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_get_history_events_default(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_history_tool", {})).data
        assert r["status"] == "ok"
        assert r["data"]["history_type"] == "events"


@pytest.mark.asyncio
async def test_get_history_consolidations(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_history_tool", {
            "history_type": "consolidations",
        })).data
        assert r["data"]["history_type"] == "consolidations"


@pytest.mark.asyncio
async def test_get_history_invalid_type(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_history_tool", {
            "history_type": "bad",
        })).data
        assert r["status"] == "error"


@pytest.mark.asyncio
async def test_get_history_respects_limit(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_history_tool", {"limit": 2})).data
        assert len(r["data"]["entries"]) <= 2


@pytest.mark.asyncio
async def test_get_history_episodes(mcp_client):
    async with mcp_client as client:
        # Create an episode
        await client.call_tool("experience_event_tool", {
            "event": "History test episode",
            "source": "user_message",
            "appraisal": {
                "goal_relevance": 0.6, "novelty": 0.5,
                "valence": 0.7, "agency": 0.3, "social_significance": 0.5,
            },
        })
        r = (await client.call_tool("get_history_tool", {
            "history_type": "episodes",
        })).data
        assert r["data"]["history_type"] == "episodes"
        assert len(r["data"]["entries"]) >= 1
        ep = r["data"]["entries"][0]
        assert "primary_emotion" in ep
        assert "appraisal" in ep
        assert "intensity" in ep
