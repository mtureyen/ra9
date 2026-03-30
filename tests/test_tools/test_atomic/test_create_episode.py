"""Tests for create_episode MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_create_episode_with_explicit_values(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("create_episode_tool", {
            "trigger_event": "Test manual episode",
            "goal_relevance": 0.7,
            "novelty": 0.5,
            "valence": 0.8,
            "agency": 0.3,
            "social_significance": 0.6,
        })).data
        assert r["status"] == "ok"
        assert "episode_id" in r["data"]
        assert r["data"]["primary_emotion"]


@pytest.mark.asyncio
async def test_create_episode_high_intensity_is_formative(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("create_episode_tool", {
            "trigger_event": "Extremely intense event",
            "goal_relevance": 0.95,
            "novelty": 0.95,
            "valence": 0.95,
            "agency": 0.9,
            "social_significance": 0.95,
        })).data
        assert r["data"]["intensity"] > 0.7
