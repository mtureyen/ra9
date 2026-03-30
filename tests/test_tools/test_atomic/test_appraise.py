"""Tests for appraise MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_appraise_returns_vector_and_emotion(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("appraise_tool", {
            "event": "Someone shared a beautiful personal story",
            "source": "user_message",
        })).data
        assert r["status"] == "ok"
        assert "appraisal" in r["data"]
        assert "primary_emotion" in r["data"]
        assert "intensity" in r["data"]


@pytest.mark.asyncio
async def test_appraise_is_dry_run(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("appraise_tool", {
            "event": "Test dry run",
        })).data
        assert "Dry run" in r["data"]["note"]


@pytest.mark.asyncio
async def test_appraise_returns_all_dimensions(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("appraise_tool", {
            "event": "Something happened",
        })).data
        appraisal = r["data"]["appraisal"]
        for key in ["goal_relevance", "novelty", "valence", "agency", "social_significance"]:
            assert key in appraisal
            assert 0 <= appraisal[key] <= 1
