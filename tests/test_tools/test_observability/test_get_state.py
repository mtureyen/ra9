"""Tests for get_state MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_get_state_all(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_state_tool", {})).data
        assert r["status"] == "ok"
        assert "temperament" in r["data"]
        assert "memory_stats" in r["data"]
        assert "active_config" in r["data"]


@pytest.mark.asyncio
async def test_get_state_temperament_only(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_state_tool", {"layers": ["temperament"]})).data
        assert "temperament" in r["data"]
        assert r["data"]["temperament"]["novelty_seeking"] == 0.5


@pytest.mark.asyncio
async def test_get_state_memory_stats_only(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_state_tool", {"layers": ["memory_stats"]})).data
        stats = r["data"]["memory_stats"]
        assert "total_memories" in stats
        assert "by_type" in stats


@pytest.mark.asyncio
async def test_get_state_includes_active_config(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("get_state_tool", {})).data
        assert r["data"]["active_config"]["phase"] == 0
