"""Tests for decay_memories MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_decay_memories_returns_counts(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("decay_memories_tool", {})).data
        assert r["status"] == "ok"
        assert "memories_decayed" in r["data"]
        assert "memories_archived" in r["data"]
