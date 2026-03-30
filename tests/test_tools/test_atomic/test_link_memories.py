"""Tests for link_memories MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_link_memories_creates_link(mcp_client):
    async with mcp_client as client:
        m1 = (await client.call_tool("store_memory_tool", {
            "content": "Link test source",
            "memory_type": "episodic",
        })).data
        m2 = (await client.call_tool("store_memory_tool", {
            "content": "Link test target",
            "memory_type": "episodic",
        })).data

        r = (await client.call_tool("link_memories_tool", {
            "source_memory_id": m1["data"]["memory_id"],
            "target_memory_id": m2["data"]["memory_id"],
            "link_type": "causal",
            "strength": 0.8,
        })).data
        assert r["status"] == "ok"
        assert r["data"]["created"] is True


@pytest.mark.asyncio
async def test_link_memories_invalid_type(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("link_memories_tool", {
            "source_memory_id": "00000000-0000-0000-0000-000000000001",
            "target_memory_id": "00000000-0000-0000-0000-000000000002",
            "link_type": "invalid_type",
        })).data
        assert r["status"] == "error"
        assert r["error"] == "invalid_link_type"
