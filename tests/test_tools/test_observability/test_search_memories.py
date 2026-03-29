"""Tests for search_memories MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_search_memories_by_query(mcp_client):
    async with mcp_client as client:
        await client.call_tool("store_memory_tool", {
            "content": "Search test unique content zzz",
            "memory_type": "episodic",
        })
        r = (await client.call_tool("search_memories_tool", {
            "query": "search test unique zzz",
        })).data
        assert r["status"] == "ok"
        assert r["data"]["count"] > 0


@pytest.mark.asyncio
async def test_search_memories_filter_only(mcp_client):
    async with mcp_client as client:
        await client.call_tool("store_memory_tool", {
            "content": "Filter only test",
            "memory_type": "episodic",
        })
        r = (await client.call_tool("search_memories_tool", {
            "memory_type": "episodic",
        })).data
        assert r["status"] == "ok"
        for m in r["data"]["memories"]:
            assert m["memory_type"] == "episodic"


@pytest.mark.asyncio
async def test_search_memories_biases_applied_false(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("search_memories_tool", {
            "query": "anything",
        })).data
        assert r["data"]["biases_applied"] is False
