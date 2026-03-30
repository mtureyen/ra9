"""Tests for store_memory MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_store_memory_tool_episodic(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "Test episodic memory",
            "memory_type": "episodic",
        })).data
        assert r["status"] == "ok"
        assert "memory_id" in r["data"]
        assert r["data"]["memory_type"] == "episodic"


@pytest.mark.asyncio
async def test_store_memory_tool_semantic(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "Vulnerability builds trust",
            "memory_type": "semantic",
        })).data
        assert r["data"]["memory_type"] == "semantic"


@pytest.mark.asyncio
async def test_store_memory_tool_procedural(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "How to listen actively",
            "memory_type": "procedural",
        })).data
        assert r["data"]["memory_type"] == "procedural"


@pytest.mark.asyncio
async def test_store_memory_tool_invalid_type(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "bad type",
            "memory_type": "invalid",
        })).data
        assert r["status"] == "error"
        assert r["error"] == "invalid_memory_type"


@pytest.mark.asyncio
async def test_store_memory_tool_with_tags(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "Tagged memory",
            "memory_type": "episodic",
            "tags": ["important", "social"],
        })).data
        assert r["data"]["tags"] == ["important", "social"]


@pytest.mark.asyncio
async def test_store_memory_tool_with_significance(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "Very important memory",
            "memory_type": "episodic",
            "significance": 0.9,
        })).data
        assert r["status"] == "ok"
        assert r["data"]["significance"] == 0.9


@pytest.mark.asyncio
async def test_store_memory_tool_invalid_significance(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("store_memory_tool", {
            "content": "Bad significance",
            "memory_type": "episodic",
            "significance": 1.5,
        })).data
        assert r["status"] == "error"
        assert r["error"] == "invalid_significance"
