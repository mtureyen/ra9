"""Tests for begin_session MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_begin_session_returns_conversation_id(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        assert r["status"] == "ok"
        assert "conversation_id" in r["data"]
        assert len(r["data"]["conversation_id"]) == 36  # UUID format


@pytest.mark.asyncio
async def test_begin_session_returns_temperament(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        temp = r["data"]["temperament"]
        assert temp["novelty_seeking"] == 0.5
        assert temp["sensitivity"] == 0.5
        assert len(temp) == 8


@pytest.mark.asyncio
async def test_begin_session_returns_active_config(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        cfg = r["data"]["active_config"]
        assert cfg["phase"] == 0
        assert "temperament" in cfg["layers_enabled"]


@pytest.mark.asyncio
async def test_begin_session_reports_orphan_cleanup(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        assert "orphaned_sessions_cleaned" in r["data"]
        assert isinstance(r["data"]["orphaned_sessions_cleaned"], int)


@pytest.mark.asyncio
async def test_begin_session_returns_instructions(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        assert "instructions" in r["data"]
        assert "persistent memory system" in r["data"]["instructions"]
        assert "store_memory" in r["data"]["instructions"]


@pytest.mark.asyncio
async def test_begin_session_returns_identity_memories(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        assert "identity_memories" in r["data"]
        assert isinstance(r["data"]["identity_memories"], list)


@pytest.mark.asyncio
async def test_begin_session_identity_loads_recalled_memories(mcp_client):
    async with mcp_client as client:
        # Store and recall a memory to bump its retrieval count
        await client.call_tool("store_memory_tool", {
            "content": "Identity anchor test unique abc789",
            "memory_type": "semantic",
            "significance": 0.95,
        })
        await client.call_tool("recall_tool", {
            "query": "Identity anchor test unique abc789",
        })

        # Start new session — recalled memories should appear in identity
        r = (await client.call_tool("begin_session_tool", {})).data
        identity = r["data"]["identity_memories"]
        # Should have at least some memories (from this test or others)
        assert len(identity) > 0
        # All entries should have expected fields
        for m in identity:
            assert "content" in m
            assert "memory_type" in m
            assert "retrieval_count" in m
