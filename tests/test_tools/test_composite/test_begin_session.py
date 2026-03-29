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
