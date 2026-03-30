"""Tests for end_session MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_end_session_closes_conversation(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        conv_id = r["data"]["conversation_id"]

        r = (await client.call_tool("end_session_tool", {
            "conversation_id": conv_id,
        })).data
        assert r["status"] == "ok"


@pytest.mark.asyncio
async def test_end_session_with_summary_and_topics(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        conv_id = r["data"]["conversation_id"]

        r = (await client.call_tool("end_session_tool", {
            "conversation_id": conv_id,
            "summary": "Test conversation",
            "topics": ["testing"],
        })).data
        assert r["status"] == "ok"


@pytest.mark.asyncio
async def test_end_session_nonexistent_conversation(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("end_session_tool", {
            "conversation_id": "00000000-0000-0000-0000-000000000000",
        })).data
        assert r["status"] == "error"
        assert r["error"] == "conversation_not_found"


@pytest.mark.asyncio
async def test_end_session_returns_episodes_archived(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("begin_session_tool", {})).data
        conv_id = r["data"]["conversation_id"]
        r = (await client.call_tool("end_session_tool", {
            "conversation_id": conv_id,
        })).data
        assert "episodes_archived" in r["data"]
        assert isinstance(r["data"]["episodes_archived"], int)
