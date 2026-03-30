"""Tests for reset_layer MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_reset_layer_episodes(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("reset_layer_tool", {
            "layer": "episodes",
            "reason": "test reset",
        })).data
        assert r["status"] == "ok"
        assert r["data"]["layer"] == "episodes"


@pytest.mark.asyncio
async def test_reset_layer_invalid(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("reset_layer_tool", {
            "layer": "temperament",
            "reason": "test",
        })).data
        assert r["status"] == "error"
        assert r["error"] == "invalid_layer"


@pytest.mark.asyncio
async def test_reset_layer_unavailable(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("reset_layer_tool", {
            "layer": "mood",
            "reason": "test",
        })).data
        assert r["status"] == "error"
        assert r["error"] == "layer_not_available"
