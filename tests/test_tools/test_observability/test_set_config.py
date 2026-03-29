"""Tests for set_config MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_set_config_updates_value(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("set_config_tool", {
            "key": "consolidation.significance_threshold",
            "value": 0.5,
            "reason": "test",
        })).data
        assert r["status"] == "ok"
        assert r["data"]["old_value"] == 0.3
        assert r["data"]["new_value"] == 0.5


@pytest.mark.asyncio
async def test_set_config_invalid_key(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("set_config_tool", {
            "key": "nonexistent.key.path",
            "value": 1,
            "reason": "test",
        })).data
        assert r["status"] == "error"
        assert r["error"] == "invalid_key"
