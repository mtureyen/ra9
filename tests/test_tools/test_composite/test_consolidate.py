"""Tests for consolidate MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_consolidate_manual_trigger(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("consolidate_tool", {
            "trigger_type": "manual",
        })).data
        assert r["status"] == "ok"


@pytest.mark.asyncio
async def test_consolidate_returns_all_report_sections(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("consolidate_tool", {})).data
        data = r["data"]
        assert "promotion" in data
        assert "extraction" in data
        assert "linking" in data
        assert "decay" in data
        assert "consolidation_id" in data
        assert "duration_ms" in data
