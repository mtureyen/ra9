"""Tests for export_timeline MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_export_timeline_default(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("export_timeline_tool", {})).data
        assert r["status"] == "ok"
        assert "episodes" in r["data"]
        assert "memories" in r["data"]
        assert "consolidations" in r["data"]


@pytest.mark.asyncio
async def test_export_timeline_specific_include(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("export_timeline_tool", {
            "include": ["memories"],
        })).data
        assert "memories" in r["data"]
        assert "episodes" not in r["data"]
