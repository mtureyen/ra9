"""Tests for recall MCP tool."""

import pytest


@pytest.mark.asyncio
async def test_recall_returns_memories(mcp_client):
    async with mcp_client as client:
        await client.call_tool("store_memory_tool", {
            "content": "Unique hiking memory test xyz123",
            "memory_type": "episodic",
        })
        r = (await client.call_tool("recall_tool", {
            "query": "hiking memory test xyz123",
            "limit": 3,
        })).data
        assert r["status"] == "ok"
        assert len(r["data"]["memories"]) > 0


@pytest.mark.asyncio
async def test_recall_respects_limit(mcp_client):
    async with mcp_client as client:
        for i in range(3):
            await client.call_tool("store_memory_tool", {
                "content": f"Limit test memory {i}",
                "memory_type": "episodic",
            })
        r = (await client.call_tool("recall_tool", {
            "query": "limit test memory",
            "limit": 1,
        })).data
        assert len(r["data"]["memories"]) <= 1


@pytest.mark.asyncio
async def test_recall_response_has_relevance_scores(mcp_client):
    async with mcp_client as client:
        await client.call_tool("store_memory_tool", {
            "content": "Relevance score test abc",
            "memory_type": "episodic",
        })
        r = (await client.call_tool("recall_tool", {
            "query": "relevance score test abc",
        })).data
        mem = r["data"]["memories"][0]
        scores = mem["relevance_scores"]
        assert "semantic_similarity" in scores
        assert "recency_weight" in scores
        assert "spreading_activation" in scores
        assert "final_rank" in scores


@pytest.mark.asyncio
async def test_recall_response_has_phase0_bias_info(mcp_client):
    async with mcp_client as client:
        r = (await client.call_tool("recall_tool", {"query": "anything"})).data
        biases = r["data"]["biases_applied"]
        assert "disabled" in biases["mood_congruence"].lower()
        assert "disabled" in biases["personality_bias"].lower()
