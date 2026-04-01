"""Tests for LLM-generated semantic summaries in consolidation."""

from unittest.mock import AsyncMock, MagicMock

from emotive.db.models.memory import Memory
from emotive.memory.semantic import (
    _fallback_summary,
    _llm_summarize_cluster,
    extract_semantic_from_cluster,
)


class TestFallbackSummary:
    def test_produces_pipe_delimited(self):
        contents = ["memory one", "memory two", "memory three"]
        result = _fallback_summary(contents, 3)
        assert "Pattern from 3 observations:" in result
        assert "memory one" in result
        assert "|" in result

    def test_truncates_long_content(self):
        contents = ["x" * 200, "y" * 200]
        result = _fallback_summary(contents, 2)
        # Each content truncated to 100 chars
        assert len(result) < 250


class TestLLMSummarizeCluster:
    def test_returns_summary_on_success(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value="Mertcan and Ryo share a deep bond of trust built through consistency."
        )
        result = _llm_summarize_cluster(
            mock_llm,
            ["trust conversation 1", "trust conversation 2"],
            ["trust"],
        )
        assert result is not None
        assert "trust" in result.lower()

    def test_returns_none_on_empty_response(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="")
        result = _llm_summarize_cluster(mock_llm, ["a", "b"], [])
        assert result is None

    def test_returns_none_on_short_response(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="ok")
        result = _llm_summarize_cluster(mock_llm, ["a", "b"], [])
        assert result is None

    def test_returns_none_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM down"))
        result = _llm_summarize_cluster(mock_llm, ["a", "b"], [])
        assert result is None


class TestExtractSemanticWithLLM:
    def test_uses_llm_when_available(self, db_session, embedding_service):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value="A recurring pattern of emotional growth through shared vulnerability."
        )
        # Store episodic memories for the cluster
        from emotive.memory.episodic import store_episodic
        mems = []
        for content in [
            "conversation about vulnerability and growth",
            "another deep exchange about emotional openness",
            "shared moment of vulnerability leading to growth",
        ]:
            m = store_episodic(db_session, embedding_service, content=content)
            mems.append(m)

        result = extract_semantic_from_cluster(
            db_session, embedding_service, mems, llm=mock_llm,
        )
        assert result is not None
        # Should use LLM summary, not pipe concatenation
        assert "|" not in result.content
        assert "vulnerability" in result.content.lower() or "growth" in result.content.lower()
        mock_llm.generate.assert_called_once()

    def test_fallback_without_llm(self, db_session, embedding_service):
        from emotive.memory.episodic import store_episodic
        mems = []
        for content in [
            "discussion about trust",
            "another discussion about trust and loyalty",
        ]:
            m = store_episodic(db_session, embedding_service, content=content)
            mems.append(m)

        result = extract_semantic_from_cluster(
            db_session, embedding_service, mems, llm=None,
        )
        assert result is not None
        # Should use pipe concatenation fallback
        assert "Pattern from" in result.content
        assert "|" in result.content

    def test_fallback_on_llm_failure(self, db_session, embedding_service):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        from emotive.memory.episodic import store_episodic
        mems = []
        for content in ["memory about topic x", "another memory about topic x"]:
            m = store_episodic(db_session, embedding_service, content=content)
            mems.append(m)

        result = extract_semantic_from_cluster(
            db_session, embedding_service, mems, llm=mock_llm,
        )
        assert result is not None
        # Should fall back to concatenation
        assert "Pattern from" in result.content
