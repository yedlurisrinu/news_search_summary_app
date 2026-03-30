"""
tests/agents/news_summary_agent_test.py

Unit tests for agents/news_summary_agent.py:
  - _build_context()    — article formatting, truncation at 800 chars, empty case
  - run_summary_agent() — happy path, no-articles path, LLM exception
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents.news_summary_agent  # noqa: F401 — needed for patch() target resolution


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_returns_no_articles_message_when_empty(self, empty_search_result):
        from agents.news_summary_agent import _build_context
        ctx = _build_context(empty_search_result)
        assert "No articles were retrieved" in ctx

    def test_single_article_formatted_correctly(self, sample_article_hit):
        from agents.base import NewsSearchResult
        from agents.news_summary_agent import _build_context
        result = NewsSearchResult(articles=[sample_article_hit], query_used="AI")
        ctx = _build_context(result)

        assert sample_article_hit.title in ctx
        assert sample_article_hit.source in ctx
        assert sample_article_hit.link in ctx

    def test_multiple_articles_separated_by_divider(self, news_search_result):
        from agents.news_summary_agent import _build_context
        ctx = _build_context(news_search_result)
        assert "---" in ctx

    def test_article_numbering_starts_at_one(self, news_search_result):
        from agents.news_summary_agent import _build_context
        ctx = _build_context(news_search_result)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_content_truncated_to_800_chars(self):
        from agents.base import ArticleHit, NewsSearchResult
        from agents.news_summary_agent import _build_context

        long_content = "x" * 1500
        art = ArticleHit(
            article_id="trunc-1",
            title="Truncation Test",
            content=long_content,
            source="Src",
            published_at="2026-01-01",
            link="https://example.com",
        )
        result = NewsSearchResult(articles=[art], query_used="test")
        ctx = _build_context(result)

        # The context should not contain the full 1500-char content
        assert long_content not in ctx
        assert "x" * 800 in ctx

    def test_published_at_truncated_to_date(self, sample_article_hit):
        from agents.base import NewsSearchResult
        from agents.news_summary_agent import _build_context
        result = NewsSearchResult(articles=[sample_article_hit], query_used="AI")
        ctx = _build_context(result)

        # Only the date part (first 10 chars) should appear, not full timestamp
        assert "2026-03-28" in ctx
        assert "T10:00:00Z" not in ctx

    def test_empty_published_at_shows_unknown_date(self):
        from agents.base import ArticleHit, NewsSearchResult
        from agents.news_summary_agent import _build_context

        art = ArticleHit(
            article_id="no-date",
            title="No Date Article",
            content="content",
            source="Src",
            published_at="",
            link="https://example.com",
        )
        result = NewsSearchResult(articles=[art], query_used="test")
        ctx = _build_context(result)
        assert "unknown date" in ctx


# ---------------------------------------------------------------------------
# run_summary_agent
# ---------------------------------------------------------------------------

class TestRunSummaryAgent:
    def test_returns_no_articles_message_when_empty(self, empty_search_result):
        from agents.news_summary_agent import run_summary_agent
        summary = run_summary_agent(empty_search_result)
        assert "No relevant articles" in summary

    def test_no_llm_call_when_no_articles(self, empty_search_result):
        mock_chain = MagicMock()
        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            run_summary_agent(empty_search_result)
        mock_chain.invoke.assert_not_called()

    def test_returns_llm_summary_string(self, news_search_result):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is the generated summary."

        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            summary = run_summary_agent(news_search_result)

        assert summary == "This is the generated summary."

    def test_chain_invoked_with_query_and_context(self, news_search_result):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Summary."

        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            run_summary_agent(news_search_result)

        call_args = mock_chain.invoke.call_args
        payload = call_args.args[0] if call_args.args else call_args.kwargs
        assert "query" in payload
        assert "context" in payload
        assert payload["query"] == news_search_result.query_used

    def test_context_includes_article_titles(self, news_search_result):
        captured = {}

        def fake_invoke(payload):
            captured.update(payload)
            return "Summary."

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = fake_invoke

        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            run_summary_agent(news_search_result)

        for article in news_search_result.articles:
            assert article.title in captured["context"]

    def test_llm_exception_propagates(self, news_search_result):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("OpenAI rate limit")

        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            with pytest.raises(RuntimeError, match="OpenAI rate limit"):
                run_summary_agent(news_search_result)

    def test_summary_is_string(self, news_search_result):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Returned summary string."

        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            summary = run_summary_agent(news_search_result)

        assert isinstance(summary, str)

    def test_single_article_produces_summary(self, sample_article_hit):
        from agents.base import NewsSearchResult
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "One article summary."

        single = NewsSearchResult(articles=[sample_article_hit], query_used="AI health")
        with patch("agents.news_summary_agent._build_chain", return_value=mock_chain):
            from agents.news_summary_agent import run_summary_agent
            summary = run_summary_agent(single)

        assert summary == "One article summary."


# ---------------------------------------------------------------------------
# _build_chain (smoke test — verifies chain construction without live LLM)
# ---------------------------------------------------------------------------

class TestBuildChain:
    def test_build_chain_returns_callable(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        fake_llm = MagicMock()
        fake_chain = MagicMock()
        fake_llm.__or__ = MagicMock(return_value=fake_chain)

        with patch("agents.news_summary_agent.ChatOpenAI", return_value=fake_llm):
            from agents.news_summary_agent import _build_chain
            chain = _build_chain()

        assert chain is not None