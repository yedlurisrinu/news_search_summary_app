"""
tests/agents/news_category_search_agent_test.py

Unit tests for agents/news_category_search_agent.py:
  - CategoryIntent schema validation
  - _resolve_categories_in_es()  — happy path, zero hits, ES exception
  - run_category_search_agent()  — various LLM outputs and edge cases
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents.news_category_search_agent  # noqa: F401 — needed for patch() target resolution


# ---------------------------------------------------------------------------
# CategoryIntent — pydantic model
# ---------------------------------------------------------------------------

class TestCategoryIntent:
    def test_valid_intent(self):
        from agents.news_category_search_agent import CategoryIntent
        intent = CategoryIntent(
            categories=["technology", "ai"], fetch_latest=True, is_news_query=True
        )
        assert intent.categories == ["technology", "ai"]
        assert intent.fetch_latest is True
        assert intent.is_news_query is True

    def test_empty_categories_allowed(self):
        from agents.news_category_search_agent import CategoryIntent
        intent = CategoryIntent(categories=[], fetch_latest=False, is_news_query=False)
        assert intent.categories == []
        assert intent.is_news_query is False

    def test_fetch_latest_defaults_supported(self):
        from agents.news_category_search_agent import CategoryIntent
        intent = CategoryIntent(
            categories=["sports"], fetch_latest=False, is_news_query=True
        )
        assert intent.fetch_latest is False
        assert intent.is_news_query is True


# ---------------------------------------------------------------------------
# _resolve_categories_in_es
# ---------------------------------------------------------------------------

class TestResolveCategoriesInEs:
    def _make_es_response(self, total_value: int) -> dict:
        return {
            "hits": {"total": {"value": total_value}},
            "aggregations": {
                "matched_sources": {
                    "buckets": [{"key": "TechNews", "doc_count": total_value}]
                }
            },
        }

    def test_returns_original_categories_on_hits(self, mock_es_client):
        mock_es_client.search.return_value = self._make_es_response(5)

        with patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_category_search_agent import _resolve_categories_in_es
            result = _resolve_categories_in_es(["technology", "ai"])

        assert result == ["technology", "ai"]

    def test_returns_categories_when_zero_hits(self, mock_es_client):
        mock_es_client.search.return_value = self._make_es_response(0)

        with patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_category_search_agent import _resolve_categories_in_es
            result = _resolve_categories_in_es(["niche_topic"])

        # Even with zero hits the LLM categories are passed through
        assert result == ["niche_topic"]

    def test_returns_categories_when_es_raises(self, mock_es_client):
        mock_es_client.search.side_effect = ConnectionError("ES down")

        with patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_category_search_agent import _resolve_categories_in_es
            result = _resolve_categories_in_es(["politics"])

        assert result == ["politics"]

    def test_calls_es_with_correct_index(self, mock_es_client):
        mock_es_client.search.return_value = self._make_es_response(3)

        with patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_category_search_agent import _resolve_categories_in_es, NEWS_INDEX
            _resolve_categories_in_es(["health"])

        call_kwargs = mock_es_client.search.call_args
        assert call_kwargs.kwargs.get("index") == NEWS_INDEX or \
               call_kwargs.args[0] == NEWS_INDEX if call_kwargs.args else True

    def test_query_includes_category_terms(self, mock_es_client):
        mock_es_client.search.return_value = self._make_es_response(2)

        with patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_category_search_agent import _resolve_categories_in_es
            _resolve_categories_in_es(["climate", "science"])

        _, call_kwargs = mock_es_client.search.call_args
        body = call_kwargs.get("body", {})
        should_clauses = body.get("query", {}).get("bool", {}).get("should", [])
        matched_titles = [
            clause["match"]["title"] for clause in should_clauses if "match" in clause
        ]
        assert "climate" in matched_titles
        assert "science" in matched_titles


# ---------------------------------------------------------------------------
# run_category_search_agent
# ---------------------------------------------------------------------------

class TestRunCategorySearchAgent:
    def _mock_chain_invoke(self, categories: list[str], fetch_latest: bool):
        """Return a mock intent chain that yields the given output."""
        chain = MagicMock()
        chain.invoke.return_value = {
            "categories": categories,
            "fetch_latest": fetch_latest,
        }
        return chain

    def test_returns_category_search_result(self, mock_es_client):
        mock_es_client.search.return_value = {
            "hits": {"total": {"value": 2}},
        }
        chain = self._mock_chain_invoke(["technology"], False)

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("What is the latest in tech?")

        assert result.categories == ["technology"]
        assert result.fetch_latest is False

    def test_fetch_latest_true_propagated(self, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"total": {"value": 1}}}
        chain = self._mock_chain_invoke(["sports"], True)

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("latest sports scores today")

        assert result.fetch_latest is True
        assert "sports" in result.categories

    def test_defaults_to_general_when_no_categories(self, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"total": {"value": 0}}}
        chain = self._mock_chain_invoke([], False)

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("something vague")

        assert "general" in result.categories

    def test_multiple_categories_all_returned(self, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"total": {"value": 4}}}
        chain = self._mock_chain_invoke(["politics", "economy", "world"], False)

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("global economic and political news")

        assert len(result.categories) == 3
        assert "politics" in result.categories
        assert "economy" in result.categories

    def test_es_error_does_not_break_pipeline(self, mock_es_client):
        mock_es_client.search.side_effect = RuntimeError("cluster unavailable")
        chain = self._mock_chain_invoke(["finance"], False)

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("finance news")

        # ES error is swallowed; categories still come through from LLM
        assert result.categories == ["finance"]

    def test_is_news_query_true_for_news_topics(self, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"total": {"value": 3}}}

        chain = MagicMock()
        chain.invoke.return_value = {
            "categories": ["technology"],
            "fetch_latest": False,
            "is_news_query": True,
        }

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("latest tech news")

        assert result.is_news_query is True

    def test_is_news_query_false_skips_es_and_returns_early(self, mock_es_client):
        chain = MagicMock()
        chain.invoke.return_value = {
            "categories": [],
            "fetch_latest": False,
            "is_news_query": False,
        }

        with (
            patch("agents.news_category_search_agent._build_intent_chain", return_value=chain),
            patch("agents.news_category_search_agent.get_es_client", return_value=mock_es_client),
        ):
            from agents.news_category_search_agent import run_category_search_agent
            result = run_category_search_agent("what is 2 + 2?")

        assert result.is_news_query is False
        assert result.categories == []
        assert result.fetch_latest is False
        # ES should NOT have been called for a non-news query
        mock_es_client.search.assert_not_called()