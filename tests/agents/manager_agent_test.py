"""
tests/agents/manager_agent_test.py

Unit tests for agents/manager_agent.py:
  - _serialise_article()      — field selection / omission of raw content
  - run_manager_agent()       — happy path, each stage failure, empty results
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Pre-import so patch() can resolve the dotted target paths
import agents.manager_agent  # noqa: F401
import agents.news_category_search_agent  # noqa: F401
import agents.news_search_agent  # noqa: F401
import agents.news_summary_agent  # noqa: F401


# ---------------------------------------------------------------------------
# _serialise_article
# ---------------------------------------------------------------------------

class TestSerialiseArticle:
    def test_returns_expected_keys(self, sample_article_hit):
        from agents.manager_agent import _serialise_article
        result = _serialise_article(sample_article_hit)

        assert set(result.keys()) == {"article_id", "title", "source", "published_at", "link"}

    def test_content_field_omitted(self, sample_article_hit):
        from agents.manager_agent import _serialise_article
        result = _serialise_article(sample_article_hit)
        assert "content" not in result

    def test_score_field_omitted(self, sample_article_hit):
        from agents.manager_agent import _serialise_article
        result = _serialise_article(sample_article_hit)
        assert "score" not in result

    def test_values_match_article(self, sample_article_hit):
        from agents.manager_agent import _serialise_article
        result = _serialise_article(sample_article_hit)

        assert result["article_id"] == sample_article_hit.article_id
        assert result["title"] == sample_article_hit.title
        assert result["source"] == sample_article_hit.source
        assert result["published_at"] == sample_article_hit.published_at
        assert result["link"] == sample_article_hit.link


# ---------------------------------------------------------------------------
# run_manager_agent — full pipeline success
# ---------------------------------------------------------------------------

class TestRunManagerAgentSuccess:
    def test_happy_path_returns_agent_response(
        self, category_result, news_search_result, sample_articles
    ):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=news_search_result),
            patch("agents.manager_agent.run_summary_agent", return_value="Great AI summary."),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("What is happening in AI?")

        assert response.query == "What is happening in AI?"
        assert response.categories == ["technology", "ai"]
        assert response.summary == "Great AI summary."
        assert len(response.articles) == len(sample_articles)
        assert response.error is None
        assert response.duration_seconds >= 0

    def test_articles_are_serialised_dicts(self, category_result, news_search_result):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=news_search_result),
            patch("agents.manager_agent.run_summary_agent", return_value="Summary text."),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("test query")

        for art_dict in response.articles:
            assert isinstance(art_dict, dict)
            assert "content" not in art_dict

    def test_duration_is_non_negative(self, category_result, news_search_result):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=news_search_result),
            patch("agents.manager_agent.run_summary_agent", return_value="Summary."),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("query")

        assert response.duration_seconds >= 0


# ---------------------------------------------------------------------------
# run_manager_agent — CategorySearchAgent failure
# ---------------------------------------------------------------------------

class TestRunManagerAgentCategoryFailure:
    def test_returns_error_when_category_agent_raises(self):
        with patch(
            "agents.manager_agent.run_category_search_agent",
            side_effect=RuntimeError("LLM timeout"),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("AI news")

        assert response.error is not None
        assert "CategorySearchAgent failed" in response.error
        assert "LLM timeout" in response.error

    def test_categories_empty_on_category_failure(self):
        with patch(
            "agents.manager_agent.run_category_search_agent",
            side_effect=ValueError("bad response"),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("sports news")

        assert response.categories == []
        assert response.articles == []
        assert response.summary == ""


# ---------------------------------------------------------------------------
# run_manager_agent — NewsSearchAgent failure
# ---------------------------------------------------------------------------

class TestRunManagerAgentSearchFailure:
    def test_returns_error_when_search_agent_raises(self, category_result):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch(
                "agents.manager_agent.run_news_search_agent",
                side_effect=ConnectionError("ES unreachable"),
            ),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("technology news")

        assert response.error is not None
        assert "NewsSearchAgent failed" in response.error

    def test_categories_preserved_on_search_failure(self, category_result):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch(
                "agents.manager_agent.run_news_search_agent",
                side_effect=Exception("index not found"),
            ),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("query")

        assert response.categories == category_result.categories
        assert response.articles == []
        assert response.summary == ""


# ---------------------------------------------------------------------------
# run_manager_agent — SummaryAgent failure
# ---------------------------------------------------------------------------

class TestRunManagerAgentSummaryFailure:
    def test_returns_error_when_summary_agent_raises(
        self, category_result, news_search_result, sample_articles
    ):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=news_search_result),
            patch(
                "agents.manager_agent.run_summary_agent",
                side_effect=RuntimeError("OpenAI quota exceeded"),
            ),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("tech news")

        assert response.error is not None
        assert "SummaryAgent failed" in response.error

    def test_articles_still_returned_on_summary_failure(
        self, category_result, news_search_result, sample_articles
    ):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=news_search_result),
            patch(
                "agents.manager_agent.run_summary_agent",
                side_effect=Exception("network error"),
            ),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("tech news")

        # Articles retrieved before summary should still be in the response
        assert len(response.articles) == len(sample_articles)
        assert response.summary == ""
        assert response.categories == category_result.categories


# ---------------------------------------------------------------------------
# run_manager_agent — edge cases
# ---------------------------------------------------------------------------

class TestRunManagerAgentEdgeCases:
    def test_empty_articles_returns_valid_response(self, category_result, empty_search_result):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=empty_search_result),
            patch("agents.manager_agent.run_summary_agent", return_value="No articles found."),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("very obscure topic")

        assert response.articles == []
        assert response.error is None
        assert response.summary == "No articles found."

    def test_query_preserved_in_response(self, category_result, news_search_result):
        original_query = "What are the latest developments in quantum computing?"
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=news_search_result),
            patch("agents.manager_agent.run_summary_agent", return_value="Quantum summary."),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent(original_query)

        assert response.query == original_query