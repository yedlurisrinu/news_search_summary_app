"""
tests/agents/manager_agent_test.py

Unit tests for agents/manager_agent.py:
  - _serialise_article()      — field selection / omission of raw content
  - run_manager_agent()       — happy path, guard rails, each stage failure, edge cases
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
# run_manager_agent — guard rail 1: off-topic query
# ---------------------------------------------------------------------------

class TestRunManagerAgentOffTopic:
    def test_off_topic_returns_polite_refusal(self, off_topic_category_result):
        with patch(
            "agents.manager_agent.run_category_search_agent",
            return_value=off_topic_category_result,
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("What is the square root of 144?")

        assert response.error is None
        assert response.summary != ""
        assert "news" in response.summary.lower()

    def test_off_topic_does_not_call_search_agent(self, off_topic_category_result):
        with (
            patch(
                "agents.manager_agent.run_category_search_agent",
                return_value=off_topic_category_result,
            ),
            patch("agents.manager_agent.run_news_search_agent") as mock_search,
        ):
            from agents.manager_agent import run_manager_agent
            run_manager_agent("Write me a poem")

        mock_search.assert_not_called()

    def test_off_topic_does_not_call_summary_agent(self, off_topic_category_result):
        with (
            patch(
                "agents.manager_agent.run_category_search_agent",
                return_value=off_topic_category_result,
            ),
            patch("agents.manager_agent.run_summary_agent") as mock_summary,
        ):
            from agents.manager_agent import run_manager_agent
            run_manager_agent("Help me write a cover letter")

        mock_summary.assert_not_called()

    def test_off_topic_returns_empty_categories(self, off_topic_category_result):
        with patch(
            "agents.manager_agent.run_category_search_agent",
            return_value=off_topic_category_result,
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("What is 2 + 2?")

        assert response.categories == []
        assert response.articles == []


# ---------------------------------------------------------------------------
# run_manager_agent — guard rail 2: no articles found
# ---------------------------------------------------------------------------

class TestRunManagerAgentNoArticles:
    def test_no_articles_returns_polite_no_results_reply(
        self, category_result, empty_search_result
    ):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=empty_search_result),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("very obscure niche topic")

        assert response.error is None
        assert response.summary != ""
        assert response.articles == []

    def test_no_articles_does_not_call_summary_agent(
        self, category_result, empty_search_result
    ):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=empty_search_result),
            patch("agents.manager_agent.run_summary_agent") as mock_summary,
        ):
            from agents.manager_agent import run_manager_agent
            run_manager_agent("no results query")

        mock_summary.assert_not_called()

    def test_no_articles_preserves_categories(self, category_result, empty_search_result):
        with (
            patch("agents.manager_agent.run_category_search_agent", return_value=category_result),
            patch("agents.manager_agent.run_news_search_agent", return_value=empty_search_result),
        ):
            from agents.manager_agent import run_manager_agent
            response = run_manager_agent("no results query")

        assert response.categories == category_result.categories


# ---------------------------------------------------------------------------
# run_manager_agent — edge cases
# ---------------------------------------------------------------------------

class TestRunManagerAgentEdgeCases:
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