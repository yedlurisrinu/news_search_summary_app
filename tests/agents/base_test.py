"""
tests/agents/base_test.py

Unit tests for agents/base.py:
  - get_es_client()  — happy path + missing-env-var guard
  - ArticleHit dataclass
  - CategorySearchResult dataclass
  - NewsSearchResult dataclass
  - AgentResponse dataclass
"""
from __future__ import annotations

import os
from functools import lru_cache
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_es_cache():
    """Clear the lru_cache on get_es_client so tests don't bleed state."""
    from agents.base import get_es_client
    get_es_client.cache_clear()


# ---------------------------------------------------------------------------
# get_es_client
# ---------------------------------------------------------------------------

class TestGetEsClient:
    def setup_method(self):
        _clear_es_cache()

    def teardown_method(self):
        _clear_es_cache()

    def test_raises_when_env_vars_missing(self, monkeypatch):
        monkeypatch.delenv("ELASTIC_URL", raising=False)
        monkeypatch.delenv("ELASTIC_API_KEY", raising=False)

        with patch("agents.base.get_config", return_value={
            "request_timeout": "60",
            "retry_on_timeout": "True",
            "max_retries": "3",
            "http_compress": "True",
            "connections_per_node": "10",
        }):
            with pytest.raises(RuntimeError, match="ELASTIC_URL and ELASTIC_API_KEY"):
                from agents.base import get_es_client
                get_es_client()

    def test_raises_when_only_url_set(self, monkeypatch):
        monkeypatch.setenv("ELASTIC_URL", "https://elastic.example.com")
        monkeypatch.delenv("ELASTIC_API_KEY", raising=False)

        with patch("agents.base.get_config", return_value={
            "request_timeout": "60",
            "retry_on_timeout": "True",
            "max_retries": "3",
            "http_compress": "True",
            "connections_per_node": "10",
        }):
            with pytest.raises(RuntimeError, match="ELASTIC_URL and ELASTIC_API_KEY"):
                from agents.base import get_es_client
                get_es_client()

    def test_returns_elasticsearch_instance(self, monkeypatch):
        monkeypatch.setenv("ELASTIC_URL", "https://elastic.example.com")
        monkeypatch.setenv("ELASTIC_API_KEY", "test-api-key")

        fake_es = MagicMock()
        with patch("agents.base.get_config", return_value={
            "request_timeout": "60",
            "retry_on_timeout": "True",
            "max_retries": "3",
            "http_compress": "True",
            "connections_per_node": "10",
        }), patch("agents.base.Elasticsearch", return_value=fake_es) as mock_cls:
            from agents.base import get_es_client
            result = get_es_client()

        assert result is fake_es
        mock_cls.assert_called_once()

    def test_result_is_cached(self, monkeypatch):
        """Calling get_es_client() twice must return the same object."""
        monkeypatch.setenv("ELASTIC_URL", "https://elastic.example.com")
        monkeypatch.setenv("ELASTIC_API_KEY", "test-api-key")

        fake_es = MagicMock()
        with patch("agents.base.get_config", return_value={
            "request_timeout": "60",
            "retry_on_timeout": "True",
            "max_retries": "3",
            "http_compress": "True",
            "connections_per_node": "10",
        }), patch("agents.base.Elasticsearch", return_value=fake_es):
            from agents.base import get_es_client
            first = get_es_client()
            second = get_es_client()

        assert first is second


# ---------------------------------------------------------------------------
# ArticleHit
# ---------------------------------------------------------------------------

class TestArticleHit:
    def test_defaults_score_to_zero(self):
        from agents.base import ArticleHit
        art = ArticleHit(
            article_id="id-1",
            title="Test",
            content="Body",
            source="src",
            published_at="2026-01-01",
            link="https://example.com",
        )
        assert art.score == 0.0

    def test_all_fields_stored(self):
        from agents.base import ArticleHit
        art = ArticleHit(
            article_id="id-2",
            title="Title",
            content="Content body",
            source="Source Name",
            published_at="2026-03-01T09:00:00Z",
            link="https://example.com/article",
            score=2.5,
        )
        assert art.article_id == "id-2"
        assert art.title == "Title"
        assert art.content == "Content body"
        assert art.source == "Source Name"
        assert art.published_at == "2026-03-01T09:00:00Z"
        assert art.link == "https://example.com/article"
        assert art.score == 2.5

    def test_empty_strings_allowed(self):
        from agents.base import ArticleHit
        art = ArticleHit(
            article_id="",
            title="",
            content="",
            source="",
            published_at="",
            link="",
        )
        assert art.title == ""


# ---------------------------------------------------------------------------
# CategorySearchResult
# ---------------------------------------------------------------------------

class TestCategorySearchResult:
    def test_stores_categories_and_flag(self):
        from agents.base import CategorySearchResult
        result = CategorySearchResult(
            categories=["technology", "sports"],
            fetch_latest=True,
        )
        assert result.categories == ["technology", "sports"]
        assert result.fetch_latest is True

    def test_empty_categories_allowed(self):
        from agents.base import CategorySearchResult
        result = CategorySearchResult(categories=[], fetch_latest=False)
        assert result.categories == []
        assert result.fetch_latest is False


# ---------------------------------------------------------------------------
# NewsSearchResult
# ---------------------------------------------------------------------------

class TestNewsSearchResult:
    def test_stores_articles_and_query(self, sample_articles):
        from agents.base import NewsSearchResult
        result = NewsSearchResult(
            articles=sample_articles,
            query_used="AI healthcare news",
        )
        assert len(result.articles) == 2
        assert result.query_used == "AI healthcare news"

    def test_empty_articles(self):
        from agents.base import NewsSearchResult
        result = NewsSearchResult(articles=[], query_used="nothing")
        assert result.articles == []


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

class TestAgentResponse:
    def test_defaults_error_to_none(self):
        from agents.base import AgentResponse
        resp = AgentResponse(
            query="test",
            categories=["tech"],
            summary="Some summary",
            articles=[],
            duration_seconds=0.5,
        )
        assert resp.error is None

    def test_error_field_stored(self):
        from agents.base import AgentResponse
        resp = AgentResponse(
            query="test",
            categories=[],
            summary="",
            articles=[],
            duration_seconds=0.1,
            error="Something went wrong",
        )
        assert resp.error == "Something went wrong"

    def test_all_fields_stored(self, sample_articles):
        from agents.base import AgentResponse
        articles_dict = [{"title": a.title} for a in sample_articles]
        resp = AgentResponse(
            query="latest tech news",
            categories=["technology"],
            summary="Tech is advancing rapidly.",
            articles=articles_dict,
            duration_seconds=1.23,
        )
        assert resp.query == "latest tech news"
        assert resp.categories == ["technology"]
        assert resp.summary == "Tech is advancing rapidly."
        assert len(resp.articles) == 2
        assert resp.duration_seconds == 1.23