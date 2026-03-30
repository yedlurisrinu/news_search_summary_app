"""
conftest.py — pytest fixtures shared across all test modules.

Provides:
  - Stub for py_commons_per so tests run without the private wheel installed.
  - Common ArticleHit / CategorySearchResult / NewsSearchResult builders.
  - A reusable mock Elasticsearch client.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub the private wheel (py_commons_per) before any app module is imported
# ---------------------------------------------------------------------------

def _make_py_commons_stub() -> None:
    """Insert a minimal stub so imports of py_commons_per don't fail."""
    pkg = types.ModuleType("py_commons_per")
    pkg.logging_setup = types.ModuleType("py_commons_per.logging_setup")
    pkg.logging_setup.setup_logging = MagicMock()
    pkg.vault_secret_loader = types.ModuleType("py_commons_per.vault_secret_loader")
    pkg.vault_secret_loader.load_secrets = MagicMock()

    sys.modules["py_commons_per"] = pkg
    sys.modules["py_commons_per.logging_setup"] = pkg.logging_setup
    sys.modules["py_commons_per.vault_secret_loader"] = pkg.vault_secret_loader


_make_py_commons_stub()


# Note: langsmith IS installed in the venv (langsmith==0.1.120) so no stub is
# needed.  The @traceable decorator will be a no-op in tests because
# LANGCHAIN_TRACING_V2 is not set in the test environment.


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_article_hit():
    """Return a single populated ArticleHit."""
    from agents.base import ArticleHit
    return ArticleHit(
        article_id="abc-123",
        title="AI Revolutionises Healthcare",
        content="Artificial intelligence is transforming how doctors diagnose diseases.",
        source="TechNews",
        published_at="2026-03-28T10:00:00Z",
        link="https://technews.example.com/ai-health",
        score=1.42,
    )


@pytest.fixture()
def sample_articles(sample_article_hit):
    """Return a small list of ArticleHit objects."""
    from agents.base import ArticleHit
    second = ArticleHit(
        article_id="def-456",
        title="Global Markets Rally on Trade News",
        content="Stock markets surged worldwide following positive trade developments.",
        source="FinanceDaily",
        published_at="2026-03-27T08:30:00Z",
        link="https://financedaily.example.com/markets",
        score=1.10,
    )
    return [sample_article_hit, second]


@pytest.fixture()
def category_result():
    """Return a CategorySearchResult for technology topics."""
    from agents.base import CategorySearchResult
    return CategorySearchResult(categories=["technology", "ai"], fetch_latest=False)


@pytest.fixture()
def category_result_latest():
    """Return a CategorySearchResult with fetch_latest=True."""
    from agents.base import CategorySearchResult
    return CategorySearchResult(categories=["technology"], fetch_latest=True)


@pytest.fixture()
def news_search_result(sample_articles):
    """Return a NewsSearchResult wrapping sample_articles."""
    from agents.base import NewsSearchResult
    return NewsSearchResult(articles=sample_articles, query_used="latest AI news")


@pytest.fixture()
def empty_search_result():
    """Return a NewsSearchResult with no articles."""
    from agents.base import NewsSearchResult
    return NewsSearchResult(articles=[], query_used="obscure topic")


@pytest.fixture()
def mock_es_client():
    """Return a MagicMock that mimics the Elasticsearch client."""
    client = MagicMock()
    client.ping.return_value = True
    return client


@pytest.fixture()
def es_search_response(sample_articles):
    """
    Return a dict that looks like an ES search response containing
    two article hits.
    """
    hits = []
    for art in sample_articles:
        hits.append({
            "_id": art.article_id,
            "_score": art.score,
            "_source": {
                "article_id": art.article_id,
                "title": art.title,
                "content": art.content,
                "source": art.source,
                "published_at": art.published_at,
                "link": art.link,
            },
        })
    return {"hits": {"total": {"value": len(hits)}, "hits": hits}}