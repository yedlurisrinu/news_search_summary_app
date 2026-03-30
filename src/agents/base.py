"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:48 AM
@File: base.py
"""
from __future__ import annotations

"""
agents/base.py

Shared Elasticsearch client factory and common data models used across all agents.
Credentials are read from environment variables populated by HashiCorp Vault at startup.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from config.elastic_setup import get_config
from elasticsearch import Elasticsearch

# ---------------------------------------------------------------------------
# Elasticsearch client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_es_client() -> Elasticsearch:
    """
    Build and return a cached Elasticsearch client.
    Expects the following env vars (loaded from Vault):
      ELASTIC_URL   — Elastic Cloud URL
      ELASTIC_API_KEY    — Elastic Cloud API key
      And other config values loaded form elastic-config.properties
    """

    cloud_url = os.getenv("ELASTIC_URL")
    api_key = os.getenv("ELASTIC_API_KEY")
    config = get_config()
    if not cloud_url or not api_key:
        raise RuntimeError(
            "ELASTIC_URL and ELASTIC_API_KEY must be set. "
            "Check that Vault secrets loaded correctly."
        )

    return Elasticsearch(cloud_url,
        api_key = api_key,
        request_timeout = int(config['request_timeout']),
        retry_on_timeout = bool(config['retry_on_timeout']),
        max_retries = int(config['max_retries']),
        http_compress = bool(config['http_compress']),
        # Connection pooling for cloud,
        connections_per_node = int(config['connections_per_node']))

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NEWS_INDEX = "news_articles"
INFERENCE_ID = "news-embedding-endpoint"

# ---------------------------------------------------------------------------
# Data models passed between agents
# ---------------------------------------------------------------------------

@dataclass
class ArticleHit:
    """A single retrieved news article."""
    article_id: str
    title: str
    content: str
    source: str
    published_at: str
    link: str
    score: float = 0.0


@dataclass
class CategorySearchResult:
    """Output of news_category_search_agent."""
    categories: list[str]
    fetch_latest: bool          # True when user wants newest articles in those categories
    is_news_query: bool = True  # False when the query is not news-related


@dataclass
class NewsSearchResult:
    """Output of news_search_agent."""
    articles: list[ArticleHit]
    query_used: str


@dataclass
class AgentResponse:
    """Final structured response returned by manager_agent to the API."""
    query: str
    categories: list[str]
    summary: str
    articles: list[dict]        # serializable form of ArticleHit
    duration_seconds: float
    error: str | None = None