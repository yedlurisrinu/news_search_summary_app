"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:53 AM
@File: news_search_agent.py
"""
from __future__ import annotations

"""
agents/news_search_agent.py

Responsibility:
  Search the news_articles index using Elasticsearch semantic_text queries
  (inference handled server-side by the 'news-embedding-endpoint' inference
  pipeline — no client-side embedding needed).

  When fetch_latest=True, results are filtered to the most recent 7 days
  and sorted by published_at descending before semantic re-ranking.

Technology: Elasticsearch Python client (semantic query), LangSmith tracing.
"""

import logging
from datetime import datetime, timedelta, timezone

from langsmith import traceable

from agents.base import (
    ArticleHit,
    CategorySearchResult,
    NEWS_INDEX,
    NewsSearchResult,
    get_es_client,
)

logger = logging.getLogger(__name__)

# Maximum number of articles to retrieve per search
TOP_K = 8


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _semantic_query(query_text: str, categories: list[str]) -> dict:
    """
    Build an ES query that:
      - Uses semantic search on the `content.semantic` field (semantic_text).
      - Boosts documents whose title matches any of the extracted categories.
    """
    category_boosts = [
        {"match": {"title": {"query": cat, "boost": 1.5}}}
        for cat in categories
    ]

    return {
        "size": TOP_K,
        "query": {
            "bool": {
                "must": [
                    {
                        "semantic": {
                            "field": "content.semantic",
                            "query": query_text,
                        }
                    }
                ],
                "should": category_boosts,
            }
        },
        "_source": ["article_id", "title", "content", "source", "published_at", "link"],
    }


def _latest_semantic_query(query_text: str, categories: list[str]) -> dict:
    """
    Same as _semantic_query but restricts results to the last 7 days and
    adds a published_at sort so the most recent articles surface first.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    base = _semantic_query(query_text, categories)
    base["query"]["bool"]["filter"] = [
        {"range": {"published_at": {"gte": since}}}
    ]
    base["sort"] = [
        {"published_at": {"order": "desc"}},
        "_score",
    ]
    return base


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

def _parse_hits(hits: list[dict]) -> list[ArticleHit]:
    articles: list[ArticleHit] = []
    for hit in hits:
        src = hit.get("_source", {})
        articles.append(
            ArticleHit(
                article_id=src.get("article_id", hit.get("_id", "")),
                title=src.get("title", ""),
                content=src.get("content", ""),
                source=src.get("source", ""),
                published_at=str(src.get("published_at", "")),
                link=src.get("link", ""),
                score=hit.get("_score") or 0.0,
            )
        )
    return articles


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="NewsSearchAgent")
def run_news_search_agent(
    query: str,
    category_result: CategorySearchResult,
) -> NewsSearchResult:
    """
    Search Elasticsearch for articles relevant to *query* using semantic search.

    Args:
        query:           Original user query (used as the semantic search text).
        category_result: Output from news_category_search_agent containing
                         confirmed categories and fetch_latest flag.

    Returns:
        NewsSearchResult with a list of ArticleHit objects and the query used.
    """
    logger.info(
        "NewsSearchAgent: query=%r categories=%s fetch_latest=%s",
        query,
        category_result.categories,
        category_result.fetch_latest,
    )

    if category_result.fetch_latest:
        es_query = _latest_semantic_query(query, category_result.categories)
        logger.info("NewsSearchAgent: using latest-filtered semantic query")
    else:
        es_query = _semantic_query(query, category_result.categories)
        logger.info("NewsSearchAgent: using standard semantic query")

    es = get_es_client()
    try:
        resp = es.search(index=NEWS_INDEX, body=es_query)
    except Exception as exc:
        logger.exception("NewsSearchAgent: ES query failed: %s", exc)
        raise

    hits = resp.get("hits", {}).get("hits", [])
    articles = _parse_hits(hits)

    logger.info("NewsSearchAgent: retrieved %d articles", len(articles))
    return NewsSearchResult(articles=articles, query_used=query)
