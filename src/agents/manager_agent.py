"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:56 AM
@File: manager_agent.py
"""
from __future__ import annotations

"""
agents/manager_agent.py

Responsibility:
  Orchestrate the full pipeline in order:
    1. news_category_search_agent  — extract categories + recency intent
    2. news_search_agent           — semantic search in Elasticsearch
    3. news_summary_agent          — OpenAI gpt-4o-mini summarization

  Return a single AgentResponse with summary, categories, and source articles.

Technology: LangSmith end-to-end trace wraps all child agent traces.
"""

import logging
import time

from langsmith import traceable

from agents.base import AgentResponse, ArticleHit
from agents.news_category_search_agent import run_category_search_agent
from agents.news_search_agent import run_news_search_agent
from agents.news_summary_agent import run_summary_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polite static replies for out-of-scope situations
# ---------------------------------------------------------------------------

_OFF_TOPIC_REPLY = (
    "I'm a news search assistant and can only help with news-related queries. "
    "Please ask me about topics such as technology, politics, sports, business, "
    "science, health, climate, or world events."
)

_NO_NEWS_FOUND_REPLY = (
    "I searched for news articles matching your query but couldn't find any relevant results. "
    "Try broadening your search terms or asking about a different topic."
)


def _serialise_article(article: ArticleHit) -> dict:
    """Return a clean dict for the API response — omit raw content."""
    return {
        "article_id": article.article_id,
        "title": article.title,
        "source": article.source,
        "published_at": article.published_at,
        "link": article.link,
    }


@traceable(name="ManagerAgent")
def run_manager_agent(query: str) -> AgentResponse:
    """
    Run the full news RAG pipeline for *query*.

    Pipeline:
        query
          → CategorySearchAgent  (LLM intent + ES category verification)
          → NewsSearchAgent       (ES semantic_text query)
          → SummaryAgent          (OpenAI gpt-4o-mini)
          → AgentResponse

    Args:
        query: Raw user query string from the API layer.

    Returns:
        AgentResponse containing summary, categories, articles, and timing.
        On error, AgentResponse.error is populated and summary is empty.
    """
    logger.info("ManagerAgent: starting pipeline for query=%r", query)
    start = time.perf_counter()

    # ── Step 1: Category extraction ──────────────────────────────────────────
    try:
        category_result = run_category_search_agent(query)
    except Exception as exc:
        logger.exception("ManagerAgent: CategorySearchAgent failed")
        return AgentResponse(
            query=query,
            categories=[],
            summary="",
            articles=[],
            duration_seconds=time.perf_counter() - start,
            error=f"CategorySearchAgent failed: {exc}",
        )

    # ── Guard rail 1: Reject off-topic queries ────────────────────────────────
    if not category_result.is_news_query:
        logger.info("ManagerAgent: query is not news-related — returning polite refusal.")
        return AgentResponse(
            query=query,
            categories=[],
            summary=_OFF_TOPIC_REPLY,
            articles=[],
            duration_seconds=round(time.perf_counter() - start, 3),
        )

    # ── Step 2: Semantic news search ─────────────────────────────────────────
    try:
        search_result = run_news_search_agent(
            query=query,
            category_result=category_result,
        )
    except Exception as exc:
        logger.exception("ManagerAgent: NewsSearchAgent failed")
        return AgentResponse(
            query=query,
            categories=category_result.categories,
            summary="",
            articles=[],
            duration_seconds=time.perf_counter() - start,
            error=f"NewsSearchAgent failed: {exc}",
        )

    # ── Guard rail 2: No articles found ──────────────────────────────────────
    if not search_result.articles:
        logger.info("ManagerAgent: no articles found — returning polite no-results reply.")
        return AgentResponse(
            query=query,
            categories=category_result.categories,
            summary=_NO_NEWS_FOUND_REPLY,
            articles=[],
            duration_seconds=round(time.perf_counter() - start, 3),
        )

    # ── Step 3: Summarization ────────────────────────────────────────────────
    try:
        summary = run_summary_agent(search_result)
    except Exception as exc:
        logger.exception("ManagerAgent: SummaryAgent failed")
        return AgentResponse(
            query=query,
            categories=category_result.categories,
            summary="",
            articles=[_serialise_article(a) for a in search_result.articles],
            duration_seconds=time.perf_counter() - start,
            error=f"SummaryAgent failed: {exc}",
        )

    duration = time.perf_counter() - start
    logger.info("ManagerAgent: pipeline completed in %.2fs", duration)

    return AgentResponse(
        query=query,
        categories=category_result.categories,
        summary=summary,
        articles=[_serialise_article(a) for a in search_result.articles],
        duration_seconds=round(duration, 3),
    )