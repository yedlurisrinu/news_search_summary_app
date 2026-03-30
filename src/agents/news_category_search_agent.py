"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:51 AM
@File: news_category_search_agent.py
"""
from __future__ import annotations

"""
agents/news_category_search_agent.py

Responsibility:
  1. Parse the user query to identify news categories of interest.
  2. Verify those categories exist in the news_articles index.
  3. Detect whether the user wants the *latest* articles (recency intent).
  4. Return a CategorySearchResult consumed by news_search_agent.

Technology: LangChain LCEL + OpenAI function-calling + direct ES aggregation.
Tracing:    LangSmith via LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY env vars.
"""

import logging
import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from agents.base import CategorySearchResult, NEWS_INDEX, get_es_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class CategoryIntent(BaseModel):
    categories: list[str] = Field(
        description=(
            "List of news category names extracted from the user query. "
            "Use lowercase, singular nouns e.g. 'technology', 'politics', 'sports'."
        )
    )
    fetch_latest: bool = Field(
        description=(
            "True if the user explicitly asks for latest, recent, today's, "
            "or newest articles. False otherwise."
        )
    )


# ---------------------------------------------------------------------------
# LLM chain — extract intent from the query
# ---------------------------------------------------------------------------

_CATEGORY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a news query analyser. Given a user query, extract:
1. `categories`: the news topic categories being asked about.
   Use well-known category names such as: technology, politics, sports, business,
   entertainment, science, health, world, finance, climate, ai, crypto.
   If the query spans multiple topics return all of them.
2. `fetch_latest`: true only when the user explicitly wants recent/latest/today's news.

Respond ONLY with valid JSON matching this schema:
{{"categories": ["..."], "fetch_latest": true/false}}""",
    ),
    ("human", "{query}"),
])


def _build_intent_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return _CATEGORY_PROMPT | llm | JsonOutputParser()


# ---------------------------------------------------------------------------
# Elasticsearch — verify which categories actually have documents
# ---------------------------------------------------------------------------

def _resolve_categories_in_es(categories: list[str]) -> list[str]:
    """
    Run a terms aggregation on source.keyword to confirm at least one article
    exists for each requested category.  Falls back to returning the LLM
    categories as-is if ES has no 'category' field — the search agent will
    still use them as semantic query hints.
    """
    es = get_es_client()
    try:
        resp = es.search(
            index=NEWS_INDEX,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"title": cat}} for cat in categories
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "aggs": {
                    "matched_sources": {
                        "terms": {"field": "source.keyword", "size": 20}
                    }
                },
            },
        )
        total_hits = resp["hits"]["total"]["value"]
        if total_hits == 0:
            logger.warning(
                "No ES documents matched categories %s — proceeding anyway.", categories
            )
        else:
            logger.info(
                "Category check: %d documents matched for %s", total_hits, categories
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ES category verification failed (%s) — skipping check.", exc)

    # Return the LLM-extracted categories regardless; the search agent will
    # use them semantically even if ES has no explicit category field.
    return categories


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="NewsCategorySearchAgent")
def run_category_search_agent(query: str) -> CategorySearchResult:
    """
    Parse *query*, extract categories and recency intent, verify against ES,
    and return a CategorySearchResult.

    Args:
        query: Raw user query string.

    Returns:
        CategorySearchResult with confirmed categories and fetch_latest flag.
    """
    logger.info("CategorySearchAgent: analysing query=%r", query)

    chain = _build_intent_chain()
    intent: dict = chain.invoke({"query": query})

    categories: list[str] = intent.get("categories", [])
    fetch_latest: bool = intent.get("fetch_latest", False)

    if not categories:
        logger.warning("No categories extracted — defaulting to 'general news'.")
        categories = ["general"]

    verified = _resolve_categories_in_es(categories)

    logger.info(
        "CategorySearchAgent: categories=%s fetch_latest=%s", verified, fetch_latest
    )
    return CategorySearchResult(categories=verified, fetch_latest=fetch_latest)