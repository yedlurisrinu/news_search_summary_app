"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:55 AM
@File: news_summary_agent.py
"""
from __future__ import annotations

"""
agents/news_summary_agent.py

Responsibility:
  Take the retrieved articles from news_search_agent and produce a concise,
  well-structured news summary using OpenAI gpt-4o-mini via LangChain LCEL.

Technology: LangChain LCEL (Prompt | ChatOpenAI | StrOutputParser), LangSmith.
"""

import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

from agents.base import NewsSearchResult

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert news analyst. You will receive a user query and a
collection of news article snippets retrieved from a search engine.

Your task:
- Write a clear, factual summary of 3–5 paragraphs that directly answers the query.
- Lead with the most important finding.
- When referencing a specific fact, cite the article title in brackets, e.g. [Article Title].
- Do not invent facts. If the articles are insufficient, say so explicitly.
- Remain objective and neutral in tone.""",
    ),
    (
        "human",
        """USER QUERY:
{query}

RETRIEVED ARTICLES:
{context}

Write the summary:""",
    ),
])


def _build_context(search_result: NewsSearchResult) -> str:
    """Format retrieved articles into a numbered context block for the prompt."""
    if not search_result.articles:
        return "No articles were retrieved."

    blocks: list[str] = []
    for i, article in enumerate(search_result.articles, start=1):
        pub = article.published_at[:10] if article.published_at else "unknown date"
        blocks.append(
            f"[{i}] {article.title}\n"
            f"Source: {article.source}  |  Published: {pub}\n"
            f"Link: {article.link}\n"
            f"{article.content[:800]}"   # cap per-article content to keep prompt focused
        )
    return "\n\n---\n\n".join(blocks)


def _build_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return _SUMMARY_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="NewsSummaryAgent")
def run_summary_agent(search_result: NewsSearchResult) -> str:
    """
    Summarise articles in *search_result* relative to the original query.

    Args:
        search_result: Output from news_search_agent containing articles
                       and the query string used.

    Returns:
        A human-readable news summary string.
    """
    if not search_result.articles:
        return (
            "No relevant articles were found for your query. "
            "Try broadening your search terms."
        )

    logger.info(
        "SummaryAgent: summarising %d articles for query=%r",
        len(search_result.articles),
        search_result.query_used,
    )

    context = _build_context(search_result)
    chain = _build_chain()
    summary: str = chain.invoke({
        "query": search_result.query_used,
        "context": context,
    })

    logger.info("SummaryAgent: produced %d-char summary", len(summary))
    return summary