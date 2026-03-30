"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:57 AM
@File: app.py
"""
from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

"""
api/app.py

FastAPI application — listens on port 8003.

Endpoints:
  POST /query    — run the full multi-agent pipeline
  GET  /health   — liveness probe
  GET  /ready    — readiness probe (pings Elasticsearch)
"""

from py_commons_per.logging_setup import setup_logging
from py_commons_per.vault_secret_loader import load_secrets
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agents.manager_agent import run_manager_agent
import logging

setup_logging()
load_secrets()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Summary RAG",
    description="Multi-agent pipeline: Category Search → Semantic Search → Summarise",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language news query")


class ArticleOut(BaseModel):
    title: str
    link: str
    source: str
    published_at: str

class QueryResponse(BaseModel):
    query: str
    categories: list[str]
    summary: str
    articles: list[ArticleOut]
    duration_seconds: float
    error: str | None = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
chat_ui_path = Path(__name__).parent.parent

""" End point that will render chat ui html for chat interaction, this is context less """
@app.get("/")
def serve_ui():
    return FileResponse(str(chat_ui_path)+"/static/chat-ui.html")

@app.get("/health", tags=["Ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["Ops"])
async def ready() -> dict:
    """Ping Elasticsearch to confirm the app is ready to serve traffic."""
    from agents.base import get_es_client
    try:
        es = get_es_client()
        if not es.ping():
            raise RuntimeError("ping returned False")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Elasticsearch not reachable: {exc}",
        ) from exc
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse, tags=["Pipeline"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Run the full multi-agent news RAG pipeline.

    Flow:
      1. CategorySearchAgent  — identifies topic categories from the query
      2. NewsSearchAgent      — semantic search in Elasticsearch
      3. SummaryAgent         — gpt-4o-mini summarization
    """
    logger.info("POST /query  query=%r", request.query)
    result = run_manager_agent(request.query)

    if result.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )

    return QueryResponse(
        query=result.query,
        categories=result.categories,
        summary=result.summary,
        articles=[ArticleOut(**a) for a in result.articles],
        duration_seconds=result.duration_seconds,
    )

