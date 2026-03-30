"""
tests/api/app_test.py

Unit / integration tests for the FastAPI application (api/app.py).

Uses httpx + FastAPI TestClient so no real server process is needed.
All agent calls, Vault / logging setup, and ES calls are mocked.

Endpoints tested:
  GET  /          — serves chat-ui.html
  GET  /health    — liveness probe
  GET  /ready     — readiness probe (ES ping)
  POST /query     — full multi-agent pipeline
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure the static file exists so the FileResponse doesn't explode
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parents[2] / "static"
CHAT_HTML = STATIC_DIR / "chat-ui.html"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    """
    Build a TestClient for the FastAPI app.
    Static dir and agent calls are patched so no real ES / OpenAI needed.
    """
    from fastapi.testclient import TestClient

    with patch("api.app.run_manager_agent"):  # prevent import-time side effects
        from api.app import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def mock_agent_response():
    """Return a fully populated AgentResponse."""
    from agents.base import AgentResponse
    return AgentResponse(
        query="What is happening in AI?",
        categories=["technology", "ai"],
        summary="AI is advancing rapidly across multiple domains.",
        articles=[
            {
                "article_id": "abc-1",
                "title": "AI Breakthrough",
                "source": "TechNews",
                "published_at": "2026-03-28T10:00:00Z",
                "link": "https://technews.example.com/ai",
            }
        ],
        duration_seconds=1.23,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, test_client):
        resp = test_client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /ready
# ---------------------------------------------------------------------------

class TestReadyEndpoint:
    def test_returns_200_when_es_ping_succeeds(self, test_client, mock_es_client):
        mock_es_client.ping.return_value = True

        with patch("agents.base.get_es_client", return_value=mock_es_client):
            resp = test_client.get("/ready")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ready"}

    def test_returns_503_when_es_ping_returns_false(self, test_client, mock_es_client):
        mock_es_client.ping.return_value = False

        with patch("agents.base.get_es_client", return_value=mock_es_client):
            resp = test_client.get("/ready")

        assert resp.status_code == 503

    def test_returns_503_when_es_raises(self, test_client, mock_es_client):
        mock_es_client.ping.side_effect = ConnectionError("unreachable")

        with patch("agents.base.get_es_client", return_value=mock_es_client):
            resp = test_client.get("/ready")

        assert resp.status_code == 503

    def test_error_detail_contains_reason(self, test_client, mock_es_client):
        mock_es_client.ping.side_effect = RuntimeError("cluster offline")

        with patch("agents.base.get_es_client", return_value=mock_es_client):
            resp = test_client.get("/ready")

        assert "Elasticsearch not reachable" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_valid_query_returns_200(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response):
            resp = test_client.post("/query", json={"query": "What is happening in AI?"})

        assert resp.status_code == 200

    def test_response_contains_query(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response):
            resp = test_client.post("/query", json={"query": "What is happening in AI?"})

        data = resp.json()
        assert data["query"] == "What is happening in AI?"

    def test_response_contains_categories(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response):
            resp = test_client.post("/query", json={"query": "What is happening in AI?"})

        data = resp.json()
        assert "technology" in data["categories"]
        assert "ai" in data["categories"]

    def test_response_contains_summary(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response):
            resp = test_client.post("/query", json={"query": "AI news"})

        data = resp.json()
        assert data["summary"] == mock_agent_response.summary

    def test_response_contains_articles(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response):
            resp = test_client.post("/query", json={"query": "AI news"})

        data = resp.json()
        assert len(data["articles"]) == 1
        assert data["articles"][0]["title"] == "AI Breakthrough"

    def test_response_contains_duration(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response):
            resp = test_client.post("/query", json={"query": "AI news"})

        data = resp.json()
        assert data["duration_seconds"] == pytest.approx(1.23)

    def test_short_query_returns_422(self, test_client):
        """Query shorter than 3 characters should fail Pydantic validation."""
        resp = test_client.post("/query", json={"query": "AI"})
        assert resp.status_code == 422

    def test_empty_query_returns_422(self, test_client):
        resp = test_client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(self, test_client):
        resp = test_client.post("/query", json={})
        assert resp.status_code == 422

    def test_agent_error_returns_500(self, test_client):
        from agents.base import AgentResponse
        error_response = AgentResponse(
            query="test",
            categories=[],
            summary="",
            articles=[],
            duration_seconds=0.1,
            error="CategorySearchAgent failed: LLM timeout",
        )
        with patch("api.app.run_manager_agent", return_value=error_response):
            resp = test_client.post("/query", json={"query": "some valid query"})

        assert resp.status_code == 500

    def test_agent_error_detail_in_response(self, test_client):
        from agents.base import AgentResponse
        error_response = AgentResponse(
            query="test",
            categories=[],
            summary="",
            articles=[],
            duration_seconds=0.1,
            error="NewsSearchAgent failed: index missing",
        )
        with patch("api.app.run_manager_agent", return_value=error_response):
            resp = test_client.post("/query", json={"query": "valid query text"})

        assert "NewsSearchAgent failed" in resp.json()["detail"]

    def test_manager_agent_called_with_query(self, test_client, mock_agent_response):
        with patch("api.app.run_manager_agent", return_value=mock_agent_response) as mock_fn:
            test_client.post("/query", json={"query": "latest AI developments"})

        mock_fn.assert_called_once_with("latest AI developments")

    def test_non_json_body_returns_422(self, test_client):
        resp = test_client.post(
            "/query",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET / — serve chat UI
# ---------------------------------------------------------------------------

class TestServeUi:
    def test_root_returns_200_when_file_exists(self, test_client):
        if not CHAT_HTML.exists():
            pytest.skip("chat-ui.html not present in static/")
        resp = test_client.get("/")
        assert resp.status_code == 200

    def test_root_content_type_is_html(self, test_client):
        if not CHAT_HTML.exists():
            pytest.skip("chat-ui.html not present in static/")
        resp = test_client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Schema validation — QueryRequest / QueryResponse
# ---------------------------------------------------------------------------

class TestQuerySchemas:
    def test_query_request_min_length_enforced(self):
        from pydantic import ValidationError
        from api.app import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="ab")  # only 2 chars, min is 3

    def test_query_request_accepts_three_chars(self):
        from api.app import QueryRequest
        req = QueryRequest(query="abc")
        assert req.query == "abc"

    def test_article_out_schema_fields(self):
        from api.app import ArticleOut
        art = ArticleOut(
            title="Test",
            link="https://example.com",
            source="Src",
            published_at="2026-01-01",
        )
        assert art.title == "Test"
        assert art.link == "https://example.com"