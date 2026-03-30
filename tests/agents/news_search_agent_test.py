"""
tests/agents/news_search_agent_test.py

Unit tests for agents/news_search_agent.py:
  - _semantic_query()        — query structure, boosting
  - _latest_semantic_query() — date filter, sort injection
  - _parse_hits()            — full hit, partial hit, empty list
  - run_news_search_agent()  — standard path, latest path, ES failure
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import agents.news_search_agent  # noqa: F401 — needed for patch() target resolution


# ---------------------------------------------------------------------------
# _semantic_query
# ---------------------------------------------------------------------------

class TestSemanticQuery:
    def test_returns_dict_with_size(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("AI news", ["technology"])
        assert "size" in q
        assert q["size"] > 0

    def test_semantic_must_clause_present(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("AI healthcare", ["health"])
        must_clauses = q["query"]["bool"]["must"]
        semantic_clause = next(
            (c for c in must_clauses if "semantic" in c), None
        )
        assert semantic_clause is not None
        assert semantic_clause["semantic"]["query"] == "AI healthcare"

    def test_semantic_field_is_content_semantic(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("query text", ["technology"])
        must_clauses = q["query"]["bool"]["must"]
        semantic_clause = next(c for c in must_clauses if "semantic" in c)
        assert semantic_clause["semantic"]["field"] == "content.semantic"

    def test_category_boosts_in_should(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("market news", ["finance", "economy"])
        should_clauses = q["query"]["bool"]["should"]
        boost_titles = [
            c["match"]["title"]["query"]
            for c in should_clauses
            if "match" in c and "title" in c["match"]
        ]
        assert "finance" in boost_titles
        assert "economy" in boost_titles

    def test_category_boost_value_is_positive(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("query", ["technology"])
        should_clauses = q["query"]["bool"]["should"]
        for clause in should_clauses:
            if "match" in clause:
                assert clause["match"]["title"]["boost"] > 0

    def test_source_fields_in_query(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("query", ["tech"])
        source = q.get("_source", [])
        expected = {"article_id", "title", "content", "source", "published_at", "link"}
        assert expected.issubset(set(source))

    def test_no_filter_in_standard_query(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("news", ["world"])
        assert "filter" not in q["query"]["bool"]

    def test_no_sort_in_standard_query(self):
        from agents.news_search_agent import _semantic_query
        q = _semantic_query("news", ["world"])
        assert "sort" not in q


# ---------------------------------------------------------------------------
# _latest_semantic_query
# ---------------------------------------------------------------------------

class TestLatestSemanticQuery:
    def test_contains_date_range_filter(self):
        from agents.news_search_agent import _latest_semantic_query
        q = _latest_semantic_query("latest AI", ["ai"])
        filters = q["query"]["bool"].get("filter", [])
        range_filter = next(
            (f for f in filters if "range" in f), None
        )
        assert range_filter is not None
        assert "published_at" in range_filter["range"]

    def test_filter_uses_gte(self):
        from agents.news_search_agent import _latest_semantic_query
        q = _latest_semantic_query("latest AI", ["ai"])
        filters = q["query"]["bool"]["filter"]
        range_filter = next(f for f in filters if "range" in f)
        assert "gte" in range_filter["range"]["published_at"]

    def test_sort_by_published_at_desc(self):
        from agents.news_search_agent import _latest_semantic_query
        q = _latest_semantic_query("latest news", ["world"])
        sort = q.get("sort", [])
        pub_sort = next(
            (s for s in sort if isinstance(s, dict) and "published_at" in s), None
        )
        assert pub_sort is not None
        assert pub_sort["published_at"]["order"] == "desc"

    def test_score_sort_is_secondary(self):
        from agents.news_search_agent import _latest_semantic_query
        q = _latest_semantic_query("latest news", ["world"])
        sort = q.get("sort", [])
        assert "_score" in sort

    def test_semantic_query_still_present(self):
        from agents.news_search_agent import _latest_semantic_query
        q = _latest_semantic_query("latest tech", ["technology"])
        must_clauses = q["query"]["bool"]["must"]
        semantic = next((c for c in must_clauses if "semantic" in c), None)
        assert semantic is not None

    def test_since_date_is_7_days_ago(self):
        from agents.news_search_agent import _latest_semantic_query
        q = _latest_semantic_query("news", ["world"])
        filters = q["query"]["bool"]["filter"]
        range_filter = next(f for f in filters if "range" in f)
        since_str = range_filter["range"]["published_at"]["gte"]
        since_dt = datetime.strptime(since_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        now = datetime.now(timezone.utc)
        delta_days = (now - since_dt).days
        assert 6 <= delta_days <= 8  # allow 1-day tolerance


# ---------------------------------------------------------------------------
# _parse_hits
# ---------------------------------------------------------------------------

class TestParseHits:
    def test_empty_list_returns_empty(self):
        from agents.news_search_agent import _parse_hits
        assert _parse_hits([]) == []

    def test_parses_full_hit(self):
        from agents.news_search_agent import _parse_hits
        hits = [{
            "_id": "hit-1",
            "_score": 1.5,
            "_source": {
                "article_id": "art-1",
                "title": "Test Title",
                "content": "Test content",
                "source": "TestSource",
                "published_at": "2026-03-01T00:00:00Z",
                "link": "https://example.com/test",
            },
        }]
        articles = _parse_hits(hits)
        assert len(articles) == 1
        a = articles[0]
        assert a.article_id == "art-1"
        assert a.title == "Test Title"
        assert a.content == "Test content"
        assert a.source == "TestSource"
        assert a.published_at == "2026-03-01T00:00:00Z"
        assert a.link == "https://example.com/test"
        assert a.score == 1.5

    def test_falls_back_to_id_when_no_article_id(self):
        from agents.news_search_agent import _parse_hits
        hits = [{
            "_id": "es-doc-id",
            "_score": 0.9,
            "_source": {"title": "No article_id"},
        }]
        articles = _parse_hits(hits)
        assert articles[0].article_id == "es-doc-id"

    def test_score_defaults_to_zero_when_none(self):
        from agents.news_search_agent import _parse_hits
        hits = [{"_id": "x", "_score": None, "_source": {"title": "T"}}]
        articles = _parse_hits(hits)
        assert articles[0].score == 0.0

    def test_parses_multiple_hits(self, es_search_response):
        from agents.news_search_agent import _parse_hits
        raw_hits = es_search_response["hits"]["hits"]
        articles = _parse_hits(raw_hits)
        assert len(articles) == 2

    def test_missing_source_fields_default_to_empty(self):
        from agents.news_search_agent import _parse_hits
        hits = [{"_id": "y", "_score": 0.5, "_source": {}}]
        articles = _parse_hits(hits)
        assert articles[0].title == ""
        assert articles[0].content == ""
        assert articles[0].source == ""
        assert articles[0].link == ""


# ---------------------------------------------------------------------------
# run_news_search_agent
# ---------------------------------------------------------------------------

class TestRunNewsSearchAgent:
    def test_standard_query_path(self, category_result, mock_es_client, es_search_response):
        mock_es_client.search.return_value = es_search_response

        with patch("agents.news_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_search_agent import run_news_search_agent
            result = run_news_search_agent("AI news", category_result)

        assert len(result.articles) == 2
        assert result.query_used == "AI news"
        mock_es_client.search.assert_called_once()

    def test_latest_query_path(
        self, category_result_latest, mock_es_client, es_search_response
    ):
        mock_es_client.search.return_value = es_search_response

        with patch("agents.news_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_search_agent import run_news_search_agent
            result = run_news_search_agent("latest tech news", category_result_latest)

        # When fetch_latest=True the query body should contain a date filter
        call_body = mock_es_client.search.call_args.kwargs.get("body", {})
        assert "filter" in call_body.get("query", {}).get("bool", {})

    def test_returns_empty_list_when_no_hits(self, category_result, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}

        with patch("agents.news_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_search_agent import run_news_search_agent
            result = run_news_search_agent("no results query", category_result)

        assert result.articles == []

    def test_es_exception_is_propagated(self, category_result, mock_es_client):
        mock_es_client.search.side_effect = RuntimeError("index missing")

        with patch("agents.news_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_search_agent import run_news_search_agent
            with pytest.raises(RuntimeError, match="index missing"):
                run_news_search_agent("query", category_result)

    def test_query_used_matches_input(self, category_result, mock_es_client, es_search_response):
        mock_es_client.search.return_value = es_search_response
        user_query = "What is happening in quantum computing?"

        with patch("agents.news_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_search_agent import run_news_search_agent
            result = run_news_search_agent(user_query, category_result)

        assert result.query_used == user_query

    def test_search_called_with_correct_index(
        self, category_result, mock_es_client, es_search_response
    ):
        mock_es_client.search.return_value = es_search_response

        with patch("agents.news_search_agent.get_es_client", return_value=mock_es_client):
            from agents.news_search_agent import run_news_search_agent, NEWS_INDEX
            run_news_search_agent("tech", category_result)

        call_kwargs = mock_es_client.search.call_args.kwargs
        assert call_kwargs.get("index") == NEWS_INDEX