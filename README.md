# News Search & Summary App

A **FastAPI-based multi-agent RAG application** that accepts natural-language news queries,
performs semantic search over an Elasticsearch index, and returns an AI-generated summary
together with the source articles that informed it.

This service is the **user-facing query layer** in a three-part news intelligence pipeline:

```
┌──────────────────────────┐      Kafka (Confluent Cloud)      ┌─────────────────────────────┐
│  news_fetch_scheduler    │ ─────────────────────────────────► │  news-consumer-ingester     │
│  • RSS / API ingestion   │                                     │  • Full-article fetch       │
│  • Dedup (warm cache +   │                                     │  • Elasticsearch indexing   │
│    PostgreSQL)           │                                     │  • Embedding pipeline       │
│  • Confluent publish     │                                     │    (news-embedding-endpoint)│
└──────────────────────────┘                                     └─────────────┬───────────────┘
                                                                               │ Elasticsearch
                                                                               ▼  (news_articles)
                                                                 ┌─────────────────────────────┐
                                                                 │  news_search_summary_app    │  ◄── YOU ARE HERE
                                                                 │  • Category extraction LLM  │
                                                                 │  • Semantic search (ES)     │
                                                                 │  • GPT-4o-mini summary      │
                                                                 │  • FastAPI  port 8003       │
                                                                 └─────────────────────────────┘
```

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Local Infrastructure Setup](#local-infrastructure-setup)
4. [Building the Private Wheel (py_commons_per)](#building-the-private-wheel-py_commons_per)
5. [Running the Application](#running-the-application)
6. [API Endpoints](#api-endpoints)
7. [Chat UI](#chat-ui)
8. [Configuration](#configuration)
9. [Running Tests](#running-tests)
10. [Docker](#docker)
11. [Demo](#demo)

---

## Architecture

### Multi-Agent Pipeline

```
User Query (POST /query)
        │
        ▼
┌───────────────────────┐
│  NewsCategorySearch   │  LangChain LCEL + gpt-4o-mini
│  Agent                │  Extracts topic categories &
│                       │  fetch_latest intent from query
└──────────┬────────────┘
           │  CategorySearchResult
           ▼
┌───────────────────────┐
│  NewsSearch Agent     │  Elasticsearch semantic_text query
│                       │  Server-side embeddings via
│                       │  news-embedding-endpoint inference
└──────────┬────────────┘
           │  NewsSearchResult  (up to 8 ArticleHit objects)
           ▼
┌───────────────────────┐
│  NewsSummary Agent    │  LangChain LCEL + gpt-4o-mini
│                       │  Produces a 3–5 paragraph factual
│                       │  summary with inline citations
└──────────┬────────────┘
           │  AgentResponse
           ▼
    JSON response to client
```

### Key Components

| Module                                     | Responsibility                                                           |
|--------------------------------------------|--------------------------------------------------------------------------|
| `src/api/app.py`                           | FastAPI application, request validation, CORS, static file serving       |
| `src/agents/manager_agent.py`              | Orchestrates the three-agent pipeline, error handling, LangSmith tracing |
| `src/agents/news_category_search_agent.py` | LLM intent parsing + ES category verification                            |
| `src/agents/news_search_agent.py`          | Elasticsearch semantic search query builder and executor                 |
| `src/agents/news_summary_agent.py`         | GPT-4o-mini summarization chain                                          |
| `src/agents/base.py`                       | Shared ES client factory, data models                                    |
| `src/config/elastic_setup.py`              | Properties-file reader for Elasticsearch connection config               |
| `static/chat-ui.html`                      | Self-contained browser chat interface                                    |

---

## Prerequisites

### External Services

| Service                        | Purpose                                                                                                                             | Default address         |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| **HashiCorp Vault**            | Stores `ELASTIC_URL`, `ELASTIC_API_KEY`, `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`                                                      | `http://localhost:8200` |
| **PyPI server** (`pypiserver`) | Hosts the private `py-commons-per` wheel                                                                                            | `http://localhost:8080` |
| **PostgreSQL**                 | Used by `news_fetch_scheduler` for article deduplication (not directly by this service, but required in the shared `infra-network`) | `localhost:5432`        |

### Runtime Requirements

- Python **3.12**
- Docker & Docker Compose (for containerised runs)
- Access to **Elastic Cloud** (Elasticsearch 8.x with a semantic inference endpoint named `news-embedding-endpoint`)
- **OpenAI API key** (gpt-4o-mini)
- Optional: LangSmith account for end-to-end tracing

---

## Local Infrastructure Setup

All three infrastructure services must run inside shared Docker networks before starting this app.

### 1 — Create shared Docker networks

```bash
docker network create vault-network
docker network create infra-network
```

### 2 — Start HashiCorp Vault

```bash
docker run -d \
  --name vault-server \
  --network vault-network \
  -p 8200:8200 \
  -e 'VAULT_DEV_ROOT_TOKEN_ID=dev-root-token' \
  -e 'VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200' \
  hashicorp/vault:latest server -dev
```

Populate the secrets this app expects:

```bash
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=dev-root-token

vault kv put secret/NEWS_SUMMARY_APP \
  ELASTIC_URL="https://<your-elastic-cloud-endpoint>" \
  ELASTIC_API_KEY="<your-elastic-api-key>" \
  OPENAI_API_KEY="<your-openai-api-key>" \
  LANGCHAIN_API_KEY="<your-langsmith-api-key>" \
  LANGCHAIN_TRACING_V2="true" \
  LANGCHAIN_PROJECT="news-summary-app"
```

### 3 — Start the local PyPI server

```bash
# Create the package directory first
mkdir -p ./pypi-packages

docker run -d \
  --name pypi-server \
  --network infra-network \
  -p 8080:8080 \
  -v $(pwd)/pypi-packages:/data/packages \
  pypiserver/pypiserver:latest \
  run -p 8080 /data/packages
```

### 4 — Start PostgreSQL (required by the ingestion pipeline)

```bash
docker run -d \
  --name postgres-server \
  --network infra-network \
  -p 5432:5432 \
  -e POSTGRES_USER=news \
  -e POSTGRES_PASSWORD=news \
  -e POSTGRES_DB=newsdb \
  postgres:16-alpine
```

---

## Building the Private Wheel (py_commons_per)

`py-commons-per` is a private shared library that provides Vault secret loading
(`load_secrets`) and structured logging setup (`setup_logging`).
It must be built from source and uploaded to the local PyPI server before installing this app.

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-org>/py_commons_per.git
cd py_commons_per
```

### Step 2 — Build the wheel

```bash
pip install build
python -m build --wheel
# Wheel created in dist/  e.g. py_commons_per-1.0.0-py3-none-any.whl
```

### Step 3 — Upload to the local PyPI server

```bash
pip install twine
twine upload \
  --repository-url http://localhost:8080 \
  --username "" --password "" \
  dist/py_commons_per-*.whl
```

Alternatively, copy the wheel directly into the volume directory:

```bash
cp dist/py_commons_per-*.whl ./pypi-packages/
```

---

## Running the Application

### Option A — Local (virtualenv)

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install public dependencies
pip install -r requirements.txt

# 3. Install the private wheel from local PyPI
pip install \
  --trusted-host localhost \
  --index-url http://localhost:8080/simple/ \
  py-commons-per

# 4. Set Vault environment variables
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=dev-root-token
export VAULT_SECRET_PATH=secret/NEWS_SUMMARY_APP

# 5. Start the server (run from repo root so static/ is on the path)
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8003 --reload
```

The API will be available at `http://localhost:8003`.

### Option B — Docker Compose

```bash
# Export the Vault token so docker-compose can inject it
export VAULT_TOKEN=dev-root-token

docker compose up --build
```

The `docker-compose.yml` attaches the container to `vault-network` and
`infra-network` so it can reach `vault-server` and the Elasticsearch cluster.

---

## API Endpoints

| Method  | Path      | Description                                                       |
|---------|-----------|-------------------------------------------------------------------|
| `GET`   | `/`       | Serves the Chat UI (`static/chat-ui.html`)                        |
| `GET`   | `/health` | Liveness probe — always returns `{"status": "ok"}`                |
| `GET`   | `/ready`  | Readiness probe — pings Elasticsearch; returns 503 if unreachable |
| `POST`  | `/query`  | Runs the full multi-agent pipeline and returns a news summary     |

### POST /query — Request

```json
{
  "query": "What are the latest developments in AI regulation?"
}
```

**Validation:** `query` must be at least 3 characters.

### POST /query — Response

```json
{
  "query": "What are the latest developments in AI regulation?",
  "categories": ["technology", "ai", "politics"],
  "summary": "Regulators across the EU and US are accelerating AI oversight...",
  "articles": [
    {
      "article_id": "abc-123",
      "title": "EU AI Act Enters Final Phase",
      "source": "EuroTech",
      "published_at": "2026-03-28T09:00:00Z",
      "link": "https://eurotech.example.com/eu-ai-act"
    }
  ],
  "duration_seconds": 2.341,
  "error": null
}
```

### Interactive API Docs

FastAPI generates live docs at:

- **Swagger UI:** `http://localhost:8003/docs`
- **ReDoc:** `http://localhost:8003/redoc`

---

## Chat UI

A self-contained single-page chat interface is bundled at `static/chat-ui.html`.

### Access via the server (recommended)

Once the FastAPI server is running, open your browser at:

```
http://localhost:8003/
```

The server mounts `static/` and serves `chat-ui.html` at the root path `/`.
No additional configuration is needed — the UI will POST queries to `http://127.0.0.1:8003/query` automatically.

### Access as a standalone file (without the server)

```bash
# macOS
open static/chat-ui.html

# Linux
xdg-open static/chat-ui.html
```

> **Note:** When opened from disk the UI still POSTs to `http://127.0.0.1:8003/query`,
> so the FastAPI server must be running for queries to work.

### Chat UI features

- Dark-themed terminal-style interface with CSS animations
- Markdown rendering — headers, bold, code blocks, ordered/unordered lists
- Animated typing indicator while the pipeline is processing
- Structured response display: summary, categories, source articles, duration
- Session management — type `bye` to gracefully end the session
- Keyboard shortcuts: `Enter` to send, `Shift+Enter` for a new line

---

## Configuration

### Vault secrets (`secret/NEWS_SUMMARY_APP`)

| Secret key             | Required                   | Description                                |
|------------------------|----------------------------|--------------------------------------------|
| `ELASTIC_URL`          | Yes                        | Elastic Cloud HTTPS endpoint               |
| `ELASTIC_API_KEY`      | Yes                        | Elastic Cloud API key                      |
| `OPENAI_API_KEY`       | Yes                        | OpenAI key for gpt-4o-mini                 |
| `LANGCHAIN_API_KEY`    | No                         | LangSmith API key for tracing              |
| `LANGCHAIN_TRACING_V2` | No                         | Set to `"true"` to enable LangSmith traces |
| `LANGCHAIN_PROJECT`    | No                         | LangSmith project name                     |
| `VAULT_ADDR`           | HashiCorp Vault address    | `http://vault:8200`                        |
| `VAULT_TOKEN`          | Vault token                | —                                          |
| `VAULT_SECRET_PATH`    | KV v2 path for app secrets | `secret/data/news-summary`                 |

### Elasticsearch connection (`src/resources/elastic-config.properties`)

```properties
request_timeout=60
retry_on_timeout=True
max_retries=3
http_compress=True
connections_per_node=10
```

### Elasticsearch index requirements

| Setting            | Value                                                              |
|--------------------|--------------------------------------------------------------------|
| Index name         | `news_articles`                                                    |
| Semantic field     | `content.semantic` (type `semantic_text`)                          |
| Inference endpoint | `news-embedding-endpoint`                                          |
| Source fields      | `article_id`, `title`, `content`, `source`, `published_at`, `link` |

---

## Running Tests

### Install test dependencies

```bash
pip install pytest httpx
```

> `httpx` is required by FastAPI's `TestClient`.
> `langsmith` and `py-commons-per` are fully stubbed out in `tests/conftest.py`,
> so the **private wheel is not required** to run the test suite.

### Run all tests

```bash
pytest
```

### Run a specific test module

```bash
pytest tests/agents/news_search_agent_test.py -v
```

### Run with coverage report

```bash
pip install pytest-cov
pytest --cov=src --cov-report=term-missing
```

### Test structure

```
tests/
├── conftest.py                              # shared fixtures + py_commons_per / langsmith stubs
├── agents/
│   ├── base_test.py                         # ES client factory, all dataclasses
│   ├── manager_agent_test.py                # pipeline orchestration, per-stage failure paths
│   ├── news_category_search_agent_test.py   # LLM intent parsing, ES category verification
│   ├── news_search_agent_test.py            # query builders, hit parser, fetch_latest path
│   └── news_summary_agent_test.py           # context builder, LLM chain invocation
├── api/
│   └── app_test.py                          # all FastAPI endpoints via TestClient
└── config/
    └── elastic_setup_test.py                # properties file parsing edge cases
```

---

## Docker

### Build the image

```bash
docker build -t news-search-summary-app:latest .
```

The Dockerfile uses a two-stage build (builder + runtime) with a non-root `appuser`
and exposes port `8003`.

### Run standalone

```bash
docker run -d \
  --name news-search-summary-app \
  --network vault-network \
  --network infra-network \
  -p 8003:8003 \
  -e VAULT_ADDR=http://vault-server:8200 \
  -e VAULT_TOKEN=dev-root-token \
  -e VAULT_SECRET_PATH=secret/NEWS_SUMMARY_APP \
  news-search-summary-app:latest
```

### Docker Compose

```bash
export VAULT_TOKEN=dev-root-token
docker compose up --build -d

# Tail logs
docker compose logs -f news-summary-app

# Stop
docker compose down
```

The service performs a health check every 60 s against `GET /health` and restarts
automatically (`unless-stopped`) if it crashes.

---

## Demo

Screenshots and screen recordings live in the `demo/` folder.

| File                                           | Description                                                |
|------------------------------------------------|------------------------------------------------------------|
| `demo/user_chat_interaction_live_session.webm` | End-to-end screen recording: startup → query → response    |
| `demo/services_running_on_locale_docker.png`   | Services running in local docker                           |
| `demo/user_chat_interaction_logs.png`          | User chat interaction logs                                 |



> **Capturing recordings:**
>
> Screenshots — any browser screenshot tool, or `flameshot` on Linux.
>
> Screen recordings (.webm) — OBS Studio, `simplescreenrecorder`, or:
> ```bash
> ffmpeg -f x11grab -r 25 -s 1920x1080 -i :0.0 \
>   -c:v libvpx-vp9 demo/full-walkthrough.webm
> ```

---

## Project Structure

```
news_search_summary_app/
├── demo/                        # Screenshots and screen recordings
├── src/
│   ├── agents/
│   │   ├── base.py              # Shared ES client and data models
│   │   ├── manager_agent.py     # Pipeline orchestrator
│   │   ├── news_category_search_agent.py
│   │   ├── news_search_agent.py
│   │   └── news_summary_agent.py
│   ├── api/
│   │   └── app.py               # FastAPI application
│   ├── config/
│   │   └── elastic_setup.py     # Properties file reader
│   ├── resources/
│   │   └── elastic-config.properties
│   └── main.py                  # Uvicorn entry point
├── static/
│   └── chat-ui.html             # Browser chat interface (served at /)
├── tests/
│   ├── conftest.py
│   ├── agents/
│   │   ├── base_test.py
│   │   ├── manager_agent_test.py
│   │   ├── news_category_search_agent_test.py
│   │   ├── news_search_agent_test.py
│   │   └── news_summary_agent_test.py
│   ├── api/
│   │   └── app_test.py
│   └── config/
│       └── elastic_setup_test.py
├── Dockerfile
├── docker-compose.yml
├── logging_config.json
├── pytest.ini
├── requirements.txt             # Public dependencies
└── requirements_locales.txt     # Private wheel (py-commons-per via local PyPI)
```

---

## Troubleshooting

| Symptom                                                       | Likely cause                  | Fix                                                                                                       |
|---------------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------|
| `RuntimeError: ELASTIC_URL and ELASTIC_API_KEY must be set`   | Vault secrets not loaded      | Verify `VAULT_ADDR`, `VAULT_TOKEN`, and `VAULT_SECRET_PATH` env vars                                      |
| `GET /ready` returns 503                                      | Elasticsearch unreachable     | Check Elastic Cloud endpoint, API key, and network connectivity                                           |
| `ConnectionError` on startup                                  | Docker network missing        | `docker network create vault-network && docker network create infra-network`                              |
| `ModuleNotFoundError: py_commons_per`                         | Private wheel not installed   | Build and upload the wheel — see [Building the Private Wheel](uilding-the-private-wheel-py_commons_p)     |
| Chat UI shows "Could not reach the server"                    | App not running or wrong port | Start the FastAPI server on port 8003                                                                     |
| LangSmith traces not appearing                                | Missing env vars              | Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in Vault                                          |
| `422 Unprocessable Entity` on `/query`                        | Query string too short        | Minimum query length is 3 characters                                                                      |

---

## Author

**Srini Yedluri**