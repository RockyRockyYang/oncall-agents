# OnCall Agents

An AI-powered on-call assistant that answers operational questions by retrieving relevant context from your runbooks. Built with Claude (Anthropic), OpenAI embeddings, and PostgreSQL + pgvector as the vector store. Includes a React web UI (chat + AI Ops investigation panel + knowledge base upload).

## How it works

1. **Ingest** — runbook markdown files in `docs/` are chunked, embedded via OpenAI (`text-embedding-3-small`), and stored in a PostgreSQL + pgvector collection (`oncall_kb`).
2. **Query** — at query time the question is embedded and a cosine similarity search retrieves the top-k relevant chunks.
3. **Answer** — the retrieved context is passed to Claude (`claude-sonnet-4-6`) to generate a grounded, runbook-backed answer.
4. **MCP tools** — live system metrics (CPU, memory, processes) are exposed via a separate MCP server and loaded dynamically into the agent at startup.

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (Sonnet) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector DB | PostgreSQL 16 + pgvector (via Docker) |
| API | FastAPI + uvicorn |
| Agent orchestration | LangChain / LangGraph |
| External tools | MCP (FastMCP + psutil) |
| Frontend | React + TypeScript + Vite + MUI |

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Node.js (LTS) — for the frontend
- Docker & Docker Compose
- API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

## Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Start PostgreSQL + pgvector
docker compose up -d

# 3. Install dependencies (creates .venv automatically)
uv sync

# 4. Copy the env template and fill in your keys
cp .env.example .env   # edit ANTHROPIC_API_KEY and OPENAI_API_KEY
```

## Run the API

```bash
# 1. Start the MCP monitor server (separate process)
uv run mcp_servers/monitor_server.py

# 2. Start the MCP logs server (separate process)
uv run mcp_servers/logs_server.py

# 3. Start the FastAPI server
uv run oncall-api
```

Both MCP servers must be running before starting the FastAPI server — the agent connects to them during startup. Skipping the logs server won't error at startup, but any log/error-summary/deployment tool call will silently return "no data found".

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens at `localhost:5173`. The dev server proxies `/api/*` to the FastAPI backend at `localhost:9900`
(see `frontend/vite.config.ts`), so the backend (and MCP servers) need to be running first.

## Testing

All tests are integration tests and require:
- PostgreSQL running (`docker compose up -d`)
- MCP monitor server running (`uv run python mcp_servers/monitor_server.py`)
- MCP logs server running (`uv run python mcp_servers/logs_server.py`)
- Valid API keys in `.env`

```bash
# Run all integration tests
uv run python -m pytest tests/ -v -m integration

# Run a specific test file
uv run python -m pytest tests/api/test_chat.py -v
uv run python -m pytest tests/mcp/test_monitor_server.py -v
uv run python -m pytest tests/services/test_vector_store.py -v
```

### Test files

| File | What it tests |
|---|---|
| `tests/api/test_chat.py` | SSE streaming, session memory via HTTP |
| `tests/api/test_dialogue.py` | Multi-turn conversation context |
| `tests/api/test_ingest.py` | Document ingestion endpoint |
| `tests/mcp/test_monitor_server.py` | MCP tool loading and responses |
| `tests/mcp/test_logs_server.py` | Log search MCP tools |
| `tests/tools/test_tools.py` | Local LangChain tools |
| `tests/services/test_vector_store.py` | pgvector embed + search pipeline |
| `tests/agent/aiops/` | AIOps planner, executor, replanner |
| `tests/e2e/test_aiops_e2e.py` | Full AIOps investigation workflow |

## Configuration

All settings are in `app/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `OPENAI_API_KEY` | — | Required (embeddings) |
| `DATABASE_URL` | `postgresql+psycopg://oncall:oncall@localhost:5432/oncall` | PostgreSQL connection string |
| `RAG_MODEL` | `claude-sonnet-4-6` | Claude model for answer generation |
| `RAG_TOP_K` | `3` | Number of chunks to retrieve |
| `CHUNK_MAX_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `PORT` | `9900` | API server port |
| `MCP_MONITOR_URL` | `http://localhost:8004/mcp` | MCP monitor server URL |
| `MCP_LOGS_URL` | `http://localhost:8003/mcp` | MCP logs server URL |

## Project structure

```
app/
  agent/      # LangGraph agent definitions
  api/        # FastAPI route handlers
  core/       # Shared infrastructure
  services/   # Business logic (vector store, RAG, AIOps)
  tools/      # LangChain tools
  config.py   # Settings
  main.py     # FastAPI app entrypoint
mcp_servers/  # Standalone MCP tool servers
docs/         # Runbook markdown files
tests/        # pytest tests (api/, services/, agent/, mcp/, e2e/)
frontend/     # React + TypeScript + Vite web UI
  src/
    components/  # Sidebar, ChatArea, AIOpsPanel, KnowledgeBaseDialog
    hooks/       # useChat, useAIOps (SSE streaming)
    lib/         # session.ts (session id + local history)
docker-compose.yml
pyproject.toml
```
