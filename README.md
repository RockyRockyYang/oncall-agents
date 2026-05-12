# OnCall Agents

An AI-powered on-call assistant that answers operational questions by retrieving relevant context from your runbooks. Built with Claude (Anthropic), Voyage AI embeddings, and Milvus as the vector store.

## How it works

1. **Ingest** — runbook markdown files in `docs/` are chunked, embedded via Voyage AI (`voyage-3-lite`), and stored in a Milvus collection (`oncall_kb`).
2. **Query** — at query time the question is embedded and a cosine similarity search retrieves the top-k relevant chunks.
3. **Answer** — the retrieved context is passed to Claude (`claude-sonnet-4-6`) to generate a grounded, runbook-backed answer.
4. **MCP tools** — live system metrics (CPU, memory, processes) are exposed via a separate MCP server and loaded dynamically into the agent at startup.

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (Sonnet) |
| Embeddings | Voyage AI `voyage-3-lite` (512-dim) |
| Vector DB | Milvus v2.4 (standalone via Docker) |
| API | FastAPI + uvicorn |
| Agent orchestration | LangChain / LangGraph |
| External tools | MCP (FastMCP + psutil) |

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- API keys: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`

## Setup

```bash
# 1. Start Milvus (and its dependencies etcd + MinIO)
docker compose up -d

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Copy the env template and fill in your keys
cp .env.example .env   # edit ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

## Run the API

```bash
# 1. Start the MCP monitor server (separate process)
python mcp_servers/monitor_server.py

# 2. Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload
```

The MCP monitor server must be running before starting the FastAPI server — the agent connects to it during startup.

## Testing

All tests are integration tests and require:
- Milvus running (`docker compose up -d`)
- MCP monitor server running (`python mcp_servers/monitor_server.py`)
- Valid API keys in `.env`

```bash
# Run all integration tests
pytest tests/ -v -m integration

# Run a specific test file
pytest tests/test_chat.py -v -m integration
pytest tests/test_mcp_server.py -v -m integration
pytest tests/test_dialogue.py -v -m integration
```

### Test files

| File | What it tests |
|---|---|
| `tests/test_chat.py` | SSE streaming, session memory via HTTP |
| `tests/test_mcp_server.py` | MCP tool loading and responses |
| `tests/test_dialogue.py` | Multi-turn conversation context |
| `tests/test_ingest.py` | Document ingestion endpoint |
| `tests/test_tools.py` | Local LangChain tools |
| `tests/test_vector_store.py` | Milvus embed + search pipeline |

## Configuration

All settings are in `app/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `VOYAGE_API_KEY` | — | Required |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |
| `RAG_MODEL` | `claude-sonnet-4-6` | Claude model for answer generation |
| `RAG_TOP_K` | `3` | Number of chunks to retrieve |
| `CHUNK_MAX_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `PORT` | `9900` | API server port |
| `MCP_MONITOR_URL` | `http://localhost:8004/mcp` | MCP monitor server URL |

## Project structure

```
app/
  agent/      # LangGraph agent definitions
  api/        # FastAPI route handlers
  core/       # Shared infrastructure (Milvus client)
  services/   # Business logic
  tools/      # LangChain tools
  config.py   # Settings
  main.py     # FastAPI app entrypoint
mcp_servers/  # Standalone MCP tool servers
docs/         # Runbook markdown files
tests/        # pytest integration tests
docker-compose.yml
pyproject.toml
```
