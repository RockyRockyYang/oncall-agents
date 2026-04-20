# OnCall Agents

An AI-powered on-call assistant that answers operational questions by retrieving relevant context from your runbooks. Built with Claude (Anthropic), Voyage AI embeddings, and Milvus as the vector store.

## How it works

1. **Ingest** — runbook markdown files in `docs/` are chunked, embedded via Voyage AI (`voyage-3-lite`), and stored in a Milvus collection (`oncall_kb`).
2. **Query** — at query time the question is embedded and a cosine similarity search retrieves the top-k relevant chunks.
3. **Answer** — the retrieved context is passed to Claude (`claude-sonnet-4-6`) to generate a grounded, runbook-backed answer.

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (Sonnet) |
| Embeddings | Voyage AI `voyage-3-lite` (512-dim) |
| Vector DB | Milvus v2.4 (standalone via Docker) |
| API | FastAPI + uvicorn |
| Agent orchestration | LangChain / LangGraph |

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

## Testing

```bash
pytest tests/ -v
```

Tests are integration tests that require Milvus running and valid API keys in `.env`.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload
```

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
docs/         # Runbook markdown files
tests/        # pytest integration tests
docker-compose.yml
pyproject.toml
```
