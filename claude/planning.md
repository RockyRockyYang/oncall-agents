# On-Call Agent — Build Plan

## Overview

Building a RAG-based on-call assistant using:
**Claude API** · **VoyageAI** (embeddings) · **Milvus** (vector DB) · **LangGraph** · **FastAPI**

Three phases planned: RAG Agent → Dialogue Agent → Operation Agent

---

## Phase 1 — RAG Agent

### ✅ Step 1 — Project Setup
- [x] `pyproject.toml` with all dependencies (langchain, langgraph, pymilvus, voyageai, fastapi)
- [x] `.env` with API keys and service config
- [x] `app/config.py` — Pydantic `Settings` singleton
- [x] `app/core/milvus_client.py` — Milvus singleton
- [x] `docker-compose.yml` — Milvus + etcd + minio
- [x] `docs/high_cpu.md` — first runbook document

---

### ✅ Step 2 — Verify Claude API
- [x] `test_claude.py` — confirmed Claude API key works and model responds

---

### ✅ Step 3 — Verify VoyageAI Embeddings
- [x] `test_embeddings.py` — confirmed `voyage-3-lite` returns 512-dim vectors

---

### ✅ Step 4 — Verify Milvus Vector Store
- [x] `test_milvus.py` — end-to-end: chunk → embed → insert → search
- [x] Confirmed explicit schema required (pymilvus 2.6.x)
- [x] Confirmed `load_collection()` needed before search

---

### ⬜ Step 5 — VectorStoreService
**File:** `app/services/vector_store.py`

Wrap test_milvus.py into a reusable class:

| Method | Description |
|--------|-------------|
| `__init__` | Connect to Milvus + VoyageAI, call `_ensure_collection()` |
| `_ensure_collection()` | Create collection with explicit schema if not exists |
| `ingest(chunks, source)` | Embed chunks, insert rows |
| `search(query, top_k) → list[str]` | Embed query, return matching content strings |

---

### ⬜ Step 6 — LangChain Retrieval Tool
**File:** `app/tools/retrieval.py`

Wrap `VectorStoreService.search()` as a LangChain `@tool` so the agent can call it:

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the on-call knowledge base for relevant runbook content."""
```

Update `app/tools/__init__.py` to export it.

---

### ⬜ Step 7 — LangGraph RAG Agent
**File:** `app/agent/rag_agent.py`

Build a two-node LangGraph graph:

```
START → llm_node → [has tool calls?] → tools_node → llm_node → END
                 ↘ [no tool calls]  → END
```

- `llm_node`: calls Claude with `search_knowledge_base` bound as a tool
- `tools_node`: executes tool calls via LangGraph's `ToolNode`
- System prompt: "you are an on-call assistant, use the knowledge base to answer"

---

### ⬜ Step 8 — Session Memory
**File:** `app/agent/rag_agent.py` (update)

Add `MemorySaver` checkpointer — conversations persist across turns per `thread_id`:

```python
from langgraph.checkpoint.memory import MemorySaver
agent = graph.compile(checkpointer=MemorySaver())
# invoke with: config={"configurable": {"thread_id": session_id}}
```

---

### ⬜ Step 9 — FastAPI App + SSE Streaming
**Files:** `app/main.py`, `app/api/chat.py`

- `POST /chat` — body: `{ "message": str, "session_id": str }`
- Stream tokens back using `agent.astream_events(...)` + SSE
- `app/main.py` initializes FastAPI and mounts routes
- Run: `uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload`

---

### ⬜ Step 10 — Ingest Endpoint
**File:** `app/api/ingest.py`

- `POST /ingest` — accept a file path or raw markdown text
- Chunk by double-newline, call `VectorStoreService.ingest()`
- Returns count of inserted chunks
- Allows loading new runbooks without restarting the server

---

### ⬜ Step 11 — End-to-End Test
1. `docker-compose up -d` — start Milvus
2. `POST /ingest` with `docs/high_cpu.md`
3. `POST /chat` with `"how do I find a runaway process?"`
4. Verify response streams back with runbook content

---

## Files Summary

| File | Status | Step |
|------|--------|------|
| `app/services/vector_store.py` | ⬜ TODO | 5 |
| `app/tools/retrieval.py` | ⬜ TODO | 6 |
| `app/tools/__init__.py` | ⬜ TODO | 6 |
| `app/agent/rag_agent.py` | ⬜ TODO | 7–8 |
| `app/agent/__init__.py` | ⬜ TODO | 7 |
| `app/api/chat.py` | ⬜ TODO | 9 |
| `app/main.py` | ⬜ TODO | 9 |
| `app/api/ingest.py` | ⬜ TODO | 10 |

---

## Phase 2 — Dialogue Agent _(future)_

Multi-turn conversation with clarification and context tracking.

## Phase 3 — Operation Agent _(future)_

Executes shell commands and calls external APIs to resolve incidents.
