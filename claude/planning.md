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

### ✅ Step 5 — VectorStoreService
**File:** `app/services/vector_store.py`

Wrap test_milvus.py into a reusable class:

| Method | Description |
|--------|-------------|
| `__init__` | Connect to Milvus + VoyageAI, call `_ensure_collection()` |
| `_ensure_collection()` | Create collection with explicit schema if not exists |
| `ingest(chunks, source)` | Embed chunks, insert rows |
| `search(query, top_k) → list[str]` | Embed query, return matching content strings |

---

### ✅ Step 6 — LangChain Retrieval Tool
**File:** `app/tools/retrieval.py`

Wrap `VectorStoreService.search()` as a LangChain `@tool` so the agent can call it:

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the on-call knowledge base for relevant runbook content."""
```

Update `app/tools/__init__.py` to export it.

---

### ✅ Step 7 — LangGraph RAG Agent
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

### ✅ Step 8 — Session Memory
**File:** `app/agent/rag_agent.py` (update)

Add `MemorySaver` checkpointer — conversations persist across turns per `thread_id`:

```python
from langgraph.checkpoint.memory import MemorySaver
agent = graph.compile(checkpointer=MemorySaver())
# invoke with: config={"configurable": {"thread_id": session_id}}
```

---

### ✅ Step 9 — FastAPI App + SSE Streaming
**Files:** `app/main.py`, `app/api/chat.py`

- `POST /chat` — body: `{ "message": str, "session_id": str }`
- Stream tokens back using `agent.astream_events(...)` + SSE
- `app/main.py` initializes FastAPI and mounts routes
- Run: `uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload`

---

### ✅ Step 10 — Ingest Endpoint
**File:** `app/api/ingest.py`

- `POST /ingest` — accept a file path or raw markdown text
- Chunk by double-newline, call `VectorStoreService.ingest()`
- Returns count of inserted chunks
- Allows loading new runbooks without restarting the server

---

### ✅ Step 11 — End-to-End Test
1. `docker-compose up -d` — start Milvus
2. `POST /ingest` with `docs/high_cpu.md`
3. `POST /chat` with `"how do I find a runaway process?"`
4. Verify response streams back with runbook content

---

## Files Summary

| File | Status | Step |
|------|--------|------|
| `app/services/vector_store.py` | ✅ Done | 5 |
| `app/tools/retrieval.py` | ✅ Done | 6 |
| `app/tools/__init__.py` | ✅ Done | 6 |
| `app/agent/rag_agent.py` | ✅ Done | 7–8 |
| `app/agent/__init__.py` | ✅ Done | 7 |
| `app/api/chat.py` | ✅ Done | 9 |
| `app/main.py` | ✅ Done | 9 |
| `app/api/ingest.py` | ✅ Done | 10 |

---

---

## Phase 2 — Dialogue Agent

Our Phase 1 agent already handles multi-turn conversation via `MemorySaver`. Phase 2 enhances it with three things:
1. **MCP tools** — the agent can call external tool servers (the main new concept)
2. **Dialogue improvements** — message trimming, session management API
3. **Additional local tools** — time tool

### What is MCP?
MCP (Model Context Protocol) is a standard for connecting LLMs to external tool servers. Instead of writing tools as Python functions inside the agent, you run a separate process that exposes tools over HTTP. The agent connects to it and loads tools dynamically at startup. This decouples tools from the agent — you can add or update tools without changing agent code.

### New Dependencies
Before starting, add to `pyproject.toml`:
- `fastmcp` — Python library for building MCP servers
- `langchain-mcp-adapters` — connects LangChain agents to MCP servers
- `psutil` — system metrics (CPU, memory, processes)

---

### ✅ Step 12 — Dialogue Improvements
**Files:** `app/tools/time_tool.py`, `app/agent/rag_agent.py`, `app/api/chat.py`

Three small improvements bundled together:

**Time tool** — local tool so the agent knows the current time during diagnosis:
```python
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time."""
```

**Message trimming** — prevents context overflow in long conversations. Added inside `llm_node`, keeps system message + last 10 messages:
```python
def _trim_messages(messages: list) -> list:
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return system + rest[-10:]
```

**Session management API** — two new endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /chat/session/{session_id}` | Return message history for a session |
| `DELETE /chat/session/{session_id}` | Clear session history |

Reading history requires reading directly from the `MemorySaver` checkpointer.

---

### ✅ Step 13 — MCP Server (Monitor)
**File:** `mcp_servers/monitor_server.py`

Build a standalone MCP server exposing system metrics as tools. Runs as a separate process on port 8004.

Tools to expose:
- `get_cpu_usage()` — current CPU % via `psutil`
- `get_memory_usage()` — memory stats (total, used, %)
- `list_top_processes(limit)` — top N processes by CPU usage

Built with FastMCP:
```python
from fastmcp import FastMCP
mcp = FastMCP("monitor")

@mcp.tool()
def get_cpu_usage() -> str:
    """Get current CPU usage percentage."""
```

Start with: `python mcp_servers/monitor_server.py`
Test independently with curl before wiring into the agent.

---

### ✅ Step 14 — MCP Client
**File:** `app/agent/mcp_client.py`

Singleton client that connects to MCP servers using `langchain-mcp-adapters`.

Key concepts:
- `MultiServerMCPClient` — connects to one or more MCP servers over HTTP
- Tools loaded async: `await client.get_tools()` returns standard LangChain tools
- Retry interceptor: wraps every tool call with exponential backoff (1s, 2s, 4s, max 3 retries) — returns error string instead of raising

Config in `.env`:
```
MCP_MONITOR_URL=http://localhost:8004/mcp
```
Add `mcp_monitor_url` to `app/config.py`.

---

### ✅ Step 15 — FastAPI Lifespan
**File:** `app/main.py` (update)

When MCP tools are added, agent initialization must become async (MCP requires an active event loop to connect). This means the agent can no longer be compiled at module load time.

FastAPI's lifespan context manager handles this — it runs startup logic before the server accepts requests:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()  # connect to MCP, load tools, compile graph
    yield
    await agent.shutdown()    # cleanup MCP connections

app = FastAPI(lifespan=lifespan)
```

This is the standard production pattern for any async initialization (DB connections, external services, etc.).

---

### ✅ Step 16 — Wire MCP Tools into Agent
**File:** `app/agent/rag_agent.py` (update)

Refactor agent from module-level compilation to a class with async `initialize()`:

```python
class RagAgent:
    async def initialize(self) -> None:
        mcp_tools = await mcp_client.get_tools()
        all_tools = [search_knowledge_base, get_current_time] + mcp_tools
        llm = ChatAnthropic(...).bind_tools(all_tools)
        # rebuild graph with all_tools
        self.graph = build_graph(llm, all_tools)
```

The agent now has access to runbook search + live system metrics from MCP.

---

### ✅ Step 17 — Tests + End-to-End
- `tests/test_mcp_server.py` — call MCP tools directly, verify responses
- `tests/test_dialogue.py` — multi-turn conversation, verify context is preserved across turns
- End-to-end: ask "what is the current CPU usage?" → agent calls MCP tool → answers with real data

---

## Phase 2 Files Summary

| File | Status | Step |
|------|--------|------|
| `app/tools/time_tool.py` | ✅ Done | 12 |
| `app/agent/rag_agent.py` | ✅ Done (update) | 12, 16 |
| `app/api/chat.py` | ✅ Done (update) | 12 |
| `mcp_servers/monitor_server.py` | ✅ Done | 13 |
| `app/agent/mcp_client.py` | ✅ Done | 14 |
| `app/config.py` | ✅ Done (update) | 14 |
| `app/main.py` | ✅ Done (update) | 15 |
| `tests/test_mcp_server.py` | ✅ Done | 17 |
| `tests/test_dialogue.py` | ✅ Done | 17 |

---

## Phase 3 — Operation Agent _(future)_

AIOps Plan-Execute-Replan workflow for structured incident diagnosis.
- Planner: queries KB + generates step-by-step diagnosis plan
- Executor: runs each step using tools
- Replanner: decides continue / replan / respond (max 8 steps)
- Final output: structured markdown diagnostic report
