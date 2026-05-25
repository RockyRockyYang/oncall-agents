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

## Phase 3 — Operation Agent

AIOps Plan-Execute-Replan workflow for structured incident diagnosis.

### Architecture

Unlike the Chat Agent (ReAct, user-driven), the Operation Agent is **agent-driven**: given one alert, it independently generates a multi-step investigation plan, executes each step with real tools, adapts the plan based on results, and produces a structured root-cause report.

```
START → planner → executor → replanner
                     ↑             |
                     |    continue |
                     +─────────────+
                              |
                    replan  → executor
                    respond → END
```

### Demo Scenario

**Alert:** `"payment-service: HTTP 5xx error rate exceeded 10% for 15 minutes"`

Initial plan (5-6 steps): check time → check CPU → check memory → search error logs → get error summary → generate report

**Replanner triggers replan** after seeing CPU/memory are nominal but logs show `"connection pool exhausted"` errors:
- Drops remaining generic steps
- Adds: get deployment events for payment-service

**Final report** (structured markdown):
- Root Cause: PostgreSQL connection pool exhausted
- Evidence: CPU 35% avg / memory 62% avg (infra ruled out); 87% of 5xx = connection_pool_exhausted; deploy at T-5min scaled replicas 3→5 (each opens its own pool)
- Immediate Actions: reduce DB_POOL_SIZE + rolling restart, or deploy PgBouncer

This scenario shows the value of replanning: once the root cause is clear (DB, not infra), the agent drops irrelevant steps and adds targeted ones — without user intervention.

---

### New MCP Server: Logs Server (port 8003)

Replaces Tencent CLS. Inspired by **Splunk** (raw log search) + **Honeybadger** (error aggregation/tracking).

| Tool | Maps to | Description |
|------|---------|-------------|
| `search_logs(service, start_time, end_time, query, limit)` | Splunk | Raw log entries with level, message, trace_id |
| `get_error_summary(service, window_minutes)` | Honeybadger | Error counts by type + rate; surfaces "connection_pool_exhausted" as dominant error |
| `get_service_deployments(service, hours)` | CI/CD system | Recent deploy history; shows replica scale-up 5min before alert |

Mock data is deterministic and keyed by `service_name` so tests are reproducible.

---

### Monitor Server Enhancements (port 8004)

Current tools (`get_cpu_usage`, `get_memory_usage`, `list_top_processes`) read the local machine via psutil. For microservice diagnosis, we need **service-aware** tools:

| New Tool | Description |
|----------|-------------|
| `query_cpu_metrics(service_name, start_time?, end_time?)` | Time-series CPU % for a named service + avg/max/p95 |
| `query_memory_metrics(service_name, start_time?, end_time?)` | Time-series memory for a named service |
| `query_db_connections(service_name)` | Current active DB connections vs max — directly surfaces connection pool exhaustion |

Existing psutil tools remain for real-time local system monitoring.

---

### ✅ Step 18 — Logs MCP Server
**File:** `mcp_servers/logs_server.py`

FastMCP, port 8003, `streamable-http`. Three tools above with deterministic mock data for `payment-service` (DB connection pool exhaustion scenario).

---

### ✅ Step 19 — Monitor Server Enhancement + Config Update
**Files:** `mcp_servers/monitor_server.py` (update), `app/config.py` (update), `app/agent/mcp_client.py` (update)

- Add 3 service-aware tools to monitor_server.py
- Add `mcp_logs_url` to config + `mcp_servers` property
- Ensure MCP client `MultiServerMCPClient` includes both `monitor` and `logs` servers

---

### ✅ Step 20 — Runbooks
**Files:** `docs/high_error_rate.md`, `docs/high_memory.md`, `docs/slow_response.md`

Sections: alert trigger conditions → investigation steps (referencing tool names) → common root causes (DB connection pool, downstream failure, deploy regression, OOM GC pauses) → resolution procedures → verification.

Ingested to Milvus so Planner can retrieve it as context when generating the investigation plan.

---

### ⬜ Step 21 — AIOps State + Planner
**Files:** `app/agent/aiops/state.py`, `app/agent/aiops/planner.py`

`PlanExecuteState`:
```python
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], operator.add]  # append-only
    response: str
```

Planner:
1. Retrieves runbook context via `search_knowledge_base(input)`
2. Lists all tool descriptions (local + MCP)
3. Claude with `with_structured_output(Plan)` → `{"steps": [...]}`
4. Returns `{"plan": steps}`

---

### ⬜ Step 22 — Executor
**File:** `app/agent/aiops/executor.py`

Takes `plan[0]`, binds all tools to Claude, executes (with ToolNode if tool calls are needed), returns:
```python
{"plan": plan[1:], "past_steps": [(task, result_text)]}
```
Errors recorded as `(task, "Error: {msg}")` — replanner can detect and adapt.

---

### ⬜ Step 23 — Replanner
**File:** `app/agent/aiops/replanner.py`

Structured output `Act(action: Literal["continue","replan","respond"], new_steps, rationale)`.

Decision rules:
- Hard limit: `past_steps >= 8` → force respond
- No remaining plan → force respond
- `respond` if evidence is sufficient (>= 3 steps executed with clear signal)
- `replan` only if `past_steps < 5` and plan is clearly wrong; new steps ≤ remaining steps (no expansion)
- `continue` otherwise

`_generate_report()` → structured markdown: **Root Cause** / **Evidence** / **Immediate Actions** / **Long-term Recommendations**

---

### ⬜ Step 24 — AIOps Service + API
**Files:** `app/services/aiops_service.py`, `app/models/aiops.py`, `app/api/aiops.py`, `app/main.py` (update)

`execute(input, session_id)` → async generator of SSE events:
- `{"type": "plan", "steps": [...]}`  — after planner
- `{"type": "step_start", "step": "...", "step_num": N}`  — before executor
- `{"type": "step_done", "result_preview": "...", "remaining": N}`  — after executor
- `{"type": "replanning", "rationale": "..."}`  — when replan triggered
- `{"type": "report", "content": "...markdown..."}`  — final

`POST /aiops` — `AIOpsRequest(session_id, message)` → EventSourceResponse

---

### ⬜ Step 25 — End-to-End Test

```bash
# Start services
docker-compose up -d
python mcp_servers/logs_server.py &
python mcp_servers/monitor_server.py &
uvicorn app.main:app --port 9900 --reload

# Ingest runbook
curl -X POST localhost:9900/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "docs/high_error_rate.md"}'

# Trigger diagnosis
curl -X POST localhost:9900/aiops \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "payment-service: HTTP 5xx error rate exceeded 10% for 15 minutes"}' \
  --no-buffer
```

Expected SSE stream: plan (5-6 steps) → step_done × 3 (time/CPU/memory, all normal) → step_done (error logs: DB connection errors) → `replanning` event → step_done (deployment found) → `report` (root cause + actions)

---

## Phase 3 Files Summary

| File | Status | Step |
|------|--------|------|
| `mcp_servers/logs_server.py` | ✅ Done | 18 |
| `mcp_servers/monitor_server.py` | ✅ Done (update) | 19 |
| `app/config.py` | ✅ Done (update) | 19 |
| `app/agent/mcp_client.py` | ✅ Done (update) | 19 |
| `docs/high_error_rate.md` | ✅ Done | 20 |
| `docs/high_memory.md` | ✅ Done | 20 |
| `docs/slow_response.md` | ✅ Done | 20 |
| `app/agent/aiops/__init__.py` | ⬜ TODO | 21 |
| `app/agent/aiops/state.py` | ⬜ TODO | 21 |
| `app/agent/aiops/planner.py` | ⬜ TODO | 21 |
| `app/agent/aiops/executor.py` | ⬜ TODO | 22 |
| `app/agent/aiops/replanner.py` | ⬜ TODO | 23 |
| `app/services/aiops_service.py` | ⬜ TODO | 24 |
| `app/models/aiops.py` | ⬜ TODO | 24 |
| `app/api/aiops.py` | ⬜ TODO | 24 |
| `app/main.py` | ⬜ TODO (update) | 24 |

---

## Phase 4 — Stack Migration (Mainstream)

Migrate storage layer to mainstream tools for resume visibility. Agent logic, MCP servers, and APIs are untouched.

| Component | From | To |
|-----------|------|----|
| Embeddings | VoyageAI `voyage-3-lite` (512-dim) | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector DB | Milvus + etcd + minio (3 containers) | PostgreSQL + pgvector (1 container) |

### ⬜ Step 26 — Dependencies + Infrastructure

`pyproject.toml`: remove `voyageai`, `pymilvus`, `langchain-milvus`; add `langchain-postgres`, `psycopg2-binary`, `langchain-openai`.

`docker-compose.yml`: replace Milvus stack (3 services) with a single `postgres:16` container with `pgvector` extension enabled.

`.env`: add `OPENAI_API_KEY`, `DATABASE_URL=postgresql://...`; remove Milvus + VoyageAI vars.

---

### ⬜ Step 27 — Vector Store Rewrite

**Files:** `app/config.py` (update), `app/core/milvus_client.py` → `app/core/db_client.py`, `app/services/vector_store.py` (rewrite)

Replace `VectorStoreService` internals with LangChain's `PGVector` + `OpenAIEmbeddings`:

```python
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = PGVector(embeddings=embeddings, connection=DATABASE_URL, collection_name="runbooks")
```

Public interface (`ingest`, `search`) stays identical — nothing above this layer changes.

---

### ⬜ Step 28 — Re-ingest + Verify

Dimension change (512→1536) makes old vectors incompatible; drop and recreate the collection.

```bash
# Start new stack
docker-compose up -d   # now runs postgres+pgvector only

# Re-ingest all runbooks
curl -X POST localhost:9900/ingest -d '{"file_path": "docs/high_cpu.md"}'
curl -X POST localhost:9900/ingest -d '{"file_path": "docs/high_error_rate.md"}'

# Verify retrieval still works
curl -X POST localhost:9900/chat \
  -d '{"session_id": "test", "message": "how do I debug high CPU?"}'
```

---

## Phase 4 Files Summary

| File | Status | Step |
|------|--------|------|
| `pyproject.toml` | ⬜ TODO (update) | 26 |
| `docker-compose.yml` | ⬜ TODO (update) | 26 |
| `.env` | ⬜ TODO (update) | 26 |
| `app/config.py` | ⬜ TODO (update) | 27 |
| `app/core/db_client.py` | ⬜ TODO (replaces milvus_client.py) | 27 |
| `app/services/vector_store.py` | ⬜ TODO (rewrite) | 27 |
