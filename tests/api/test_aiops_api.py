import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig

from app.agent.aiops.aiops_agent import (
    NODE_EXECUTOR,
    NODE_PLANNER,
    NODE_REPLANNER,
    aiops_agent,
)


# ── 辅助 ──────────────────────────────────────────────────────────

async def _fake_stream(events: list):
    for event in events:
        yield event


def _parse_sse(raw: bytes) -> list[dict]:
    """把 SSE 原始字节解析成 event dict 列表。"""
    result = []
    for line in raw.decode().splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                result.append(json.loads(data))
    return result


# ── 完整流程测试 ───────────────────────────────────────────────────

def test_investigate_full_flow(client: TestClient):
    """
    Planner → Executor → Replanner 完整链路：
    HTTP 层应返回正确 SSE 格式，事件顺序为 plan → step_complete → report → complete。
    """
    stream_events = [
        {NODE_PLANNER: {"plan": ["check metrics", "check logs"]}},
        {NODE_EXECUTOR: {"past_steps": [("check metrics", "P99=2.1s")], "plan": ["check logs"]}},
        {NODE_REPLANNER: {"response": "# Root Cause: DB slow query", "plan": []}},
    ]
    mock_config = RunnableConfig(configurable={"thread_id": "e2e-test"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "get_state", return_value=MagicMock(values={"response": "# Root Cause: DB slow query"})):

        with client.stream("POST", "/api/aiops/investigate",
                           json={"alert": "payment-service P99 > 3s", "session_id": "e2e-test"}) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            raw = b"".join(response.iter_bytes())

    events = _parse_sse(raw)
    types = [e["type"] for e in events]

    assert types[0] == "plan"
    assert "step_complete" in types
    assert "report" in types
    assert types[-1] == "complete"


def test_investigate_plan_content(client: TestClient):
    """plan 事件里应包含正确的步骤列表。"""
    plan_steps = ["check metrics", "check logs", "check DB"]
    stream_events = [{NODE_PLANNER: {"plan": plan_steps}}]
    mock_config = RunnableConfig(configurable={"thread_id": "t1"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "get_state", return_value=MagicMock(values={"response": ""})):

        with client.stream("POST", "/api/aiops/investigate",
                           json={"alert": "high error rate", "session_id": "t1"}) as response:
            raw = b"".join(response.iter_bytes())

    events = _parse_sse(raw)
    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["plan"] == plan_steps


def test_investigate_error_event(client: TestClient):
    """agent 抛异常时，SSE 流中应有 type=error 事件，且 HTTP 状态码仍为 200。"""
    with patch.object(aiops_agent, "astream", side_effect=Exception("MCP connection refused")):
        with client.stream("POST", "/api/aiops/investigate",
                           json={"alert": "test alert", "session_id": "err-test"}) as response:
            # SSE 协议下错误也是 200，错误信息在 data 里
            assert response.status_code == 200
            raw = b"".join(response.iter_bytes())

    events = _parse_sse(raw)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "MCP connection refused" in events[0]["message"]


def test_investigate_stream_ends_after_complete(client: TestClient):
    """complete 事件之后不应有其他事件（SSE 连接正常关闭）。"""
    stream_events = [
        {NODE_REPLANNER: {"response": "# Final Report", "plan": []}},
    ]
    mock_config = RunnableConfig(configurable={"thread_id": "t2"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "get_state", return_value=MagicMock(values={"response": "# Final Report"})):

        with client.stream("POST", "/api/aiops/investigate",
                           json={"alert": "disk full", "session_id": "t2"}) as response:
            raw = b"".join(response.iter_bytes())

    events = _parse_sse(raw)
    complete_idx = next(i for i, e in enumerate(events) if e["type"] == "complete")
    assert complete_idx == len(events) - 1  # complete 一定是最后一条


# ── 请求验证 ──────────────────────────────────────────────────────

def test_investigate_missing_alert_returns_422(client: TestClient):
    """缺少必填字段 alert 时，FastAPI 应返回 422 Unprocessable Entity。"""
    response = client.post("/api/aiops/investigate", json={"session_id": "no-alert"})
    assert response.status_code == 422


def test_investigate_default_session_id(client: TestClient):
    """session_id 有默认值，不传时也应正常工作。"""
    stream_events = [{NODE_PLANNER: {"plan": ["step1"]}}]
    mock_config = RunnableConfig(configurable={"thread_id": "default"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "get_state", return_value=MagicMock(values={"response": ""})):

        with client.stream("POST", "/api/aiops/investigate",
                           json={"alert": "some alert"}) as response:  # 没传 session_id
            assert response.status_code == 200
