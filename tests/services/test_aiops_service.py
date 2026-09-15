from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableConfig

from app.agent.aiops.aiops_agent import NODE_EXECUTOR, NODE_PLANNER, NODE_REPLANNER, aiops_agent
from app.services.aiops_service import AIOpsService, aiops_service


# ── 辅助：构造受控的 async generator ──────────────────────────────

async def _fake_stream(events: list):
    for event in events:
        yield event


# ── AIOpsService.execute() 集成流程 ───────────────────────────────

async def test_execute_yields_plan_event():
    """Planner 节点完成后，service 应 yield type=plan 事件。"""
    stream_events = [{NODE_PLANNER: {"plan": ["step1", "step2"]}}]
    mock_config = RunnableConfig(configurable={"thread_id": "test"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "aget_state", return_value=MagicMock(values={"response": ""})):

        events = [e async for e in aiops_service.execute("test alert", "test")]

    plan_events = [e for e in events if e["type"] == "plan"]
    assert len(plan_events) == 1
    assert plan_events[0]["plan"] == ["step1", "step2"]


async def test_execute_yields_step_complete_event():
    """Executor 节点完成后，service 应 yield type=step_complete 事件。"""
    stream_events = [
        {NODE_EXECUTOR: {"past_steps": [("step1", "cpu normal")], "plan": ["step2"]}},
    ]
    mock_config = RunnableConfig(configurable={"thread_id": "test"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "aget_state", return_value=MagicMock(values={"response": ""})):

        events = [e async for e in aiops_service.execute("test alert", "test")]

    step_events = [e for e in events if e["type"] == "step_complete"]
    assert len(step_events) == 1
    assert step_events[0]["current_step"] == "step1"
    assert step_events[0]["remaining_steps"] == 1


async def test_execute_yields_report_and_complete():
    """Replanner 生成报告后，service 应 yield type=report，最后 yield type=complete。"""
    stream_events = [
        {NODE_REPLANNER: {"response": "# Root Cause: DB pool exhausted"}},
    ]
    mock_config = RunnableConfig(configurable={"thread_id": "test"})

    with patch.object(aiops_agent, "astream", return_value=(_fake_stream(stream_events), mock_config)), \
         patch.object(aiops_agent, "aget_state", return_value=MagicMock(values={"response": "# Root Cause: DB pool exhausted"})):

        events = [e async for e in aiops_service.execute("test alert", "test")]

    assert any(e["type"] == "report" for e in events)
    assert events[-1]["type"] == "complete"
    assert "DB pool" in events[-1]["response"]


async def test_execute_yields_error_on_exception():
    """astream 抛异常时，service 应 yield type=error 事件。"""
    with patch.object(aiops_agent, "astream", side_effect=Exception("MCP connection refused")):
        events = [e async for e in aiops_service.execute("test alert", "test")]

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "MCP connection refused" in events[0]["message"]


# ── 事件格式化单元测试 ─────────────────────────────────────────────

def test_format_planner_event():
    svc = AIOpsService()
    result = svc._format_planner_event({"plan": ["step1", "step2", "step3"]})

    assert result["type"] == "plan"
    assert result["plan"] == ["step1", "step2", "step3"]
    assert "3" in result["message"]


def test_format_executor_event_with_steps():
    svc = AIOpsService()
    result = svc._format_executor_event({
        "past_steps": [("step1", "ok"), ("step2", "ok")],
        "plan": ["step3"],
    }, 2)

    assert result["type"] == "step_complete"
    assert result["current_step"] == "step2"  # 最后一步
    assert result["remaining_steps"] == 1


def test_format_executor_event_empty_past_steps():
    svc = AIOpsService()
    result = svc._format_executor_event({"past_steps": [], "plan": ["step1"]}, 1)

    assert result["type"] == "status"


def test_format_replanner_event_with_response():
    svc = AIOpsService()
    result = svc._format_replanner_event({"response": "# Final Report", "plan": []})

    assert result["type"] == "report"
    assert result["report"] == "# Final Report"


def test_format_replanner_event_continue():
    svc = AIOpsService()
    result = svc._format_replanner_event({"response": "", "plan": ["step2", "step3"]})

    assert result["type"] == "status"
    assert result["remaining_steps"] == 2
