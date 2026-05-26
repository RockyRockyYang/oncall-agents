from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.aiops.executor import executor
from app.agent.aiops.state import PlanExecuteState


def _state(plan: list[str]) -> PlanExecuteState:
    return PlanExecuteState(
        input="payment-service alert",
        plan=plan,
        past_steps=[],
        response="",
    )


async def test_executor_empty_plan_returns_empty():
    result = await executor(_state(plan=[]))
    assert result == {}


async def test_executor_with_tool_call():
    """LLM calls a tool → ToolNode executes it → LLM summarizes → result in past_steps."""
    tool_call = {
        "name": "query_cpu_metrics",
        "args": {"service_name": "payment-service"},
        "id": "call_1",
        "type": "tool_call",
    }
    first_response = AIMessage(content="", tool_calls=[tool_call])
    tool_result = ToolMessage(content='{"avg": 35.0}', tool_call_id="call_1")
    final_response = AIMessage(content="CPU is normal at 35% average.")

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm_with_tools.ainvoke = AsyncMock(side_effect=[first_response, final_response])

    mock_tool_node = MagicMock()
    mock_tool_node.ainvoke = AsyncMock(return_value={"messages": [tool_result]})

    state = _state(plan=["Step 1: query CPU", "Step 2: check memory"])

    with patch("app.agent.aiops.executor.ChatAnthropic", return_value=mock_llm), \
         patch("app.agent.aiops.executor.ToolNode", return_value=mock_tool_node), \
         patch("app.agent.aiops.executor.get_mcp_tools", new_callable=AsyncMock, return_value=[]):
        result = await executor(state)

    # executed step is removed from plan
    assert result["plan"] == ["Step 2: check memory"]

    # result is stored in past_steps
    assert len(result["past_steps"]) == 1
    task, outcome = result["past_steps"][0]
    assert task == "Step 1: query CPU"
    assert "CPU" in outcome


async def test_executor_no_tool_call():
    """When LLM responds without calling tools, its content is stored directly."""
    direct_response = AIMessage(content="Current time is 2025-01-01 10:00 UTC.", tool_calls=[])

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm_with_tools.ainvoke = AsyncMock(return_value=direct_response)

    state = _state(plan=["Step 1: record current time"])

    with patch("app.agent.aiops.executor.ChatAnthropic", return_value=mock_llm), \
         patch("app.agent.aiops.executor.ToolNode", return_value=MagicMock()), \
         patch("app.agent.aiops.executor.get_mcp_tools", new_callable=AsyncMock, return_value=[]):
        result = await executor(state)

    assert result["plan"] == []
    _, outcome = result["past_steps"][0]
    assert "time" in outcome.lower()


async def test_executor_error_is_recorded_and_step_removed():
    """If execution throws, error is stored in past_steps and step is still removed from plan."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.side_effect = Exception("LLM connection refused")

    state = _state(plan=["Step 1: check CPU", "Step 2: check memory"])

    with patch("app.agent.aiops.executor.ChatAnthropic", return_value=mock_llm), \
         patch("app.agent.aiops.executor.get_mcp_tools", new_callable=AsyncMock, return_value=[]):
        result = await executor(state)

    assert result["plan"] == ["Step 2: check memory"]
    task, outcome = result["past_steps"][0]
    assert task == "Step 1: check CPU"
    assert "Error" in outcome
