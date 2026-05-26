from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.runnables import RunnableLambda

from app.agent.aiops.planner import Plan, planner
from app.agent.aiops.state import PlanExecuteState


def _state(input_text: str = "payment-service high error rate") -> PlanExecuteState:
    return PlanExecuteState(input=input_text, plan=[], past_steps=[], response="")


def _mock_llm(plan: Plan) -> MagicMock:
    """Return a mock ChatAnthropic instance whose chain ainvoke returns the given plan."""
    async def fake_invoke(inputs):
        return plan

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(fake_invoke)
    return mock_llm


async def test_planner_returns_plan_steps():
    expected = Plan(steps=["Step 1: get current time", "Step 2: check CPU metrics"])

    with patch("app.agent.aiops.planner.ChatAnthropic", return_value=_mock_llm(expected)), \
         patch("app.agent.aiops.planner.get_mcp_tools", new_callable=AsyncMock, return_value=[]), \
         patch("app.agent.aiops.planner.search_knowledge_base") as mock_rag:
        mock_rag.ainvoke = AsyncMock(return_value="runbook content")

        result = await planner(_state())

    assert result["plan"] == expected.steps
    assert len(result["plan"]) == 2


async def test_planner_rag_failure_is_handled():
    """Planner still generates a plan even if RAG retrieval throws."""
    expected = Plan(steps=["Step 1: get time"])

    with patch("app.agent.aiops.planner.ChatAnthropic", return_value=_mock_llm(expected)), \
         patch("app.agent.aiops.planner.get_mcp_tools", new_callable=AsyncMock, return_value=[]), \
         patch("app.agent.aiops.planner.search_knowledge_base") as mock_rag:
        mock_rag.ainvoke = AsyncMock(side_effect=Exception("Milvus connection failed"))

        result = await planner(_state())

    assert result["plan"] == expected.steps


async def test_planner_queries_rag_with_input_text():
    """Planner should call RAG using the alert input as the query."""
    expected = Plan(steps=["Step 1: get time"])
    input_text = "slow response time for payment-service"

    with patch("app.agent.aiops.planner.ChatAnthropic", return_value=_mock_llm(expected)), \
         patch("app.agent.aiops.planner.get_mcp_tools", new_callable=AsyncMock, return_value=[]), \
         patch("app.agent.aiops.planner.search_knowledge_base") as mock_rag:
        mock_rag.ainvoke = AsyncMock(return_value="runbook content")

        await planner(_state(input_text))

    mock_rag.ainvoke.assert_called_once_with({"query": input_text})
