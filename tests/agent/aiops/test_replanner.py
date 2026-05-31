from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.runnables import RunnableLambda

from app.agent.aiops.replanner import MAX_REPLAN_STEPS, MAX_STEPS, Act, Response, replanner
from app.agent.aiops.state import PlanExecuteState


def _state(plan: list[str], past_steps: list[tuple] | None = None) -> PlanExecuteState:
    return PlanExecuteState(
        input="payment-service high error rate",
        plan=plan,
        past_steps=past_steps or [],
        response="",
    )


def _mock_llm_with_act(act: Act) -> MagicMock:
    """返回一个 mock LLM，其 with_structured_output 产出指定的 Act 决策。"""
    async def fake_act(inputs):
        return act

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(fake_act)
    return mock_llm


# ---------- plan 为空 / 超过步骤上限 ----------

async def test_replanner_plan_empty_generates_response():
    """没有剩余计划时，跳过 Act 决策，直接生成报告。"""
    state = _state(plan=[], past_steps=[("step 1", "cpu normal")])

    with patch("app.agent.aiops.replanner.ChatAnthropic"), \
         patch("app.agent.aiops.replanner._generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"response": "# Report"}
        result = await replanner(state)

    assert result == {"response": "# Report"}
    mock_gen.assert_called_once()


async def test_replanner_max_steps_forces_response():
    """past_steps >= MAX_STEPS 时，不调用 Act chain，强制生成报告。"""
    past = [("step", "result")] * MAX_STEPS
    state = _state(plan=["step X"], past_steps=past)

    with patch("app.agent.aiops.replanner.ChatAnthropic"), \
         patch("app.agent.aiops.replanner._generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"response": "# Forced Report"}
        result = await replanner(state)

    assert result == {"response": "# Forced Report"}
    mock_gen.assert_called_once()


# ---------- Act 三种决策 ----------

async def test_replanner_continue_returns_empty():
    """'continue' 返回 {}，LangGraph 不修改 state，计划继续执行。"""
    state = _state(plan=["step 2", "step 3"], past_steps=[("step 1", "cpu normal")])

    with patch("app.agent.aiops.replanner.ChatAnthropic",
               return_value=_mock_llm_with_act(Act(action="continue"))):
        result = await replanner(state)

    assert result == {}


async def test_replanner_respond_generates_report():
    """'respond' 调用 _generate_response 并返回报告。"""
    state = _state(plan=["step 2"], past_steps=[("step 1", "db pool exhausted")])

    with patch("app.agent.aiops.replanner.ChatAnthropic",
               return_value=_mock_llm_with_act(Act(action="respond"))), \
         patch("app.agent.aiops.replanner._generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"response": "# Root Cause: DB pool exhausted"}
        result = await replanner(state)

    assert "response" in result
    mock_gen.assert_called_once()


async def test_replanner_replan_updates_plan():
    """'replan' 提供新步骤时，替换剩余计划。"""
    new_steps = ["check downstream service", "check error logs"]
    state = _state(plan=["step 2", "step 3"], past_steps=[("step 1", "cpu normal")])

    with patch("app.agent.aiops.replanner.ChatAnthropic",
               return_value=_mock_llm_with_act(Act(action="replan", new_steps=new_steps))):
        result = await replanner(state)

    assert result == {"plan": new_steps}


# ---------- 安全保护 ----------

async def test_replanner_replan_overridden_after_max_replan_steps():
    """past_steps >= MAX_REPLAN_STEPS 时，replan 被强制覆盖为 respond。"""
    past = [("step", "result")] * MAX_REPLAN_STEPS
    state = _state(plan=["step X"], past_steps=past)

    with patch("app.agent.aiops.replanner.ChatAnthropic",
               return_value=_mock_llm_with_act(Act(action="replan", new_steps=["new step"]))), \
         patch("app.agent.aiops.replanner._generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"response": "# Report"}
        result = await replanner(state)

    assert "response" in result
    mock_gen.assert_called_once()


async def test_replanner_replan_truncates_excess_steps():
    """new_steps 超过剩余计划数量时，截断到 len(plan)。"""
    new_steps = ["a", "b", "c", "d"]  # 4 个新步骤，但只剩 2 步
    state = _state(plan=["step 2", "step 3"], past_steps=[("step 1", "ok")])

    with patch("app.agent.aiops.replanner.ChatAnthropic",
               return_value=_mock_llm_with_act(Act(action="replan", new_steps=new_steps))):
        result = await replanner(state)

    assert result == {"plan": ["a", "b"]}
