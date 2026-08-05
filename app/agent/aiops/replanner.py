from textwrap import dedent
from typing import Any, List, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings

from .state import PlanExecuteState

# 调查步骤总上限：超过后强制生成报告，防止无限循环
# 经验值：正常事故调查 5-7 步就能得出结论，8 是安全上限
MAX_STEPS = 8

# 允许 replan 的最大步数：超过后不再调整计划，只允许结束
# 设为 MAX_STEPS * 0.6，即执行超过 60% 的步骤后就不再改方向
MAX_REPLAN_STEPS = 5

# 在 Replanner prompt 里展示步骤结果的最大字符数
# 防止历史记录过长撑爆 context window；300 chars 足够表达核心信息
RESULT_PREVIEW_CHARS = 300


class Response(BaseModel):
    response: str = Field(
        description="Final incident report for the on-call engineer, in Markdown format"
    )


class Act(BaseModel):
    action: Literal["continue", "replan", "respond"] = Field(
        description=(
            f"- 'respond': Root cause is clear or >= {MAX_REPLAN_STEPS} steps executed — generate final report\n"
            "- 'continue': Remaining plan is still valid — keep executing\n"
            "- 'replan': Findings point to a different root cause — provide updated steps"
        )
    )
    # 仅在 action='replan' 时填充，替换剩余计划步骤
    new_steps: List[str] = Field(
        default_factory=list,
        description="Updated investigation steps (only when action is 'replan')",
    )


# Replanner 决策 prompt：让 Claude 判断下一步该做什么
# 先用 f-string 展开常量，避免 ChatPromptTemplate 将 {MAX_REPLAN_STEPS} 误判为模板变量
_replanner_system = dedent(f"""
    You are an expert on-call incident investigator reviewing investigation progress.

    Decision priority (highest to lowest):
    1. 'respond' — use when root cause is identified OR >= {MAX_REPLAN_STEPS} steps have run
    2. 'continue' — use when remaining steps are still necessary
    3. 'replan' — use only when findings reveal a completely different root cause

    Guidelines:
    - Prefer 'respond' once you have enough signal — don't wait for perfection
    - Never replan if past_steps count >= {MAX_REPLAN_STEPS}
    - If replanning, new_steps count must not exceed remaining plan count
""").strip()

replanner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _replanner_system),
        ("placeholder", "{messages}"),
    ]
)

# 最终报告 prompt：让 Claude 基于调查结果生成结构化事故报告
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                You are an expert on-call incident investigator writing a final incident report.

                Based on the investigation results, write a structured Markdown report covering:
                - **Root Cause**: what caused the incident
                - **Key Evidence**: specific metrics/logs that confirm it
                - **Resolution Steps**: concrete actions to fix it
                - **Verification**: how to confirm the fix worked

                Be specific and data-driven. Only include what the investigation actually found.
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


def _format_past_steps(past_steps: list[tuple]) -> str:
    """将已执行步骤格式化为可读字符串，结果超过 RESULT_PREVIEW_CHARS 字符时截断"""
    return "\n\n".join(
        f"Step: {task}\nResult: {result[:RESULT_PREVIEW_CHARS]}{'...' if len(result) > RESULT_PREVIEW_CHARS else ''}"
        for task, result in past_steps
    )


async def _generate_response(
    state: PlanExecuteState, llm: ChatAnthropic
) -> dict[str, Any]:
    """
    生成最终事故报告。
    被调用的三种情况：
    1. Claude 决定 action='respond'
    2. plan 已全部执行完毕
    3. 执行步骤超过 MAX_STEPS 上限
    """
    logger.info("Generating final incident report...")

    past_steps = state.get("past_steps", [])

    # 将调查历史格式化为 Markdown，供 Claude 撰写报告时参考
    execution_history = "\n\n".join(
        f"### Step: {task}\n**Result:**\n{result}" for task, result in past_steps
    )

    chain = response_prompt | llm.with_structured_output(Response)

    try:
        result = await chain.ainvoke(
            {
                "messages": [
                    ("user", f"Original alert: {state['input']}"),
                    ("user", f"Investigation history:\n{execution_history}"),
                    ("user", "Write the final incident report."),
                ]
            }
        )
        # with_structured_output 返回类型可能是 Response 或 dict，统一处理
        response_text = result.response if isinstance(result, Response) else result.get("response", "")  # type: ignore[union-attr]
        logger.info(f"Report generated ({len(response_text)} chars)")
        return {"response": response_text}

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        # 生成失败时返回纯文本兜底报告
        fallback = (
            f"# Incident Investigation Summary\n\n"
            f"**Alert:** {state['input']}\n\n"
            f"## Findings\n\n{execution_history}"
        )
        return {"response": fallback}


async def replanner(state: PlanExecuteState) -> dict[str, Any]:
    logger.info("=== Replanner: evaluating investigation progress ===")

    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    logger.info(f"Past steps: {len(past_steps)}, Remaining plan: {len(plan)}")

    llm = ChatAnthropic(
        model_name=settings.rag_model,
        temperature=0,
        timeout=30,
        stop=None,
    )

    # 安全保护：步骤数超上限，强制结束，防止无限循环
    if len(past_steps) >= MAX_STEPS:
        logger.warning(
            f"Reached {len(past_steps)} steps (max={MAX_STEPS}), forcing response"
        )
        return await _generate_response(state, llm)

    # 计划已执行完毕，直接生成报告
    if not plan:
        logger.info("No remaining steps, generating report")
        return await _generate_response(state, llm)

    # 仍有剩余步骤：让 Claude 决策下一步行动
    chain = replanner_prompt | llm.with_structured_output(Act)

    try:
        act = await chain.ainvoke(
            {
                "messages": [
                    ("user", f"Original alert: {state['input']}"),
                    ("user", f"Completed steps:\n{_format_past_steps(past_steps)}"),
                    ("user", "Remaining plan:\n" + "\n".join(f"- {s}" for s in plan)),
                    (
                        "user",
                        f"Steps executed so far: {len(past_steps)}. Prefer 'respond' if root cause is clear.",
                    ),
                ]
            }
        )

        # 统一处理 Act 对象或 dict 两种返回格式
        action = act.action if isinstance(act, Act) else act.get("action", "continue")  # type: ignore[union-attr]
        new_steps = act.new_steps if isinstance(act, Act) else act.get("new_steps", [])  # type: ignore[union-attr]
        logger.info(f"Replanner decision: {action}")

        if action == "respond":
            return await _generate_response(state, llm)
        elif action == "replan":
            # 安全保护：已执行 MAX_REPLAN_STEPS 步以上，禁止 replan，强制结束
            if len(past_steps) >= MAX_REPLAN_STEPS:
                logger.warning("Too many steps, overriding replan → respond")
                return await _generate_response(state, llm)
            # 安全保护：新步骤数不能超过剩余步骤数，截断多余部分
            if len(new_steps) > len(plan):
                new_steps = new_steps[: len(plan)]
                logger.warning(f"Truncated new_steps to {len(new_steps)}")
            if new_steps:
                logger.info(f"Replanning with {len(new_steps)} new steps")
                return {"plan": new_steps}
            else:
                # replan 但没给新步骤，降级为 continue
                logger.warning(
                    "replan chosen but no new_steps provided, falling back to continue"
                )
                return {}
        else:  # continue
            # 返回空 dict = 不修改任何 state 字段，LangGraph 继续执行下一个 Executor
            logger.info("Continuing with existing plan")
            return {}

    except Exception as e:
        # 决策失败时默认 continue，避免整个调查中断
        logger.error(f"Replanner failed: {e}, continuing with existing plan")
        return {}
