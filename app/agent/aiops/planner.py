from textwrap import dedent
from typing import Any, List, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.agent.mcp_client import get_mcp_tools
from app.config import settings
from app.tools.retrieval import search_knowledge_base
from app.tools.time_tool import get_current_time

from .state import PlanExecuteState


def _format_tools(tools: list) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


class Plan(BaseModel):
    steps: List[str] = Field(
        description="Ordered list of investigation steps. Each step must specify the tool to use and parameters to pass."
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
            You are an expert on-call incident investigator.
            Your job is to create a step-by-step investigation plan for the given alert.

            Available tools:
            {tools_description}

            {experience_context}

            Guidelines:
            - Break the investigation into logical, ordered steps
            - Each step must specify: which tool to call, what parameters to pass, and what signal to look for
            - Start by recording the current time
            - Check infrastructure metrics (CPU, memory) before application-level signals
            - If a step reveals a clear root cause, note that subsequent steps can be skipped
        """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> dict[str, Any]:
    logger.info("=== Planner: generating investigation plan ===")
    input_text = state["input"]

    # 1. RAG 检索相关 runbook
    experience_context = ""
    try:
        context = await search_knowledge_base.ainvoke({"query": input_text})
        if context and "No relevant" not in context:
            experience_context = f"Relevant runbook:\n\n{context}"
            logger.info(f"Retrieved runbook context ({len(context)} chars)")
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")

    # 2. 收集所有工具
    local_tools = [get_current_time, search_knowledge_base]
    mcp_tools = await get_mcp_tools()
    all_tools = local_tools + mcp_tools
    tools_description = _format_tools(all_tools)
    logger.info(f"Tools available: {[t.name for t in all_tools]}")

    # 3. 调用 Claude 生成结构化计划
    llm = ChatAnthropic(
        model_name=settings.rag_model,
        temperature=0,
        timeout=30,
        stop=None,
    )

    # LCEL（LangChain Expression Language），本质是管道操作符，把多个组件串成一条流水线。
    # 你传入的 dict
    #     ↓
    # planner_prompt     ← 把变量填进模板，变成消息列表
    #     ↓
    # llm.with_structured_output(Plan)  ← 发给 Claude，把回复解析成 Plan 对象
    #     ↓
    # Plan(steps=[...])
    chain = planner_prompt | llm.with_structured_output(Plan)

    result = await chain.ainvoke(
            {
                "messages": [("user", input_text)],
                "tools_description": tools_description,
                "experience_context": experience_context,
            }
        )
    steps = result.steps if isinstance(result, Plan) else result.get("steps", [])  # type: ignore[union-attr]

    logger.info(f"Plan generated: {len(steps)} steps")
    for i, step in enumerate(steps, 1):
        logger.info(f"  Step {i}: {step}")

    # 4. 返回，只更新 plan 字段
    return {"plan": steps}
