from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.mcp_client import get_mcp_tools
from app.config import settings
from app.tools.retrieval import search_knowledge_base
from app.tools.time_tool import get_current_time

from .state import PlanExecuteState


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    logger.info("=== Executor: executing next plan step ===")

    plan = state["plan"]
    if not plan:
        logger.info("Plan is empty, skipping execution")
        return {}

    task = plan[0]
    logger.info(f"Executing step: {task}")

    try:
        # 1. 收集所有工具（同 Planner，但这次是真正绑定用来执行）
        local_tools = [get_current_time, search_knowledge_base]
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        logger.info(f"Tools available: local={len(local_tools)}, mcp={len(mcp_tools)}")

        # 2. 创建 LLM 并绑定工具（让 Claude 知道可以调哪些工具）
        llm = ChatAnthropic(
            model_name=settings.rag_model,
            temperature=0,
            timeout=30,
            stop=None,
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 3. ToolNode：负责实际执行 Claude 生成的 tool_call
        tool_node = ToolNode(all_tools)

        # 4. 构建消息，告诉 Claude 当前要执行的任务
        messages = [
            SystemMessage(
                content=(
                    "You are a powerful assistant responsible for executing a specific investigation step.\n"
                    "Use the specified tool to complete the task. "
                    "Do not fabricate data — only return what the tool actually returns.\n"
                    "After calling the tool, summarize the result clearly and concisely."
                )
            ),
            HumanMessage(content=f"Execute the following step: {task}"),
        ]

        # 5. 第一步：LLM 决定调用哪个工具（生成 tool_call）
        llm_response = await llm_with_tools.ainvoke(messages)

        # 6. 如果有 tool_call，执行工具并让 LLM 总结结果
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"Tool calls detected: {len(llm_response.tool_calls)}")

            # ToolNode 需要消息列表（包含 LLM 的 tool_call 消息）
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 把工具结果加回去，让 LLM 生成最终分析结论
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content
        else:
            # LLM 没有调用工具，直接用它的回复（少见，但处理掉）
            logger.info("No tool calls, using LLM response directly")
            result = llm_response.content

        logger.info(f"Task completed. Result preview: {str(result)[:200]}")
        return {
            "plan": plan[1:],
            "past_steps": [(task, result)],
        }
    except Exception as e:
        logger.error(f"Executor failed: {e}", exc_info=True)
        return {
            "plan": plan[1:],  # remove the failed step and continue
            "past_steps": [(task, f"Error: {str(e)}")],
        }
