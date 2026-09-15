from typing import Any, AsyncGenerator
from loguru import logger
from app.agent.aiops.aiops_agent import (
    NODE_EXECUTOR,
    NODE_PLANNER,
    NODE_REPLANNER,
    aiops_agent,
)


class AIOpsService:
    """AIOps 业务逻辑层。负责与 AIOps Agent 交互"""

    async def execute(
        self, alert: str, session_id: str = "default"
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        调用 agent，将原始 LangGraph 事件格式化为 SSE 友好的 dict。
        流式推送中间事件是为了让前端能显示进度: 计划生成、步骤完成、状态更新、最终报告等。
        最后无论成功还是失败都会有一个终止事件（complete 或 error）, 让前端知道流程结束了。
        """
        logger.info(f"[{session_id}] Starting investigation: {alert}")

        try:
            stream, config = aiops_agent.astream(alert, session_id)
            step_count = 0

            async for event in stream:
                for node_name, node_output in event.items():
                    # LangGraph 1.x emits None for nodes that return {} (no state change)
                    if node_output is None:
                        logger.debug(f"Node '{node_name}' returned no update, skipping")
                        continue
                    logger.info(f"Node '{node_name}' completed")
                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)
                    elif node_name == NODE_EXECUTOR:
                        step_count += 1
                        yield self._format_executor_event(node_output, step_count)
                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            # 从 checkpointer 拿最终 response
            final_state = await aiops_agent.aget_state(config)
            final_response = (
                final_state.values.get("response", "") if final_state else ""
            )

            yield {"type": "complete", "stage": "complete", "response": final_response}
            logger.info(f"[{session_id}] Investigation complete")

        except Exception as e:
            # 注意：不能用 f-string 提前把 {e} 拼进消息里再传给 logger.error —— 如果异常文本本身
            # 带花括号（比如这次踩到的 Anthropic 错误信息里有 {'type': 'error', ...}），loguru
            # 会把它当成自己的格式化占位符去解析，直接 KeyError 崩掉，把真正的报错信息吞掉。
            # 用 loguru 自己的 {} 占位符 + 位置参数传参，异常文本原样插入，不会被二次解析。
            logger.exception("[{}] Investigation failed: {}", session_id, e)
            yield {"type": "error", "stage": "error", "message": str(e)}

    # ───────────────── 事件格式化 ─────────────────────────────────

    def _format_planner_event(self, node_output: dict) -> dict:
        plan = node_output.get("plan", [])
        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"Investigation plan created with {len(plan)} steps",
            "plan": plan,
        }

    def _format_executor_event(self, node_output: dict, step_count: int) -> dict:
        past_steps = node_output.get("past_steps", [])
        plan = node_output.get("plan", [])
        if past_steps:
            last_task, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"Step {step_count} completed",
                "current_step": last_task,
                "remaining_steps": len(plan),
            }
        return {"type": "status", "stage": "executor", "message": "Executing step..."}

    def _format_replanner_event(self, node_output: dict) -> dict:
        response = node_output.get("response", "")
        plan = node_output.get("plan", [])
        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "Final incident report generated",
                "report": response,
            }
        return {
            "type": "status",
            "stage": "replanner",
            "message": (
                f"Continuing, {len(plan)} steps remaining"
                if plan
                else "Preparing final report"
            ),
            "remaining_steps": len(plan),
        }


aiops_service = AIOpsService()
