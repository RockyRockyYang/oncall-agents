from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.agent import agent


class ChatService:
    """Chat 业务逻辑层。负责与 RAG Agent 交互，API 层只做 HTTP 包装。"""

    async def get_session(self, session_id: str) -> list[dict]:
        """从 checkpointer 读取会话历史，过滤掉系统消息，只返回用户和助手的消息。"""
        config = RunnableConfig(configurable={"thread_id": session_id})
        state = await agent.aget_state(config=config)
        result = []
        for m in state.values.get("messages", []):
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            content = self._extract_text(m.content)
            if isinstance(m, AIMessage) and not content:
                # 纯 tool_use、没有文本的中间轮次（比如"我先查一下工具"这类无文字的调用），
                # 前端没有内容可展示，跳过，避免出现一个内容永远是空的气泡。
                continue
            result.append(
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": content}
            )
        return result

    @staticmethod
    def _extract_text(content: str | list) -> str:
        """把 AIMessage.content 统一成纯文本。

        Claude 在带工具调用的轮次里，content 不是字符串，而是
        [{"type": "text", ...}, {"type": "tool_use", ...}, ...] 这样的 block 列表，
        跟 stream() 里处理 chunk.content 的逻辑保持一致：只取 text block。
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    async def clear_session(self, session_id: str) -> None:
        """清空指定会话的消息历史。checkpointer 以 thread_id 为键，写入空列表即覆盖。"""
        config = RunnableConfig(configurable={"thread_id": session_id})
        await agent.aupdate_state(config=config, values={"messages": []})

    async def stream(self, message: str, session_id: str) -> AsyncGenerator[dict, None]:
        """
        运行 RAG Agent 并以 async generator 形式逐步 yield 事件 dict。

        LangGraph 的 astream_events 会持续发出各种事件，这里只关心两种：
        - on_tool_start：Agent 调用工具时触发，告知前端正在查询哪个工具
        - on_chat_model_stream：Claude 逐 token 输出时触发，内容有两种格式：
            - str：普通文本，直接 yield
            - list[dict]：thinking block 格式（type=text 才是正文）

        最终不管正常结束还是报错，都会 yield 一个终止事件让前端知道流结束了。
        """
        config = RunnableConfig(configurable={"thread_id": session_id})
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_tool_start":
                    yield {
                        "type": "tool_call",
                        "data": {"tool": event.get("name"), "status": "start"},
                    }

                elif event["event"] == "on_chat_model_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if isinstance(content, str) and content:
                            yield {"type": "content", "data": content}
                        elif isinstance(content, list):
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    yield {"type": "content", "data": block["text"]}

            logger.info("Stream complete | session_id={}", session_id)
            yield {"type": "done"}

        except Exception as e:
            logger.error("Stream error | session_id={} error={}", session_id, e)
            yield {"type": "error", "data": str(e)}


chat_service = ChatService()
