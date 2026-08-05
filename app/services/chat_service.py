from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.agent import agent


class ChatService:
    """Chat 业务逻辑层。负责与 RAG Agent 交互，API 层只做 HTTP 包装。"""

    def get_session(self, session_id: str) -> list[dict]:
        """从 MemorySaver 读取会话历史，过滤掉系统消息，只返回用户和助手的消息。"""
        config = RunnableConfig(configurable={"thread_id": session_id})
        state = agent.get_state(config=config)
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in state.values.get("messages", [])
            if isinstance(m, (HumanMessage, AIMessage))
        ]

    def clear_session(self, session_id: str) -> None:
        """清空指定会话的消息历史。MemorySaver 以 thread_id 为键，写入空列表即覆盖。"""
        config = RunnableConfig(configurable={"thread_id": session_id})
        agent.update_state(config=config, values={"messages": []})

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
