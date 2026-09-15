"""
RAG Agent — LangGraph state machine that answers on-call questions using the knowledge base.

Graph structure:
    START → [llm_node] → should_continue → [tools_node] → [llm_node] → END
                       ↘ (no tool calls) → END
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.mcp_client import get_mcp_tools
from app.config import settings
from app.tools import get_current_time, search_knowledge_base

SYSTEM_PROMPT = """You are an on-call assistant. When asked about incidents or system issues,
  always search the knowledge base first before answering."""

_LOCAL_TOOLS = [search_knowledge_base, get_current_time]


async def llm_node(state: MessagesState) -> MessagesState:
    mcp_tools = await get_mcp_tools()
    llm = ChatAnthropic(
        model_name=settings.rag_model,
        timeout=30,
        stop=None,
    ).bind_tools(_LOCAL_TOOLS + mcp_tools)
    system = SystemMessage(content=SYSTEM_PROMPT)
    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    response = await llm.ainvoke([system] + non_system[-10:])
    return {"messages": [response]}


async def tools_node(state: MessagesState):
    mcp_tools = await get_mcp_tools()
    return await ToolNode(_LOCAL_TOOLS + mcp_tools).ainvoke(state)


class RAGAgent:
    def __init__(self) -> None:
        # 图先不编译——编译需要真正的 checkpointer，而 Postgres 版 checkpointer
        # 要在 FastAPI lifespan 里才能异步建好。在那之前 self.graph 是 None。
        self.graph = None
        logger.info("RAG agent created (graph not yet compiled)")

    def _build_graph(self, checkpointer: BaseCheckpointSaver):
        graph = StateGraph(MessagesState)

        graph.add_node("llm", llm_node)
        graph.add_node("tools", tools_node)
        graph.set_entry_point("llm")

        def should_continue(state: MessagesState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        graph.add_conditional_edges("llm", should_continue)
        graph.add_edge("tools", "llm")

        return graph.compile(checkpointer=checkpointer)

    async def use_checkpointer(self, checkpointer: BaseCheckpointSaver) -> None:
        """在 FastAPI lifespan 里调用一次：拿到真正的 checkpointer 后才编译图。"""
        self.graph = self._build_graph(checkpointer)
        logger.info("RAG agent graph compiled")

    def astream_events(self, *args, **kwargs):
        return self.graph.astream_events(*args, **kwargs)

    async def aget_state(self, *args, **kwargs):
        return await self.graph.aget_state(*args, **kwargs)

    async def aupdate_state(self, *args, **kwargs):
        return await self.graph.aupdate_state(*args, **kwargs)


agent = RAGAgent()
