"""
RAG Agent — LangGraph state machine that answers on-call questions using the knowledge base.

Graph structure:
    START → [llm_node] → should_continue → [tools_node] → [llm_node] → END
                       ↘ (no tool calls) → END
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
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
        self.graph = self._build_graph()
        logger.info("RAG agent initialized")

    def _build_graph(self):
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

        return graph.compile(checkpointer=MemorySaver())

    def astream_events(self, *args, **kwargs):
        return self.graph.astream_events(*args, **kwargs)

    def get_state(self, *args, **kwargs):
        return self.graph.get_state(*args, **kwargs)

    def update_state(self, *args, **kwargs):
        return self.graph.update_state(*args, **kwargs)


agent = RAGAgent()
