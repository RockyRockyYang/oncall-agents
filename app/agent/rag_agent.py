"""
RAG Agent — LangGraph state machine that answers on-call questions using the knowledge base.

Graph structure:
    START → [llm_node] → should_continue → [tools_node] → [llm_node] → END
                       ↘ (no tool calls) → END

Nodes:
    llm_node   : Calls Claude with the conversation history + system prompt.
                 Claude either returns a final answer or requests a tool call.
    tools_node : Executes the requested tool (search_knowledge_base) and appends
                 the result to the message list, then loops back to llm_node.

Edges:
    set_entry_point        : Graph always starts at llm_node.
    add_conditional_edges  : After llm_node, should_continue() decides next step.
    add_edge               : After tools_node, always return to llm_node.

should_continue:
    Router function. Inspects the last message — if Claude attached tool_calls,
    route to "tools". Otherwise the answer is ready, return END.

Example flow for "how do I find a runaway process?":
    1. llm_node   → Claude decides to call search_knowledge_base("runaway process")
    2. tools_node → search runs, runbook text appended to messages
    3. llm_node   → Claude reads runbook, writes final answer
    4. END        → answer returned to caller
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from app.config import settings
from app.tools import search_knowledge_base, get_current_time
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are an on-call assistant. When asked about incidents or system issues,
  always search the knowledge base first before answering."""

tools = [search_knowledge_base, get_current_time]
llm = ChatAnthropic(
    model=settings.rag_model,
    api_key=settings.anthropic_api_key,
).bind_tools(tools)


def llm_node(state: MessagesState) -> MessagesState:
    system = SystemMessage(content=SYSTEM_PROMPT)
    trimmed = trim_messages(
        state["messages"],
        max_tokens=100,
        token_counter=len,
        strategy="last",
        include_system=True,
    )
    response = llm.invoke([system] + trimmed)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


graph = StateGraph(MessagesState)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")

agent = graph.compile(checkpointer=MemorySaver())


# from langchain_core.messages import HumanMessage
# config = {"configurable": {"thread_id": "session-1"}}
# # Turn 1
# r1 = agent.invoke(
#     {"messages": [HumanMessage(content="what are symptoms of high CPU?")]},
#     config=config,
# )
# print(r1["messages"][-1].content)

# # Turn 2 — references previous turn without repeating context
# r2 = agent.invoke(
#     {"messages": [HumanMessage(content="what should I do next?")]}, config=config
# )
# print(r2["messages"][-1].content)
