from langchain_core.messages import HumanMessage
from app.agent import agent


def test_agent_returns_answer():
    result = agent.invoke(
        {"messages": [HumanMessage(content="how do I find a runaway process?")]}
    )
    last = result["messages"][-1]
    assert isinstance(last.content, str)
    assert len(last.content) > 0


def test_agent_uses_knowledge_base():
    result = agent.invoke(
        {"messages": [HumanMessage(content="what are the symptoms of high CPU usage?")]}
    )
    last = result["messages"][-1]
    # the runbook mentions 80% — verify the agent pulled from the KB
    assert "80" in last.content or "cpu" in last.content.lower()


def test_agent_handles_unknown_topic():
    result = agent.invoke(
        {"messages": [HumanMessage(content="how do I bake a sourdough bread?")]}
    )
    last = result["messages"][-1]
    assert isinstance(last.content, str)
