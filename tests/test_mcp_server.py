import pytest
from app.agent.mcp_client import get_mcp_tools
import asyncio


@pytest.mark.integration
def test_mcp_tools_load():
    tools = asyncio.run(get_mcp_tools())
    names = [t.name for t in tools]
    assert "get_cpu_usage" in names
    assert "get_memory_usage" in names
    assert "list_top_processes" in names


@pytest.mark.integration
def test_get_cpu_usage():
    tools = asyncio.run(get_mcp_tools())
    tool = next(t for t in tools if t.name == "get_cpu_usage")
    result = asyncio.run(tool.ainvoke({}))
    assert "CPU" in str(result)


@pytest.mark.integration
def test_list_top_processes():
    tools = asyncio.run(get_mcp_tools())
    tool = next(t for t in tools if t.name == "list_top_processes")
    result = asyncio.run(tool.ainvoke({"limit": 3}))
    assert "PID" in str(result)
