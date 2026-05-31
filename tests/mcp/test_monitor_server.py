import asyncio
import json

import pytest

from app.agent.mcp_client import get_mcp_tools
from mcp_servers.monitor_server import (
    query_cpu_metrics,
    query_db_connections,
    query_memory_metrics,
)


# --- integration tests (require both MCP servers running) ---

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


# --- unit tests for service-aware tools (no server needed) ---

def test_query_cpu_metrics_known_service():
    result = json.loads(query_cpu_metrics("payment-service"))
    assert result["service"] == "payment-service"
    assert result["avg"] == 35.0
    assert result["max"] == 38.0
    assert "series" in result


def test_query_memory_metrics_known_service():
    result = json.loads(query_memory_metrics("payment-service"))
    assert result["service"] == "payment-service"
    assert result["avg"] == 62.0
    assert "series" in result


def test_query_db_connections_exhausted():
    result = json.loads(query_db_connections("payment-service"))
    assert result["active"] == result["max"]
    assert result["utilization"] == "100%"
    assert result["waiting"] > 0


def test_query_unknown_service_returns_error():
    for fn in (query_cpu_metrics, query_memory_metrics, query_db_connections):
        result = json.loads(fn("unknown-service"))
        assert "error" in result
