import asyncio
import json
import pytest
from fastmcp import Client
from mcp.types import TextContent

MCP_URL = "http://localhost:8003/mcp"


@pytest.mark.integration
def test_search_logs_returns_entries():
    async def run():
        async with Client(MCP_URL) as client:
            return await client.call_tool("search_logs", {"service": "payment-service"})

    result = asyncio.run(run())

    content = result.content[0]
    assert isinstance(content, TextContent)

    data = json.loads(content.text)
    assert len(data) > 0
    assert "message" in data[0]
    assert "level" in data[0]


@pytest.mark.integration
def test_get_error_summary_dominant_error():
    async def run():
        async with Client(MCP_URL) as client:
            return await client.call_tool(
                "get_error_summary", {"service": "payment-service"}
            )

    result = asyncio.run(run())
    content = result.content[0]
    assert isinstance(content, TextContent)

    data = json.loads(content.text)
    assert data["total_errors"] > 0
    top_error = data["top_errors"][0][0]
    assert "connection pool exhausted" in top_error


@pytest.mark.integration
def test_get_service_deployments():
    async def run():
        async with Client(MCP_URL) as client:
            return await client.call_tool(
                "get_service_deployments", {"service": "payment-service"}
            )

    result = asyncio.run(run())
    content = result.content[0]
    assert isinstance(content, TextContent)

    data = json.loads(content.text)
    assert len(data) > 0
    assert "replicas" in data[0]["change"]
