"""
MCP Client — loads tools from external MCP servers for use in the agent.

Why async?
    MCP connects over HTTP. HTTP is I/O — it blocks while waiting for the network.
    We use async so FastAPI's event loop can handle other requests while waiting.
    This module never initializes at import time; get_mcp_tools() is called
    during FastAPI lifespan startup, when the event loop is already running.

RetryInterceptor:
    Wraps every MCP tool call with exponential backoff (1s, 2s, 4s).
    If the monitor server is temporarily down, the agent receives a ToolMessage
    with the error instead of crashing.
"""

import asyncio
from collections.abc import Awaitable, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    ToolCallInterceptor,
)
from mcp.types import CallToolResult
from langgraph.types import Command


from app.config import settings


class RetryInterceptor(ToolCallInterceptor):
    """Retries failed MCP tool calls with exponential backoff."""

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[
            [MCPToolCallRequest], Awaitable[CallToolResult | ToolMessage | Command]
        ],
    ) -> CallToolResult | ToolMessage | Command:
        for attempt in range(3):
            try:
                return await handler(request)
            except Exception as e:
                last_exc = e
                await asyncio.sleep(2**attempt)
        return ToolMessage(
            content=f"Tool {request.name} failed after 3 attempts: {last_exc}",
            tool_call_id="retry-error",
        )


async def get_mcp_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        connections={
            "monitor": {"transport": "streamable_http", "url": settings.mcp_monitor_url}
        },
        tool_interceptors=[RetryInterceptor()],
    )
    return await client.get_tools()
