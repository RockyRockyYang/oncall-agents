import asyncio
from unittest.mock import MagicMock, patch

from app.services.chat_service import ChatService

# sessions.id 现在是 Postgres 的 UUID 类型，插入必须是合法 UUID 格式
SESSION_ID = "11111111-1111-1111-1111-111111111111"


async def _collect(gen):
    results = []
    async for item in gen:
        results.append(item)
    return results


def make_mock_agent(events: list):
    """Return a mock agent whose astream_events yields the given events."""
    async def astream_events(*args, **kwargs):
        for event in events:
            yield event

    mock = MagicMock()
    mock.astream_events = astream_events
    return mock


def test_tool_call_event():
    mock_agent = make_mock_agent([
        {"event": "on_tool_start", "name": "search_knowledge_base"},
    ])
    with patch("app.services.chat_service.agent", mock_agent):
        events = asyncio.run(_collect(ChatService().stream("test", SESSION_ID)))

    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["tool"] == "search_knowledge_base"
    assert tool_events[0]["data"]["status"] == "start"


def test_content_event_string():
    chunk = MagicMock()
    chunk.content = "hello world"
    mock_agent = make_mock_agent([
        {"event": "on_chat_model_stream", "data": {"chunk": chunk}},
    ])
    with patch("app.services.chat_service.agent", mock_agent):
        events = asyncio.run(_collect(ChatService().stream("test", SESSION_ID)))

    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) == 1
    assert content_events[0]["data"] == "hello world"


def test_content_event_list():
    chunk = MagicMock()
    chunk.content = [{"type": "text", "text": "block text"}, {"type": "other"}]
    mock_agent = make_mock_agent([
        {"event": "on_chat_model_stream", "data": {"chunk": chunk}},
    ])
    with patch("app.services.chat_service.agent", mock_agent):
        events = asyncio.run(_collect(ChatService().stream("test", SESSION_ID)))

    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) == 1
    assert content_events[0]["data"] == "block text"


def test_done_event_always_last():
    mock_agent = make_mock_agent([])
    with patch("app.services.chat_service.agent", mock_agent):
        events = asyncio.run(_collect(ChatService().stream("test", SESSION_ID)))

    assert events[-1] == {"type": "done"}


def test_error_event_on_exception():
    async def boom(*args, **kwargs):
        raise RuntimeError("connection failed")
        yield  # make it an async generator

    mock = MagicMock()
    mock.astream_events = boom
    with patch("app.services.chat_service.agent", mock):
        events = asyncio.run(_collect(ChatService().stream("test", SESSION_ID)))

    assert events[0]["type"] == "error"
    assert "connection failed" in events[0]["data"]
