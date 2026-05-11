import json
from typing import AsyncGenerator
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from app.agent import agent

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


async def event_stream(message: str, session_id: str) -> AsyncGenerator[dict, None]:
    config = RunnableConfig(configurable={"thread_id": session_id})
    try:
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_tool_start":
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "type": "tool_call",
                            "data": {"tool": event.get("name"), "status": "start"},
                        }
                    ),
                }

            elif event["event"] == "on_chat_model_stream":
                data = event.get("data", {})
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if isinstance(content, str) and content:
                        yield {
                            "event": "message",
                            "data": json.dumps({"type": "content", "data": content}),
                        }
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield {
                                    "event": "message",
                                    "data": json.dumps(
                                        {"type": "content", "data": block["text"]}
                                    ),
                                }

        logger.info("Stream complete | session_id={}", session_id)
        yield {"event": "message", "data": json.dumps({"type": "done"})}

    except Exception as e:
        logger.error("Stream error | session_id={} error={}", session_id, e)
        yield {
            "event": "message",
            "data": json.dumps({"type": "error", "data": str(e)}),
        }


@router.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    logger.info(
        "Chat request | session_id={} message={!r}", request.session_id, request.message
    )
    return EventSourceResponse(event_stream(request.message, request.session_id))


@router.post("/chat/session/{session_id}")
def get_session(session_id: str):
    config = RunnableConfig(configurable={"thread_id": session_id})
    state = agent.get_state(config=config)
    messages = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in state.values.get("messages", [])
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return {"session_id": session_id, "messages": messages}


@router.delete("/chat/session/{session_id}")
def delete_session(session_id: str) -> dict:
    config = RunnableConfig(configurable={"thread_id": session_id})
    agent.update_state(config=config, values={"messages": []})
    return {"session_id": session_id, "cleared": True}


# testing:
# curl -X POST http://localhost:9900/chat -H "Content-Type: application/json" -d '{"message": "how do I find a runaway process?", "session_id": "test-1"}' --no-buffer
