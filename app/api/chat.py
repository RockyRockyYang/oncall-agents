import json
from typing import AsyncGenerator
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from loguru import logger
from app.services.chat_service import chat_service

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


async def event_stream(message: str, session_id: str) -> AsyncGenerator[dict, None]:
    async for event in chat_service.stream(message, session_id):
        yield {"event": "message", "data": json.dumps(event)}


@router.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    logger.info(
        "Chat request | session_id={} message={!r}", request.session_id, request.message
    )
    return EventSourceResponse(event_stream(request.message, request.session_id))


@router.get("/chat/session/{session_id}")
def get_session(session_id: str):
    messages = chat_service.get_session(session_id)
    return {"session_id": session_id, "messages": messages}


@router.delete("/chat/session/{session_id}")
def delete_session(session_id: str) -> dict:
    chat_service.clear_session(session_id)
    return {"session_id": session_id, "cleared": True}


# testing:
# curl -X POST http://localhost:9900/chat -H "Content-Type: application/json" -d '{"message": "how do I find a runaway process?", "session_id": "test-1"}' --no-buffer
