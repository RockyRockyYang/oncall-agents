from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from app.agent import agent

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


async def event_stream(message: str, session_id: str) -> AsyncGenerator[str, None]:
    config = RunnableConfig(configurable={"thread_id": session_id})

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            data = event.get("data", {})
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield f"data: {content}\n\n"
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield f"data: {block['text']}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(request.message, request.session_id),
        media_type="text/event-stream",
    )


# testing:
# curl -X POST http://localhost:9900/chat -H "Content-Type: application/json" -d '{"message": "how do I find a runaway process?", "session_id": "test-1"}' --no-buffer
