from fastapi import APIRouter

from app.agent import agent
from app.db.crud import delete_session, list_sessions

router = APIRouter()


@router.get("/sessions")
async def get_sessions():
    sessions = await list_sessions()
    return {"sessions": sessions}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str) -> dict:
    await delete_session(session_id)
    await agent.adelete_thread(session_id)
    return {"session_id": session_id, "deleted": True}
