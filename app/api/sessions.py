from fastapi import APIRouter

from app.db.crud import list_sessions

router = APIRouter()


@router.get("/sessions")
async def get_sessions():
    sessions = await list_sessions()
    return {"sessions": sessions}
