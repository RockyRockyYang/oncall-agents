from fastapi import FastAPI

from app.api.aiops import router as aiops_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(aiops_router, prefix="/api")
