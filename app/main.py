from fastapi import FastAPI
from app.config import settings
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router


app = FastAPI(title=settings.app_name)
app.include_router(chat_router)
app.include_router(ingest_router)
