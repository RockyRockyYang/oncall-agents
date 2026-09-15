from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger

from app.agent import agent
from app.agent.aiops.aiops_agent import aiops_agent
from app.api.aiops import router as aiops_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.sessions import router as sessions_router
from app.config import settings
from app.db import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # settings.database_url 是 SQLAlchemy 的写法（postgresql+psycopg://...），
    # 这里 psycopg 要的是普通 libpq URL，得去掉 +psycopg 这个 dialect 后缀
    db_uri = settings.database_url.replace("+psycopg", "")
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        await checkpointer.setup()
        await agent.use_checkpointer(checkpointer)
        await aiops_agent.use_checkpointer(checkpointer)
        logger.info("Postgres checkpointer ready")
        yield
    await engine.dispose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(aiops_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
