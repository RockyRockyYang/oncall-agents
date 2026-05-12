from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.agent.rag_agent import agent
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()  # 启动时初始化 agent
    yield  # 开始接受请求
    # yield 之后是关闭时运行（清理资源)


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(chat_router)
app.include_router(ingest_router)
