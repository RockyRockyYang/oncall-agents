from typing import AsyncGenerator, Generator
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.vector_store import VectorStoreService
from app.db.crud import delete_session
import app.api.ingest as ingest_module


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def use_test_collection() -> Generator[None, None, None]:
    test_svc = VectorStoreService(collection="oncall_kb_test")
    ingest_module._svc = test_svc
    yield
    test_svc.drop_collection()


@pytest.fixture
async def temp_session_ids() -> AsyncGenerator[list[str], None]:
    """测试里创建的 session，用完自动删掉——sessions 表连着真实开发库，不清理的话每跑一次
    测试就会在前端历史列表里多出几条垃圾数据。往这个列表里 append 创建的 id 即可。"""
    created: list[str] = []
    yield created
    for session_id in created:
        await delete_session(session_id)
