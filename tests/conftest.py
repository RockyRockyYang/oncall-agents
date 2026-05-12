from typing import Generator
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.vector_store import VectorStoreService
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
