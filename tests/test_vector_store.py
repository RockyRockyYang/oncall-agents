import pytest
from app.services.vector_store import VectorStoreService


@pytest.fixture(scope="module")
def svc():
    return VectorStoreService()


@pytest.fixture(scope="module")
def chunks():
    with open("docs/high_cpu.md") as f:
        text = f.read()
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def test_ingest(svc, chunks):
    svc.ingest(chunks, source="high_cpu.md")


def test_search_returns_results(svc):
    results = svc.search("how do I find which process is using too much CPU?")
    assert len(results) > 0


def test_search_result_is_relevant(svc):
    results = svc.search("how do I find which process is using too much CPU?")
    combined = " ".join(results).lower()
    assert any(kw in combined for kw in ["cpu", "process", "top", "htop"])
