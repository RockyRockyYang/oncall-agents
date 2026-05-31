import pytest
import pathlib
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_ingest_document(client: TestClient) -> None:
    text = pathlib.Path("docs/high_cpu.md").read_text()
    response = client.post(
        "/ingest",
        json={"source": "high_cpu.md", "text": text},
        timeout=60.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "high_cpu.md"
    assert data["chunks_inserted"] == 5


@pytest.mark.integration
def test_ingest_empty_text(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        json={"source": "empty.md", "text": ""},
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_end_to_end(client: TestClient) -> None:
    # ingest the runbook first
    text = pathlib.Path("docs/high_cpu.md").read_text()
    client.post("/ingest", json={"source": "high_cpu.md", "text": text}, timeout=60.0)

    # ask a question and verify the answer references runbook content
    with client.stream(
        "POST",
        "/chat",
        json={"message": "what are the symptoms of high CPU?", "session_id": "e2e-1"},
        timeout=60.0,
    ) as response:
        assert response.status_code == 200
        full_response = " ".join(response.iter_lines())

    assert "data:" in full_response
    assert any(kw in full_response.lower() for kw in ["cpu", "80", "symptom"])
    assert "data: [DONE]" in full_response
