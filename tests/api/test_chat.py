import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_chat_streams_response(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/chat",
        json={"message": "how do I find a runaway process?", "session_id": "test-1"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        chunks = list(response.iter_lines())
        assert any("data:" in line for line in chunks)
        assert any('"type": "done"' in line for line in chunks)


@pytest.mark.integration
def test_chat_session_memory(client: TestClient) -> None:
    config = {"session_id": "test-memory-1"}

    with client.stream(
        "POST", "/chat", json={"message": "what causes high CPU?", **config}
    ) as r1:
        list(r1.iter_lines())  # consume first turn

    with client.stream(
        "POST", "/chat", json={"message": "what should I do next?", **config}
    ) as r2:
        chunks = " ".join(r2.iter_lines())

    assert "data:" in chunks  # second turn still responds coherently


@pytest.mark.integration
def test_chat_handles_unknown_topic(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/chat",
        json={
            "message": "how do I bake sourdough bread?",
            "session_id": "test-unknown",
        },
    ) as response:
        assert response.status_code == 200
        chunks = list(response.iter_lines())
        assert any("data:" in line for line in chunks)


# to run the test
# pytest tests/test_chat.py -v -m integration
