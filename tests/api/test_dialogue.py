import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_multi_turn_remembers_context(client: TestClient):
    # 第一轮：告诉 agent 一个事实
    r1 = client.post(
        "/chat",
        json={
            "session_id": "dialogue-test-1",
            "message": "My name is Rocky.",
        },
    )
    assert r1.status_code == 200

    # 第二轮：问 agent 刚才说的内容，验证它记住了
    r2 = client.post(
        "/chat",
        json={
            "session_id": "dialogue-test-1",
            "message": "What is my name?",
        },
    )
    assert r2.status_code == 200


@pytest.mark.integration
def test_different_sessions_are_isolated(client: TestClient):
    # session A 说一个事实
    client.post(
        "/chat",
        json={
            "session_id": "session-a",
            "message": "My favorite color is blue.",
        },
    )

    # session B 不应该知道 session A 的内容
    r = client.post(
        "/chat",
        json={
            "session_id": "session-b",
            "message": "What is my favorite color?",
        },
    )
    assert r.status_code == 200
