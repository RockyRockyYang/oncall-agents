import uuid

from fastapi.testclient import TestClient

from app.db.crud import create_session_if_new


async def test_get_sessions_includes_newly_created_session(client: TestClient):
    session_id = str(uuid.uuid4())
    await create_session_if_new(session_id, "sessions api test")

    response = client.get("/api/sessions")

    assert response.status_code == 200
    ids = [s["id"] for s in response.json()["sessions"]]
    assert session_id in ids


async def test_delete_session_removes_it_from_the_list(client: TestClient):
    session_id = str(uuid.uuid4())
    await create_session_if_new(session_id, "to be deleted via api")

    response = client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json() == {"session_id": session_id, "deleted": True}

    ids = [s["id"] for s in client.get("/api/sessions").json()["sessions"]]
    assert session_id not in ids
