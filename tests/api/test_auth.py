import uuid

from fastapi.testclient import TestClient


def _unique_username() -> str:
    return f"testuser-{uuid.uuid4().hex[:10]}"


def test_register_creates_user(client: TestClient):
    username = _unique_username()
    response = client.post("/api/auth/register", json={"username": username, "password": "secret123"})

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == username
    assert "id" in body


def test_register_duplicate_username_returns_400(client: TestClient):
    username = _unique_username()
    client.post("/api/auth/register", json={"username": username, "password": "secret123"})

    response = client.post("/api/auth/register", json={"username": username, "password": "different"})
    assert response.status_code == 400


def test_login_wrong_password_returns_401(client: TestClient):
    username = _unique_username()
    client.post("/api/auth/register", json={"username": username, "password": "secret123"})

    response = client.post("/api/auth/login", json={"username": username, "password": "wrong"})
    assert response.status_code == 401


def test_login_unknown_username_returns_401(client: TestClient):
    response = client.post(
        "/api/auth/login", json={"username": _unique_username(), "password": "whatever"}
    )
    assert response.status_code == 401


def test_login_success_returns_token(client: TestClient):
    username = _unique_username()
    client.post("/api/auth/register", json={"username": username, "password": "secret123"})

    response = client.post("/api/auth/login", json={"username": username, "password": "secret123"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_me_with_valid_token_returns_current_user(client: TestClient):
    username = _unique_username()
    client.post("/api/auth/register", json={"username": username, "password": "secret123"})
    token = client.post(
        "/api/auth/login", json={"username": username, "password": "secret123"}
    ).json()["access_token"]

    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["username"] == username


def test_me_without_token_returns_401(client: TestClient):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_with_invalid_token_returns_401(client: TestClient):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer garbage.token.here"})
    assert response.status_code == 401
