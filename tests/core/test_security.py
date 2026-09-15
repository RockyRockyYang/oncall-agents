from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_is_not_plaintext():
    assert hash_password("secret123") != "secret123"


def test_verify_password_correct():
    hashed = hash_password("secret123")
    assert verify_password("secret123", hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("secret123")
    assert verify_password("wrong-password", hashed) is False


def test_access_token_roundtrip():
    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_decode_invalid_token_returns_none():
    assert decode_access_token("not.a.valid.token") is None
