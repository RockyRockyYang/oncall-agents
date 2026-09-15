import uuid

from app.db.crud import (
    create_session_if_new,
    create_user,
    get_user_by_id,
    get_user_by_username,
    list_sessions,
)


async def test_create_session_if_new_is_idempotent():
    session_id = str(uuid.uuid4())
    await create_session_if_new(session_id, "first title")
    await create_session_if_new(session_id, "second title, should be ignored")

    matching = [s for s in await list_sessions() if s["id"] == session_id]
    assert len(matching) == 1
    assert matching[0]["title"] == "first title"


async def test_list_sessions_orders_newest_first():
    older_id = str(uuid.uuid4())
    await create_session_if_new(older_id, "older")
    newer_id = str(uuid.uuid4())
    await create_session_if_new(newer_id, "newer")

    ids = [s["id"] for s in await list_sessions()]
    assert ids.index(newer_id) < ids.index(older_id)


async def test_create_and_get_user_by_username():
    username = f"user-{uuid.uuid4().hex[:8]}"
    user = await create_user(username, "hashed-password")

    fetched = await get_user_by_username(username)
    assert fetched is not None
    assert fetched.id == user.id


async def test_get_user_by_username_not_found():
    assert await get_user_by_username(f"nonexistent-{uuid.uuid4().hex}") is None


async def test_get_user_by_id():
    username = f"user-{uuid.uuid4().hex[:8]}"
    user = await create_user(username, "hashed-password")

    fetched = await get_user_by_id(user.id)
    assert fetched is not None
    assert fetched.username == username


async def test_get_user_by_id_not_found():
    assert await get_user_by_id(str(uuid.uuid4())) is None
