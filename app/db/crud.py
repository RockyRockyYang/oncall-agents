from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.engine import async_session_factory
from app.db.models import SessionRow, UserRow


async def create_session_if_new(session_id: str, title: str) -> None:
    async with async_session_factory() as session:
        stmt = (
            pg_insert(SessionRow)
            .values(id=session_id, title=title[:50])
            .on_conflict_do_nothing(index_elements=["id"])
        )
        await session.execute(stmt)
        await session.commit()


async def list_sessions() -> list[dict]:
    async with async_session_factory() as session:
        result = await session.execute(select(SessionRow).order_by(SessionRow.created_at.desc()))
        rows = result.scalars().all()
        return [{"id": row.id, "title": row.title} for row in rows]


async def delete_session(session_id: str) -> None:
    async with async_session_factory() as session:
        await session.execute(delete(SessionRow).where(SessionRow.id == session_id))
        await session.commit()


async def create_user(username: str, password_hash: str) -> UserRow:
    async with async_session_factory() as session:
        user = UserRow(username=username, password_hash=password_hash)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def get_user_by_username(username: str) -> UserRow | None:
    async with async_session_factory() as session:
        result = await session.execute(select(UserRow).where(UserRow.username == username))
        return result.scalar_one_or_none()


async def get_user_by_id(user_id: str) -> UserRow | None:
    async with async_session_factory() as session:
        return await session.get(UserRow, user_id)
