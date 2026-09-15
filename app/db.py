from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import SessionRow

engine = create_async_engine(settings.database_url)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


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
