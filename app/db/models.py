import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(PG_UUID(as_uuid=False), nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserRow(Base):
    __tablename__ = "users"

    # session 的 id 是前端生成好传过来的，但注册时客户端不知道 id，得由后端自己生成
    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
