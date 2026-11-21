"""Async SQLAlchemy engine + session factory."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from .config import get_settings


class Base(DeclarativeBase):
    pass


def _build_engine():
    url = get_settings().database_url
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        # In-memory sqlite (tests) needs a single shared connection;
        # file sqlite gets a busy timeout so concurrent writers wait
        # instead of failing.
        kwargs["connect_args"] = {"timeout": 15}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {}
    return create_async_engine(url, **kwargs)


engine = _build_engine()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def create_all() -> None:
    """Demo-grade schema bootstrap. Real deployments would use migrations."""
    from . import models  # noqa: F401  (register tables)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
