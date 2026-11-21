"""Test bootstrap.

Env is set BEFORE any app import so the (cached) settings object sees:
  * a frozen clock (2026-01-15 17:00 IST) -> cut-off rules are deterministic
  * LLM_PROVIDER=fake                    -> no API key, no quota, no flakiness
  * a file-backed sqlite DB              -> service data + LangGraph checkpoints
                                            persist, which the restart test uses
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="sca_tests_"))
TEST_DB = TEST_DATA_DIR / "test.db"

os.environ["FROZEN_NOW"] = "2026-01-15T17:00:00+05:30"
os.environ["LLM_PROVIDER"] = "fake"
os.environ["JWT_SECRET"] = "test-secret-that-is-long-enough-32bytes"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB.as_posix()}"

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.agent.tools import clear_executed  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.llm import FakeChatModel  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import (  # noqa: E402
    AuditLog,
    Cab,
    Company,
    Conversation,
    Employee,
    RosterEntry,
    ShiftRequest,
)
from app.seed import seed_if_empty  # noqa: E402

FROZEN_NOW = "2026-01-15T17:00:00+05:30"
TODAY = "2026-01-15"
TOMORROW = "2026-01-16"


@pytest_asyncio.fixture(scope="session")
async def app():
    """One app instance for the session: lifespan runs schema bootstrap,
    seeding, and the sqlite checkpointer exactly once."""
    application = create_app()
    async with application.router.lifespan_context(application):
        yield application


@pytest_asyncio.fixture(scope="session")
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture(autouse=True)
async def fresh_db(app):
    """Empty + reseed the business tables around every test. Checkpoint data
    stays (thread ids are unique per test), so nothing leaks semantically."""
    async with engine.begin() as conn:
        for table in (
            ShiftRequest, AuditLog, Conversation, RosterEntry, Cab, Employee, Company,
        ):
            await conn.execute(table.__table__.delete())
    async with SessionLocal() as session:
        await seed_if_empty(session)
    clear_executed()
    app.state.chat_service.llm.calls.clear()
    yield


@pytest.fixture
def fake_llm(app) -> FakeChatModel:
    return app.state.chat_service.llm


async def auth_token(client: AsyncClient, employee_id: str) -> str:
    resp = await client.post("/auth/token", json={"employee_id": employee_id})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def chat(client: AsyncClient, token: str, message: str, thread_id: str | None = None) -> dict:
    resp = await client.post(
        "/chat",
        json={"message": message, "thread_id": thread_id},
        headers=auth_header(token),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def count_rows(model) -> int:
    from sqlalchemy import select, func

    async with SessionLocal() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def fetch_one(model, **filters):
    from sqlalchemy import select

    async with SessionLocal() as session:
        stmt = select(model)
        for k, v in filters.items():
            stmt = stmt.where(getattr(model, k) == v)
        return (await session.execute(stmt)).scalars().first()
