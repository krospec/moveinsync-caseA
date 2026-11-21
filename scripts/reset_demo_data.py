"""Reset demo data inside the app container (truncates business tables and
reseeds the two companies against the current clock):

    docker compose exec app python scripts/reset_demo_data.py
"""
import asyncio
import sys

sys.path.insert(0, ".")

from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal, engine  # noqa: E402
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


async def main() -> None:
    async with engine.begin() as conn:
        # children before parents: Postgres enforces FKs immediately
        for t in (ShiftRequest, AuditLog, Conversation, RosterEntry, Cab, Employee, Company):
            await conn.execute(t.__table__.delete())
    async with SessionLocal() as session:
        await seed_if_empty(session)
        companies = (await session.execute(select(func.count()).select_from(Company))).scalar_one()
    await engine.dispose()
    print(f"Demo data reset: {companies} companies reseeded against the current clock.")


if __name__ == "__main__":
    asyncio.run(main())
