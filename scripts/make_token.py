"""Print a demo JWT for a seeded employee, e.g.:

    python scripts/make_token.py acme-rhea
    python scripts/make_token.py acme-mgr
"""
import sys

sys.path.insert(0, ".")

from app.auth import create_token  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import Employee  # noqa: E402
from sqlalchemy import select  # noqa: E402


async def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/make_token.py <employee_id>")
        raise SystemExit(2)
    async with SessionLocal() as session:
        employee = (await session.execute(
            select(Employee).where(Employee.id == sys.argv[1])
        )).scalars().first()
    if employee is None:
        print(f"unknown employee id: {sys.argv[1]}")
        raise SystemExit(1)
    print(create_token(employee, get_settings()))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
