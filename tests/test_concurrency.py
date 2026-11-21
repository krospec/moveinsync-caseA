"""Bonus requirement: exactly one seat given, exactly one request created.

The seat grab is a single conditional UPDATE (booked_seats < capacity), which
is atomic on Postgres (row lock) and on sqlite (single writer + busy timeout),
so the same code is provably correct in tests and in production.
"""
import asyncio

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import Cab, ShiftRequest
from app.services.booking import apply_change
from tests.conftest import TODAY, auth_token, chat


def _now():
    from app.clock import get_now

    return get_now()


async def test_double_submit_creates_one_request_and_takes_one_seat(client, fake_llm):
    """Same employee submits the same change twice (double-tap on 'send') ->
    second attempt returns the existing request; no duplicate seat grab."""
    token = await auth_token(client, "acme-rhea")
    args = {"new_time": "09:15", "trip_type": "pickup", "date": "2026-01-16"}
    fake_llm.script_tool_call("request_shift_change", args)
    first = await chat(client, token, "Change my pickup tomorrow to 09:15")
    assert first["status"] == "ok"

    fake_llm.script_tool_call("request_shift_change", args)
    second = await chat(client, token, "Change my pickup tomorrow to 09:15", thread_id=first["thread_id"])
    assert "already" in second["reply"].lower()

    async with SessionLocal() as session:
        total = (
            await session.execute(
                select(func.count())
                .select_from(ShiftRequest)
                .where(ShiftRequest.employee_id == "acme-rhea")
            )
        ).scalar_one()
        assert total == 1
        assert (await session.get(Cab, "ACME-PT2")).booked_seats == 2


async def test_two_employees_one_last_seat_exactly_one_wins():
    """Dev and Tara both sit on the 21:00 cab; both ask for the 20:00 drop cab
    which has exactly one seat left. Concurrent submissions: exactly one
    'applied', one refused as full; final occupancy is 4/4; one request row."""
    results = await asyncio.gather(
        _book("acme-dev"),
        _book("acme-tara"),
    )
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["applied", "full"]  # exactly one winner, exactly one loser

    async with SessionLocal() as session:
        assert (await session.get(Cab, "ACME-D2")).booked_seats == 4  # never overbooked
        assert (await session.get(Cab, "ACME-DF")).booked_seats == 3  # loser's old cab untouched
        total = (
            await session.execute(
                select(func.count()).select_from(ShiftRequest).where(ShiftRequest.status == "applied")
            )
        ).scalar_one()
        assert total == 1


async def _book(employee_id: str) -> dict:
    async with SessionLocal() as session:
        return await apply_change(
            session,
            employee_id=employee_id,
            company_id="acme",
            trip_date=TODAY,
            trip_type="drop",
            new_time="20:00",
            thread_id=f"race-{employee_id}",
            now=_now(),
        )


async def test_idempotency_at_service_level():
    """Calling the booking engine twice with identical arguments collapses onto
    one request - independent of any LLM behaviour."""
    first = await _book_once("acme-rhea")
    second = await _book_once("acme-rhea")
    assert first["status"] == "applied"
    assert second["status"] == "already_done"
    assert second["request_id"] == first["request_id"]

    async with SessionLocal() as session:
        assert (await session.get(Cab, "ACME-PT2")).booked_seats == 2


async def _book_once(employee_id: str) -> dict:
    async with SessionLocal() as session:
        return await apply_change(
            session,
            employee_id=employee_id,
            company_id="acme",
            trip_date="2026-01-16",
            trip_type="pickup",
            new_time="09:15",
            thread_id=f"idem-{employee_id}",
            now=_now(),
        )
