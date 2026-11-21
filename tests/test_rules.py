"""Rules enforced in code: cut-off, daily limit, operating window, bookings."""
from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import AuditLog, Cab, RosterEntry, ShiftRequest
from tests.conftest import TOMORROW, auth_token, chat, fetch_one


async def _request_count(employee_id: str) -> int:
    async with SessionLocal() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(ShiftRequest)
                .where(ShiftRequest.employee_id == employee_id)
            )
        ).scalar_one()


async def test_cutoff_rejection_blocks_booking(client, fake_llm):
    """Acme cut-off is 120 minutes. Rhea's drop is 90 minutes away (18:30 at
    17:00 IST on the frozen clock) -> a self-service change must be refused."""
    token = await auth_token(client, "acme-rhea")
    fake_llm.script_tool_call("request_shift_change", {"new_time": "20:00", "trip_type": "drop"})
    result = await chat(client, token, "Move my drop to 20:00 please")

    assert "120 minutes" in result["reply"]
    assert await _request_count("acme-rhea") == 0
    # the attempt is on the audit trail
    async with SessionLocal() as session:
        audits = (
            await session.execute(select(AuditLog).where(AuditLog.action == "change_rejected_by_rule"))
        ).scalars().all()
        assert len(audits) == 1
    # her roster is untouched
    entry = await fetch_one(RosterEntry, id="RE-RH-D")
    assert entry.scheduled_time == "18:30"


async def test_cutoff_is_per_company(client, fake_llm):
    """Globex's cut-off is 30 minutes. Priya's drop is 150 minutes away, so the
    same kind of request Acme rejects is allowed here - rules applied exactly,
    per company, from code."""
    token = await auth_token(client, "globex-priya")
    fake_llm.script_tool_call("request_shift_change", {"new_time": "20:00", "trip_type": "drop"})
    result = await chat(client, token, "Move my drop to 20:00")

    assert result["status"] == "ok"
    assert "20:00" in result["reply"]
    entry = await fetch_one(RosterEntry, id="RE-PR-D")
    assert entry.scheduled_time == "20:00"
    assert entry.cab_id == "GLOB-D3"


async def test_allowed_change_moves_roster_and_seat(client, fake_llm):
    """A rule-compliant change books immediately: seat taken on the new cab,
    released on the old one, request row + audit written."""
    token = await auth_token(client, "acme-rhea")
    fake_llm.script_tool_call(
        "request_shift_change", {"new_time": "09:15", "trip_type": "pickup", "date": TOMORROW}
    )
    result = await chat(client, token, "Change my pickup tomorrow to 09:15")

    assert "09:15" in result["reply"]
    entry = await fetch_one(RosterEntry, id="RE-RH-PT")
    assert entry.scheduled_time == "09:15"
    assert entry.cab_id == "ACME-PT2"
    assert (await fetch_one(Cab, id="ACME-PT2")).booked_seats == 2
    assert (await fetch_one(Cab, id="ACME-PT1")).booked_seats == 0
    assert await _request_count("acme-rhea") == 1


async def test_daily_limit_rejects_second_change(client, fake_llm):
    """Acme allows 1 change/day. Sam's first change books; the second is
    refused even though the cut-off would allow it."""
    token = await auth_token(client, "acme-sam")
    fake_llm.script_tool_call("request_shift_change", {"new_time": "20:30", "trip_type": "drop"})
    first = await chat(client, token, "Move my drop to 20:30")
    assert first["status"] == "ok"

    fake_llm.script_tool_call("request_shift_change", {"new_time": "21:00", "trip_type": "drop"})
    second = await chat(client, token, "Actually move my drop to 21:00 instead")
    assert "already used them" in second["reply"]
    assert await _request_count("acme-sam") == 1  # only the first booking exists


async def test_outside_operating_window_rejected(client, fake_llm):
    """23:00 is outside Acme's 06:00-22:00 window -> refused in code."""
    token = await auth_token(client, "acme-sam")
    fake_llm.script_tool_call("request_shift_change", {"new_time": "23:00", "trip_type": "drop"})
    result = await chat(client, token, "Can you move my drop to 23:00?")

    assert "06:00" in result["reply"] and "22:00" in result["reply"]
    assert await _request_count("acme-sam") == 0


async def test_tool_asks_when_trip_type_unclear(client, fake_llm):
    """A change request that doesn't name pickup or drop, for an employee with
    two trips today: the tool refuses to guess and asks - deterministically,
    even if the model would have guessed."""
    token = await auth_token(client, "acme-rhea")
    fake_llm.script_tool_call("request_shift_change", {"new_time": "20:00"})
    result = await chat(client, token, "Move me to 20:00")

    assert "pickup" in result["reply"].lower() and "drop" in result["reply"].lower()
    assert await _request_count("acme-rhea") == 0
