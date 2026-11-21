"""The booking engine: rule checks, seat moves, idempotency, escalation.

Design notes (defended in the README):

* Seat grabs are a single conditional UPDATE (`... WHERE booked_seats < capacity`).
  One statement is atomic on both Postgres (row lock, re-evaluated at commit
  under READ COMMITTED) and sqlite (single writer), which is what makes the
  "two employees, one seat" race provably safe without an ORM-level check-then-
  write race window.
* A shift request is idempotent on (employee, roster entry, date, new time).
  A double submit returns the existing request instead of creating a second one.
* Rule breaches never book. `insist=True` is the only path to manager
  escalation, and it still creates nothing but a pending_approval row.
"""
import re
import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .. import policy
from ..models import (
    REQUEST_APPLIED,
    REQUEST_PENDING,
    Cab,
    Company,
    RosterEntry,
    ShiftRequest,
)
from . import audit, cabs

TIME_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")

TODAY = "today"  # semantic marker: trip_date == today's IST date


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def idem_key(employee_id: str, entry_id: str, trip_date: str, new_time: str) -> str:
    return f"{employee_id}|{entry_id}|{trip_date}|{new_time}"


async def _grab_seat(session: AsyncSession, cab_id: str) -> bool:
    """Atomically take one seat. Returns False when the cab is already full."""
    res = await session.execute(
        update(Cab)
        .where(Cab.id == cab_id, Cab.booked_seats < Cab.capacity)
        .values(booked_seats=Cab.booked_seats + 1)
    )
    return res.rowcount == 1


async def _release_seat(session: AsyncSession, cab_id: str) -> bool:
    res = await session.execute(
        update(Cab)
        .where(Cab.id == cab_id, Cab.booked_seats > 0)
        .values(booked_seats=Cab.booked_seats - 1)
    )
    return res.rowcount == 1


async def count_changes_today(session: AsyncSession, employee_id: str, trip_date: str) -> int:
    stmt = select(ShiftRequest).where(
        ShiftRequest.employee_id == employee_id,
        ShiftRequest.trip_date == trip_date,
        ShiftRequest.status.in_([REQUEST_APPLIED]),
    )
    return len((await session.execute(stmt)).scalars().all())


async def move_roster_entry(
    session: AsyncSession,
    *,
    entry: RosterEntry,
    new_time: str,
    now: datetime,
) -> dict:
    """Move an entry to this company's cab departing at new_time.

    Returns {"ok": True, "cab": {...}} or {"ok": False, "reason": "full"}.
    Raises nothing: all failure modes are values.
    """
    target = await cabs.find_cab_at_time(
        session,
        company_id=entry.company_id,
        trip_date=entry.trip_date,
        trip_type=entry.trip_type,
        departure_time=new_time,
    )
    if target is None:
        return {"ok": False, "reason": "no_such_slot"}
    if target.id == entry.cab_id:
        return {"ok": False, "reason": "already_on_cab"}

    old_cab_id = entry.cab_id
    old_time = entry.scheduled_time
    if not await _grab_seat(session, target.id):
        return {"ok": False, "reason": "full", "cab": target.id}

    # Only release the old seat after the new one is secured.
    await _release_seat(session, old_cab_id)

    entry.cab_id = target.id
    entry.scheduled_time = new_time
    entry.status = "changed"
    entry.changed_from = old_time
    entry.updated_at = now
    return {"ok": True, "cab": target.id, "old_time": old_time, "old_cab": old_cab_id}


async def apply_change(
    session: AsyncSession,
    *,
    employee_id: str,
    company_id: str,
    trip_date: str,
    trip_type: str,
    new_time: str,
    thread_id: str,
    reason: str | None = None,
    insist: bool = False,
    now: datetime,
) -> dict:
    """Employee-facing change request. Identity (employee_id, company_id) comes
    from the JWT, never from the model."""
    if not TIME_RE.match(new_time or ""):
        return {
            "status": "invalid_time",
            "message": f"'{new_time}' is not a valid 24-hour time like 20:00.",
        }

    entry = (
        await session.execute(
            select(RosterEntry)
            .where(
                RosterEntry.employee_id == employee_id,
                RosterEntry.company_id == company_id,
                RosterEntry.trip_date == trip_date,
                RosterEntry.trip_type == trip_type,
            )
            .order_by(RosterEntry.trip_date.desc())
        )
    ).scalars().first()

    if entry is None:
        return {
            "status": "not_found",
            "message": (
                f"I could not find a {trip_type or 'roster entry'} for you on {trip_date}. "
                "Ask me for your roster and we'll go from there."
            ),
        }

    # Idempotency: same employee + same entry + same new time collapses.
    key = idem_key(employee_id, entry.id, entry.trip_date, new_time)
    existing = (
        await session.execute(
            select(ShiftRequest).where(
                ShiftRequest.idempotency_key == key,
                ShiftRequest.status.in_([REQUEST_APPLIED, REQUEST_PENDING]),
            )
        )
    ).scalars().first()
    if existing is not None:
        return {
            "status": "already_done",
            "request_id": existing.id,
            "message": (
                f"This change is already recorded ({existing.status.replace('_', ' ')}). "
                "No duplicate was created."
            ),
        }

    if entry.scheduled_time == new_time:
        return {
            "status": "no_change",
            "message": f"You are already scheduled on the {new_time} {trip_type}.",
        }

    company = await session.get(Company, company_id)
    changes_today = await count_changes_today(session, employee_id, entry.trip_date)
    checks = policy.validate_change(company, entry, new_time, changes_today, now)
    blocked = policy.blocking(checks)

    if blocked and not insist:
        options = await cabs.find_options(
            session,
            company_id=company_id,
            trip_date=entry.trip_date,
            trip_type=entry.trip_type,
            target_time=new_time,
        )
        await audit.write_audit(
            session,
            actor=employee_id,
            company_id=company_id,
            action="change_rejected_by_rule",
            details={"codes": [c.code for c in blocked], "requested_time": new_time},
            ts=now,
        )
        await session.commit()
        return {
            "status": "rejected_by_rule",
            "request_id": None,
            "checks": [{"code": c.code, "message": c.message} for c in blocked],
            "message": blocked[0].message,
            "options": options,
        }

    if blocked and insist:
        # Rule still broken; only a human manager can override. Create the
        # pending request, then the graph pauses for the manager's decision.
        req = ShiftRequest(
            id=_uuid(),
            employee_id=employee_id,
            company_id=company_id,
            thread_id=thread_id,
            roster_entry_id=entry.id,
            trip_date=entry.trip_date,
            trip_type=entry.trip_type,
            old_time=entry.scheduled_time,
            requested_time=new_time,
            reason=reason,
            status=REQUEST_PENDING,
            created_at=now,
            updated_at=now,
        )
        session.add(req)
        await audit.write_audit(
            session,
            actor=employee_id,
            company_id=company_id,
            action="escalated_to_manager",
            request_id=req.id,
            details={"codes": [c.code for c in blocked], "requested_time": new_time},
            ts=now,
        )
        await session.commit()
        return {
            "status": "escalated",
            "request_id": req.id,
            "checks": [{"code": c.code, "message": c.message} for c in blocked],
            "message": (
                "Your request breaks your company's rule, so I've sent it to your "
                "transport manager for approval. I'll confirm here as soon as they decide."
            ),
        }

    # Rule-compliant: book it now.
    old_time = entry.scheduled_time
    result = await move_roster_entry(session, entry=entry, new_time=new_time, now=now)
    if not result["ok"]:
        reason_code = result["reason"]
        options = await cabs.find_options(
            session,
            company_id=company_id,
            trip_date=entry.trip_date,
            trip_type=entry.trip_type,
            target_time=new_time,
        )
        if reason_code == "no_such_slot":
            message = (
                f"There is no {trip_type} cab at {new_time} on {entry.trip_date} "
                "for your company."
            )
        elif reason_code == "already_on_cab":
            message = f"You are already on the {new_time} {trip_type} cab."
        else:
            message = f"The {new_time} cab just filled up."
        if options:
            listed = "; ".join(
                f"{o['departure_time']} ({o['seats_free']} seat{'s' if o['seats_free'] != 1 else ''} free)"
                for o in options
            )
            message += f" Nearest with free seats: {listed}."
        await audit.write_audit(
            session,
            actor=employee_id,
            company_id=company_id,
            action=f"booking_failed_{reason_code}",
            details={"requested_time": new_time},
            ts=now,
        )
        await session.commit()
        return {"status": reason_code, "message": message, "options": options}

    req = ShiftRequest(
        id=_uuid(),
        employee_id=employee_id,
        company_id=company_id,
        thread_id=thread_id,
        roster_entry_id=entry.id,
            trip_date=entry.trip_date,
            trip_type=entry.trip_type,
            old_time=old_time,
            requested_time=new_time,
        reason=reason,
        status=REQUEST_APPLIED,
        idempotency_key=key,
        created_at=now,
        updated_at=now,
    )
    session.add(req)
    await audit.write_audit(
        session,
        actor=employee_id,
        company_id=company_id,
        action="change_applied",
        request_id=req.id,
        details={
            "from": result.get("old_time"),
            "from_cab": result.get("old_cab"),
            "to_cab": result["cab"],
            "new_time": new_time,
        },
        ts=now,
    )
    await session.commit()
    return {
        "status": "applied",
        "request_id": req.id,
        "message": (
            f"Done - your {entry.trip_type} on {entry.trip_date} is now at {new_time} "
            f"(cab {result['cab']})."
        ),
    }


async def get_request(session: AsyncSession, request_id: str) -> ShiftRequest | None:
    return await session.get(ShiftRequest, request_id)


async def latest_request_for(session: AsyncSession, employee_id: str, company_id: str) -> ShiftRequest | None:
    stmt = (
        select(ShiftRequest)
        .where(ShiftRequest.employee_id == employee_id, ShiftRequest.company_id == company_id)
        .order_by(ShiftRequest.created_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def requests_for_company(
    session: AsyncSession, company_id: str, status: str | None = None
) -> list[ShiftRequest]:
    stmt = (
        select(ShiftRequest)
        .where(ShiftRequest.company_id == company_id)
        .order_by(ShiftRequest.created_at.desc())
    )
    if status:
        stmt = stmt.where(ShiftRequest.status == status)
    return list((await session.execute(stmt)).scalars().all())


async def requests_for_employee(
    session: AsyncSession, employee_id: str, company_id: str
) -> list[ShiftRequest]:
    stmt = (
        select(ShiftRequest)
        .where(ShiftRequest.employee_id == employee_id, ShiftRequest.company_id == company_id)
        .order_by(ShiftRequest.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())
