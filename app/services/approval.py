"""Manager approval path.

A manager decision is applied by plain code, not by the model: approve means
"run the same booking path a rule-compliant request would take", reject means
"close the request". The LLM only words the message the employee will read.
"""
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    REQUEST_APPLIED,
    REQUEST_FAILED,
    REQUEST_PENDING,
    REQUEST_REJECTED,
    Employee,
    RosterEntry,
    ShiftRequest,
)
from . import audit, booking


class NotAllowed(Exception):
    """Manager and request belong to different companies (or other denial)."""


async def message_for_request(session: AsyncSession, req: ShiftRequest) -> str:
    """Employee-facing wording for a request that has reached a terminal
    state - used both right after a decision and when resuming a conversation
    whose request was already decided (e.g. decided before the resume)."""
    if req.status == REQUEST_APPLIED:
        entry = await session.get(RosterEntry, req.roster_entry_id)
        cab = entry.cab_id if entry else "?"
        return (
            f"Good news - your transport manager approved your request. Your "
            f"{req.trip_type} on {req.trip_date} is now at {req.requested_time} (cab {cab})."
        )
    if req.status == REQUEST_REJECTED:
        return (
            "Your transport manager reviewed the request and declined it. "
            "Your original schedule is unchanged."
        )
    if req.status == REQUEST_FAILED:
        return (
            "Your transport manager approved your request, but the seat on that cab "
            "was taken while it was waiting. Your schedule is unchanged - ask me for "
            "the nearest slots and I'll rebook you."
        )
    return f"Request {req.id} is currently {req.status.replace('_', ' ')}."


async def decide(
    session: AsyncSession,
    *,
    request_id: str,
    manager: Employee,
    decision: str,  # "approve" | "reject"
    note: str | None,
    now: datetime,
) -> dict:
    req = await session.get(ShiftRequest, request_id)
    if req is None or req.company_id != manager.company_id:
        # A manager of company A can never even *see*, let alone decide,
        # a request from company B.
        raise NotAllowed("request not found for this manager's company")

    if req.status != REQUEST_PENDING:
        return {
            "status": "already_decided",
            "request_status": req.status,
            "message": await message_for_request(session, req),
        }

    req.decision_by = manager.id
    req.decision_note = note
    req.decided_at = now

    if decision == "reject":
        req.status = REQUEST_REJECTED
        req.updated_at = now
        await audit.write_audit(
            session,
            actor=manager.id,
            company_id=manager.company_id,
            action="manager_rejected",
            request_id=req.id,
            details={"note": note},
            ts=now,
        )
        await session.commit()
        return {
            "status": REQUEST_REJECTED,
            "request_id": req.id,
            "message": await message_for_request(session, req),
        }

    # Approved: run the identical booking path used for rule-compliant changes.
    entry = await session.get(RosterEntry, req.roster_entry_id)
    if entry is None:
        req.status = REQUEST_FAILED
        await session.commit()
        return {
            "status": REQUEST_FAILED,
            "request_id": req.id,
            "message": "The roster entry behind this request no longer exists.",
        }

    result = await booking.move_roster_entry(
        session, entry=entry, new_time=req.requested_time, now=now
    )
    if result["ok"]:
        req.status = REQUEST_APPLIED
        req.updated_at = now
        req.idempotency_key = booking.idem_key(
            req.employee_id, req.roster_entry_id, req.trip_date, req.requested_time
        )
        await audit.write_audit(
            session,
            actor=manager.id,
            company_id=manager.company_id,
            action="manager_approved_and_booked",
            request_id=req.id,
            details={"to_cab": result["cab"], "new_time": req.requested_time},
            ts=now,
        )
    else:
        # Manager approved, but the seat was taken while everyone waited.
        req.status = REQUEST_FAILED
        req.decision_note = (note or "") + " [seat no longer available at approval time]"
        req.updated_at = now
        await audit.write_audit(
            session,
            actor=manager.id,
            company_id=manager.company_id,
            action="manager_approved_but_seat_gone",
            request_id=req.id,
            details={"reason": result["reason"]},
            ts=now,
        )

    await session.commit()
    return {"status": req.status, "request_id": req.id, "message": await message_for_request(session, req)}


async def pending_for_manager(session: AsyncSession, manager: Employee) -> list[ShiftRequest]:
    stmt = (
        select(ShiftRequest)
        .where(ShiftRequest.company_id == manager.company_id, ShiftRequest.status == REQUEST_PENDING)
        .order_by(ShiftRequest.created_at.asc())
    )
    return list((await session.execute(stmt)).scalars().all())
