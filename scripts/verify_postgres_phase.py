"""Postgres end-to-end verification, run inside containers (fake LLM).

phase1: reset+seed, escalate rhea's drop, print request/thread ids (JSON)
phase2: fresh process resumes the paused conversation with manager approval,
        then asserts booking + seats + transcript; also runs the last-seat
        race on Postgres row locks.
"""
import asyncio
import json
import sys

sys.path.insert(0, "/srv/app")
sys.path.insert(0, ".")

PHASE = sys.argv[1]

from sqlalchemy import func, select  # noqa: E402

from app.auth import Identity  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.llm import FakeChatModel  # noqa: E402
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

RHEA = Identity(employee_id="acme-rhea", company_id="acme", role="employee", name="Rhea Kapoor")


async def reset_db() -> None:
    async with engine.begin() as conn:
        for t in (ShiftRequest, AuditLog, Conversation, RosterEntry, Cab, Employee, Company):
            await conn.execute(t.__table__.delete())
    async with SessionLocal() as session:
        await seed_if_empty(session)


async def make_service() -> tuple:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from app.agent_service import ChatService
    from app.config import Settings

    uri = Settings().database_url.replace("postgresql+asyncpg://", "postgresql://")
    saver = AsyncPostgresSaver.from_conn_string(uri)
    cp = await saver.__aenter__()
    await cp.setup()
    svc = ChatService(settings=Settings(), llm=FakeChatModel(), checkpointer=cp)
    return svc, saver


async def phase1() -> None:
    await reset_db()
    # Ask for the real departure time of the cab one seat short (ACME-D2):
    # this container seeds against the live clock, so times are dynamic.
    async with SessionLocal() as session:
        target = await session.get(Cab, "ACME-D2")
        target_time = target.departure_time
    svc, saver = await make_service()
    try:
        svc.llm.script_tool_call(
            "request_shift_change", {"new_time": target_time, "trip_type": "drop", "insist": True}
        )
        result = await svc.handle_message(
            identity=RHEA,
            message=f"Move my drop to {target_time}, I know it breaks the rule",
            thread_id=None,
        )
        print("PHASE1_RESULT " + json.dumps({
            "status": result["status"],
            "request_id": result["request_id"],
            "thread_id": result["thread_id"],
            "reply": result["reply"][:160],
        }))
    finally:
        await saver.__aexit__(None, None, None)
        await engine.dispose()


async def phase2() -> None:
    phase1_out = json.loads(sys.argv[2])
    svc, saver = await make_service()
    try:
        result = await svc.resume_decision(
            thread_id=phase1_out["thread_id"],
            decision={"decision": "approve", "note": "verified on postgres", "decided_by": "acme-mgr"},
        )
        print("RESUME_REPLY", result["reply"][:160])

        async with SessionLocal() as session:
            req = await session.get(ShiftRequest, phase1_out["request_id"])
            entry = await session.get(RosterEntry, "RE-RH-D")
            d2 = await session.get(Cab, "ACME-D2")
            d1 = await session.get(Cab, "ACME-D1")
            audits = (await session.execute(select(func.count()).select_from(AuditLog))).scalar_one()
            print("REQUEST_STATUS", req.status, "| decision_by:", req.decision_by)
            print("ROSTER", entry.scheduled_time, entry.cab_id)
            print("SEATS D2:", d2.booked_seats, "D1:", d1.booked_seats)
            print("AUDIT_ROWS", audits)

        transcript = await svc.transcript(thread_id=phase1_out["thread_id"], identity=RHEA)
        texts = " ".join(m["content"] for m in transcript).lower()
        print("TRANSCRIPT_HAS_APPROVED:", "approved" in texts)

        # --- last-seat race on real Postgres row locks ---
        await reset_db()
        from app.clock import get_now
        from app.services.booking import apply_change

        async with SessionLocal() as session:
            target_time = (await session.get(Cab, "ACME-D2")).departure_time

        async def book(employee_id: str) -> dict:
            async with SessionLocal() as session:
                return await apply_change(
                    session, employee_id=employee_id, company_id="acme",
                    trip_date=get_now().strftime("%Y-%m-%d"), trip_type="drop",
                    new_time=target_time, thread_id=f"race-{employee_id}", now=get_now(),
                )

        race = await asyncio.gather(book("acme-dev"), book("acme-tara"))
        statuses = sorted(r["status"] for r in race)
        async with SessionLocal() as session:
            d2 = await session.get(Cab, "ACME-D2")
            applied = (await session.execute(
                select(func.count()).select_from(ShiftRequest).where(ShiftRequest.status == "applied")
            )).scalar_one()
        print("RACE_STATUSES", statuses, "| D2_BOOKED:", d2.booked_seats, "| APPLIED_REQUESTS:", applied)
    finally:
        await saver.__aexit__(None, None, None)
        await engine.dispose()


if PHASE == "phase1":
    asyncio.run(phase1())
else:
    asyncio.run(phase2())
