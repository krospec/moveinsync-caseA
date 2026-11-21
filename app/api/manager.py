"""Manager endpoints: see pending requests for their company, decide one.

A decision does two things:
1. Records it in the DB + audit trail (approval service).
2. Resumes the paused conversation that raised the request. The paused state
   lives in the Postgres checkpointer, so this works even after a restart.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent_service import ChatService
from ..auth import Identity, require_manager
from ..db import SessionLocal
from ..models import Employee
from ..schemas import DecisionIn, RequestOut
from ..services import approval, booking
from ..services.approval import NotAllowed
from .chat import _request_out

log = logging.getLogger("shift.api.manager")
router = APIRouter(tags=["manager"])


@router.get("/manager/requests", response_model=list[RequestOut])
async def pending_requests(identity: Identity = Depends(require_manager)) -> list[RequestOut]:
    async with SessionLocal() as session:  # type: AsyncSession
        rows = await approval.pending_for_manager(session, await _manager(session, identity))
        return [_request_out(r) for r in rows]


@router.post("/manager/requests/{request_id}/decision", response_model=RequestOut)
async def decide(
    request_id: str,
    body: DecisionIn,
    request: Request,
    identity: Identity = Depends(require_manager),
) -> RequestOut:
    from ..clock import get_now

    service: ChatService = request.app.state.chat_service
    async with SessionLocal() as session:  # type: AsyncSession
        manager = await _manager(session, identity)
        try:
            outcome = await approval.decide(
                session,
                request_id=request_id,
                manager=manager,
                decision=body.decision,
                note=body.note,
                now=get_now(),
            )
        except NotAllowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "This request does not belong to your company.",
            )

        # Resume the employee's paused conversation so they get the decision.
        # already_decided means no conversation is waiting - nothing to resume.
        if outcome["status"] not in ("already_decided",):
            req = await booking.get_request(session, request_id)
            if req is not None and req.thread_id:
                try:
                    await service.resume_decision(
                        thread_id=req.thread_id,
                        decision={
                            "decision": body.decision,
                            "note": body.note,
                            "decided_by": manager.id,
                        },
                    )
                except RuntimeError:
                    # Paused conversation not found (e.g. checkpoint pruned).
                    # The DB decision stands; flag it instead of failing.
                    log.warning("decision %s recorded but resume failed", request_id)
                    outcome["resume_failed"] = True

    updated = None
    async with SessionLocal() as session:
        from ..models import ShiftRequest

        updated = await session.get(ShiftRequest, request_id)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
    out = _request_out(updated)
    return out


async def _manager(session: AsyncSession, identity: Identity) -> Employee:
    manager = await session.get(Employee, identity.employee_id)
    if manager is None or manager.company_id != identity.company_id or manager.role != "manager":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Unknown or invalid manager account.")
    return manager
