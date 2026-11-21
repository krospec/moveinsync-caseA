"""Employee-facing endpoints: chat, transcript, own requests."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..agent_service import ChatService, ConversationNotOwned
from ..auth import Identity, get_identity
from ..db import SessionLocal
from ..schemas import ChatIn, ChatOut, MessageOut, RequestOut
from ..services import booking

log = logging.getLogger("shift.api.chat")
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn, request: Request, identity: Identity = Depends(get_identity)) -> ChatOut:
    service: ChatService = request.app.state.chat_service
    try:
        result = await service.handle_message(
            identity=identity, message=body.message, thread_id=body.thread_id
        )
    except ConversationNotOwned:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This conversation belongs to another employee.",
        )
    return ChatOut(**result)


@router.get("/chat/{thread_id}/messages", response_model=list[MessageOut])
async def transcript(thread_id: str, request: Request, identity: Identity = Depends(get_identity)) -> list[MessageOut]:
    service: ChatService = request.app.state.chat_service
    try:
        rows = await service.transcript(thread_id=thread_id, identity=identity)
    except ConversationNotOwned:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This conversation belongs to another employee.",
        )
    return [MessageOut(**r) for r in rows]


def _request_out(r) -> RequestOut:  # noqa: ANN001
    return RequestOut(
        id=r.id,
        employee_id=r.employee_id,
        company_id=r.company_id,
        trip_type=r.trip_type,
        trip_date=r.trip_date,
        old_time=r.old_time or "",
        requested_time=r.requested_time,
        status=r.status,
        reason=r.reason,
        decision_by=r.decision_by,
        decision_note=r.decision_note,
    )


@router.get("/my/requests", response_model=list[RequestOut])
async def my_requests(identity: Identity = Depends(get_identity)) -> list[RequestOut]:
    async with SessionLocal() as session:  # type: AsyncSession
        rows = await booking.requests_for_employee(session, identity.employee_id, identity.company_id)
        return [_request_out(r) for r in rows]
