"""Audit trail helpers. Every meaningful action lands here so the question
"who asked for what and who decided" always has an answer."""
import json
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ..clock import get_now
from ..models import AuditLog


async def write_audit(
    session: AsyncSession,
    *,
    actor: str,
    company_id: str,
    action: str,
    request_id: str | None = None,
    details: dict | None = None,
    ts: datetime | None = None,
) -> None:
    session.add(
        AuditLog(
            ts=ts or get_now(),
            actor=actor,
            company_id=company_id,
            action=action,
            request_id=request_id,
            details=json.dumps(details or {}, default=str),
        )
    )
