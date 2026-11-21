"""The offline fallback (no API key): plain-code intent routing through the
same tools and rules. These tests run the service with LLM_PROVIDER=groq and
no key - exactly what an evaluator gets from `docker compose up` unconfigured.
"""
import pytest_asyncio

from app.llm import OfflineFallbackModel
from tests.conftest import auth_token, chat, fetch_one
from app.models import RosterEntry, ShiftRequest


@pytest_asyncio.fixture
async def offline_app(app):
    """Swap the session app's model for the offline one (same graph/rules)."""
    from app.agent_service import ChatService

    original = app.state.chat_service
    service = ChatService(
        settings=original.settings,
        llm=OfflineFallbackModel(),
        checkpointer=original.graph.checkpointer,
    )
    app.state.chat_service = service
    yield service
    app.state.chat_service = original


async def test_offline_roster_question(client, app, offline_app):
    token = await auth_token(client, "acme-rhea")
    result = await chat(client, token, "What time is my drop today?")
    assert "18:30" in result["reply"]
    assert "Traceback" not in result["reply"]


async def test_offline_change_request_books_and_rejects_by_rule(client, app, offline_app):
    """Allowed change books without any LLM; rule-breaking one is refused by
    the same code path as with a real model."""
    token = await auth_token(client, "acme-rhea")

    allowed = await chat(client, token, "Change my pickup tomorrow to 09:15")
    assert "09:15" in allowed["reply"]
    entry = await fetch_one(RosterEntry, id="RE-RH-PT")
    assert entry.scheduled_time == "09:15"

    rejected = await chat(client, token, "Move my drop to 20:00 please")
    assert "120 minutes" in rejected["reply"]


async def test_offline_insist_escalates_and_manager_resumes(client, app, offline_app):
    """The full demo story, keyless: reject -> insist (same thread, so the
    follow-up has the earlier time in context) -> manager approve."""
    token = await auth_token(client, "acme-rhea")
    first = await chat(client, token, "Move my drop to 20:00 please")
    assert "120 minutes" in first["reply"]

    escalated = await chat(client, token, "I know, but I really need it. Please escalate.",
                           thread_id=first["thread_id"])
    assert escalated["status"] == "pending_approval"
    assert escalated["request_id"]

    from tests.conftest import auth_header
    mgr = await auth_token(client, "acme-mgr")
    resp = await client.post(
        f"/manager/requests/{escalated['request_id']}/decision",
        json={"decision": "approve", "note": "offline demo"},
        headers=auth_header(mgr),
    )
    assert resp.status_code == 200

    req = await fetch_one(ShiftRequest, id=escalated["request_id"])
    assert req.status == "applied"
    entry = await fetch_one(RosterEntry, id="RE-RH-D")
    assert entry.scheduled_time == "20:00"
