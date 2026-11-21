"""Manager approval path: escalate -> pause -> decide -> resume."""
from sqlalchemy import select

from app.db import SessionLocal
from app.models import AuditLog, Cab, RosterEntry, ShiftRequest
from tests.conftest import auth_header, auth_token, chat, fetch_one


async def _escalate(client, fake_llm, employee_id: str, message_time: str = "20:00") -> dict:
    """Two turns: first request is refused by the rule engine, the insist turn
    creates the manager-approval request and pauses the conversation."""
    token = await auth_token(client, employee_id)
    fake_llm.script_tool_call("request_shift_change", {"new_time": message_time, "trip_type": "drop"})
    first = await chat(client, token, f"Move my drop to {message_time}")

    fake_llm.script_tool_call(
        "request_shift_change",
        {"new_time": message_time, "trip_type": "drop", "insist": True, "reason": "meeting ran over"},
    )
    second = await chat(client, token, "I know it breaks the rule but I really need it", thread_id=first["thread_id"])
    assert second["status"] == "pending_approval"
    assert second["request_id"]
    return {"token": token, "thread_id": first["thread_id"], "request_id": second["request_id"]}


async def test_escalation_pauses_and_persists(client, fake_llm):
    ctx = await _escalate(client, fake_llm, "acme-rhea")
    req = await fetch_one(ShiftRequest, id=ctx["request_id"])
    assert req.status == "pending_approval"
    assert req.requested_time == "20:00"
    assert req.thread_id == ctx["thread_id"]
    async with SessionLocal() as session:
        audits = (
            await session.execute(select(AuditLog).where(AuditLog.action == "escalated_to_manager"))
        ).scalars().all()
        assert len(audits) == 1
    # nothing booked yet
    entry = await fetch_one(RosterEntry, id="RE-RH-D")
    assert entry.scheduled_time == "18:30"


async def test_manager_sees_pending_and_approval_books_it(client, fake_llm):
    ctx = await _escalate(client, fake_llm, "acme-rhea")

    mgr = await auth_token(client, "acme-mgr")
    resp = await client.get("/manager/requests", headers=auth_header(mgr))
    assert resp.status_code == 200
    pending = [r for r in resp.json() if r["id"] == ctx["request_id"]]
    assert len(pending) == 1

    resp = await client.post(
        f"/manager/requests/{ctx['request_id']}/decision",
        json={"decision": "approve", "note": "one-off exception"},
        headers=auth_header(mgr),
    )
    assert resp.status_code == 200

    req = await fetch_one(ShiftRequest, id=ctx["request_id"])
    assert req.status == "applied"
    assert req.decision_by == "acme-mgr"
    assert req.decision_note == "one-off exception"

    # the booking actually happened
    entry = await fetch_one(RosterEntry, id="RE-RH-D")
    assert entry.scheduled_time == "20:00"
    assert entry.cab_id == "ACME-D2"
    assert (await fetch_one(Cab, id="ACME-D2")).booked_seats == 4
    assert (await fetch_one(Cab, id="ACME-D1")).booked_seats == 1

    # and the employee's conversation received the confirmation
    resp = await client.get(
        f"/chat/{ctx['thread_id']}/messages", headers=auth_header(ctx["token"])
    )
    texts = " ".join(m["content"] for m in resp.json())
    assert "approved" in texts.lower()


async def test_manager_rejection_leaves_roster_unchanged(client, fake_llm):
    ctx = await _escalate(client, fake_llm, "globex-arjun", message_time="20:00")

    mgr = await auth_token(client, "globex-mgr")
    resp = await client.post(
        f"/manager/requests/{ctx['request_id']}/decision",
        json={"decision": "reject", "note": "no exceptions during peak"},
        headers=auth_header(mgr),
    )
    assert resp.status_code == 200

    req = await fetch_one(ShiftRequest, id=ctx["request_id"])
    assert req.status == "rejected"
    entry = await fetch_one(RosterEntry, id="RE-AR-D")
    assert entry.scheduled_time != "20:00"  # unchanged

    resp = await client.get(
        f"/chat/{ctx['thread_id']}/messages", headers=auth_header(ctx["token"])
    )
    texts = " ".join(m["content"] for m in resp.json())
    assert "declined" in texts.lower()


async def test_decision_is_idempotent(client, fake_llm):
    ctx = await _escalate(client, fake_llm, "acme-rhea")
    mgr = await auth_token(client, "acme-mgr")
    resp = await client.post(
        f"/manager/requests/{ctx['request_id']}/decision",
        json={"decision": "approve"},
        headers=auth_header(mgr),
    )
    assert resp.status_code == 200
    resp = await client.post(
        f"/manager/requests/{ctx['request_id']}/decision",
        json={"decision": "reject"},
        headers=auth_header(mgr),
    )
    assert resp.status_code == 200
    req = await fetch_one(ShiftRequest, id=ctx["request_id"])
    assert req.status == "applied"  # second decision was a no-op


async def test_resume_survives_a_real_process_restart(client, fake_llm):
    """The strongest form of the requirement: escalate in THIS process, then a
    brand-new OS process (fresh interpreter, fresh checkpointer) resumes the
    paused conversation and the approval books the seat."""
    import os
    import subprocess
    import sys
    import time

    import httpx

    from tests.conftest import TEST_DB

    ctx = await _escalate(client, fake_llm, "acme-rhea")

    # Bind an ephemeral free port so leftover listeners can't be hit.
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB.as_posix()}"
    env["LLM_PROVIDER"] = "fake"
    env["FROZEN_NOW"] = "2026-01-15T17:00:00+05:30"
    env["JWT_SECRET"] = "test-secret-that-is-long-enough-32bytes"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 30
        ready = False
        while time.time() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.3)
        assert ready, "subprocess did not come up"

        # The restarted process issues its own token from the shared DB.
        tok_resp = httpx.post(f"{base}/auth/token", json={"employee_id": "acme-mgr"}, timeout=5)
        assert tok_resp.status_code == 200, tok_resp.text
        tok = tok_resp.json()["access_token"]

        # Manager decides on the OTHER (original) process's paused conversation.
        resp = httpx.post(
            f"{base}/manager/requests/{ctx['request_id']}/decision",
            json={"decision": "approve", "note": "approved after restart"},
            headers={"Authorization": f"Bearer {tok}"},
            timeout=30,
        )
        assert resp.status_code == 200, resp.text

        req = await fetch_one(ShiftRequest, id=ctx["request_id"])
        assert req.status == "applied"
        entry = await fetch_one(RosterEntry, id="RE-RH-D")
        assert entry.scheduled_time == "20:00"

        # confirmation landed in the conversation, readable from this process
        resp2 = await client.get(
            f"/chat/{ctx['thread_id']}/messages", headers=auth_header(ctx["token"])
        )
        texts = " ".join(m["content"] for m in resp2.json())
        assert "approved" in texts.lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
