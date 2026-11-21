"""Evaluation runner.

Runs every case in evals/cases.jsonl against the real application (in-process
HTTP), with the scripted fake LLM and a frozen clock, so results are
deterministic and comparable across submissions and runs. Writes
evals/results.jsonl with exactly the required fields:

    id, input, expected, actual, pass, notes

No API key and no docker needed for the standard run:
    python -m evals.run_evals

(With a key in .env you can also boot the real service and run
evals/measure_latency.py against it for the p95 number - see README.)
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DATA_DIR = Path(tempfile.mkdtemp(prefix="sca_evals_"))
EVAL_DB = EVAL_DATA_DIR / "eval.db"

os.environ["FROZEN_NOW"] = "2026-01-15T17:00:00+05:30"
os.environ["LLM_PROVIDER"] = "fake"
os.environ["JWT_SECRET"] = "eval-secret-that-is-long-enough-32b"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{EVAL_DB.as_posix()}"

import httpx  # noqa: E402
from httpx import AsyncClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from app.agent.graph import BUDGET_MSG  # noqa: E402
from app.agent.tools import EXECUTED_TOOLS, clear_executed  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import REQUEST_APPLIED, Cab, RosterEntry, ShiftRequest  # noqa: E402


# ----------------------------------------------------------------- db probes

async def db_probes() -> dict:
    async def one(model, **filters):
        from sqlalchemy import select

        async with SessionLocal() as session:
            stmt = select(model)
            for k, v in filters.items():
                stmt = stmt.where(getattr(model, k) == v)
            return (await session.execute(stmt)).scalars().first()

    async def count(model, **filters):
        from sqlalchemy import func, select

        async with SessionLocal() as session:
            stmt = select(func.count()).select_from(model)
            for k, v in filters.items():
                stmt = stmt.where(getattr(model, k) == v)
            return (await session.execute(stmt)).scalar_one()

    return {"one": one, "count": count}


# --------------------------------------------------------------- http helpers

def header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def token_for(client: AsyncClient, employee_id: str) -> str:
    resp = await client.post("/auth/token", json={"employee_id": employee_id})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


async def chat(client: AsyncClient, token: str, message: str, thread_id: str | None) -> dict:
    resp = await client.post(
        "/chat", json={"message": message, "thread_id": thread_id}, headers=header(token)
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def script_llm(fake_llm, script: list[dict]) -> None:
    """Turn a case script into model responses, in order."""
    for step in script:
        if "tool" in step:
            fake_llm.script_tool_call(step["tool"], step.get("args", {}))
        elif "say" in step:
            fake_llm.script(AIMessage(content=step["say"]))


# ------------------------------------------------------------------ checking

async def evaluate(expect: dict, observed: dict, probes) -> tuple[bool, dict, list[str]]:
    checks: dict[str, bool] = {}
    notes: list[str] = []
    reply = observed.get("reply", "")
    reply_l = reply.lower()

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = ok
        if not ok:
            notes.append(f"{name}: {detail}")

    if "reply_contains_all" in expect:
        record(
            "reply_contains_all",
            all(w.lower() in reply_l for w in expect["reply_contains_all"]),
            reply[:160],
        )
    if "last_reply_contains_all" in expect:
        record(
            "last_reply_contains_all",
            all(w.lower() in reply_l for w in expect["last_reply_contains_all"]),
            reply[:160],
        )
    if "reply_not_contains" in expect:
        record(
            "reply_not_contains",
            all(w.lower() not in reply_l for w in expect["reply_not_contains"]),
            reply[:160],
        )
    if "reply_equals_budget_message" in expect:
        record("reply_equals_budget_message", reply == BUDGET_MSG, reply[:160])
    if "executed_tools" in expect:
        record("executed_tools", observed["executed"] == expect["executed_tools"],
               str(observed["executed"]))
    if "executed_tools_last" in expect:
        record("executed_tools_last", observed["executed_last"] == expect["executed_tools_last"],
               str(observed["executed_last"]))
    if "executed_tools_contains" in expect:
        record(
            "executed_tools_contains",
            all(t in observed["executed"] for t in expect["executed_tools_contains"]),
            str(observed["executed"]),
        )
    if "llm_calls" in expect:
        record("llm_calls", observed["llm_calls"] == expect["llm_calls"],
               str(observed["llm_calls"]))

    if "requests_for_employee" in expect:
        n = await _count_requests(probes, observed["employee"])
        record("requests_for_employee", n == expect["requests_for_employee"], str(n))
    if "applied_requests_total" in expect:
        n = await _count_applied_total(probes)
        record("applied_requests_total", n == expect["applied_requests_total"], str(n))
    if "requests_status" in expect or "latest_request_status" in expect:
        want = expect.get("requests_status") or expect.get("latest_request_status")
        status = await _latest_status(probes, observed["employee"])
        record("latest_request_status", status == want, f"{status} != {want}")
    if "roster_time" in expect:
        for entry_id, want in expect["roster_time"].items():
            entry = await _entry(probes, entry_id)
            got = entry.scheduled_time if entry else None
            record(f"roster_time[{entry_id}]", got == want, f"{got} != {want}")
    if "cab_booked" in expect:
        for cab_id, want in expect["cab_booked"].items():
            got = await _booked(probes, cab_id)
            record(f"cab_booked[{cab_id}]", got == want, f"{got} != {want}")
    if "exactly_one_applied" in expect:
        replies = observed.get("race_replies", [])
        winners = [r for r in replies if "Done" in r]
        losers = [r for r in replies if "filled up" in r]
        record(
            "exactly_one_applied",
            len(winners) == 1 and len(losers) == 1,
            str(replies),
        )
    if "decision_status_code" in expect:
        record("decision_status_code",
               observed.get("decision_status") == expect["decision_status_code"],
               str(observed.get("decision_status")))
    if "transcript_contains_all" in expect:
        texts = " ".join(m["content"] for m in observed.get("transcript", [])).lower()
        record("transcript_contains_all",
               all(w.lower() in texts for w in expect["transcript_contains_all"]),
               texts[:160])

    return all(checks.values()), checks, notes


async def _count_requests(probes, employee_id) -> int:
    return await probes["count"](ShiftRequest, employee_id=employee_id)


async def _count_applied_total(probes) -> int:
    return await probes["count"](ShiftRequest, status=REQUEST_APPLIED)


async def _latest_status(probes, employee_id) -> str | None:
    from sqlalchemy import select

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ShiftRequest)
                .where(ShiftRequest.employee_id == employee_id)
                .order_by(ShiftRequest.created_at.desc())
            )
        ).scalars().all()
        return rows[0].status if rows else None


async def _entry(probes, entry_id):
    return await probes["one"](RosterEntry, id=entry_id)


async def _booked(probes, cab_id) -> int | None:
    cab = await probes["one"](Cab, id=cab_id)
    return cab.booked_seats if cab else None


# ------------------------------------------------------------------ case run

async def run_case(case: dict, client: AsyncClient, app) -> dict:
    fake_llm = app.state.chat_service.llm
    observed: dict = {
        "employee": case.get("employee") or case.get("participants", [{}])[0].get("employee"),
        "executed": [],
        "executed_last": [],
        "replies": [],
        "race_statuses": [],
        "decision_status": None,
        "transcript": [],
        "llm_calls": 0,
    }

    kind = case.get("kind", "chat")

    if kind == "race":
        # Two employees ask for the same last seat at the same moment. Run at
        # the booking-engine level: the invariant "exactly one seat given" is
        # a property of the booking code under concurrency, not of the LLM,
        # and scripting two concurrent chats through one FIFO queue would be
        # nondeterministic by construction.
        from app.clock import get_now
        from app.services.booking import apply_change

        async def _book(employee_id: str) -> dict:
            async with SessionLocal() as session:
                return await apply_change(
                    session,
                    employee_id=employee_id,
                    company_id="acme",
                    trip_date=get_now().strftime("%Y-%m-%d"),
                    trip_type="drop",
                    new_time="20:00",
                    thread_id=f"race-{employee_id}",
                    now=get_now(),
                )

        results = await asyncio.gather(_book("acme-dev"), _book("acme-tara"))
        observed["race_statuses"] = [r["status"] for r in results]
        observed["race_replies"] = [r["message"] for r in results]
        observed["replies"] = [r["message"] for r in results]
        observed["reply"] = " | ".join(r["message"] for r in results)
        return observed

    token = await token_for(client, case["employee"])
    thread_id = None
    executed_before_case = len(EXECUTED_TOOLS)

    for i, turn in enumerate(case.get("turns", [])):
        script_llm(fake_llm, turn.get("script", []))
        turn_executed_before = len(EXECUTED_TOOLS)
        result = await chat(client, token, turn["input"], thread_id)
        thread_id = result["thread_id"]
        observed["replies"].append(result["reply"])
        observed["reply"] = result["reply"]
        observed["executed"] = list(EXECUTED_TOOLS[executed_before_case:])
        observed["executed_last"] = list(EXECUTED_TOOLS[turn_executed_before:])
        observed["status"] = result["status"]
        observed["request_id"] = result.get("request_id")

    observed["llm_calls"] = len(fake_llm.calls)
    fake_llm.calls.clear()

    manager_decision = None
    if case.get("manager") and case.get("kind") == "chat_manager_cross":
        mgr_token = await token_for(client, case["manager"])
        resp = await client.post(
            f"/manager/requests/{observed.get('request_id')}/decision",
            json={"decision": "approve"},
            headers=header(mgr_token),
        )
        observed["decision_status"] = resp.status_code
        return observed

    if case.get("manager") and case.get("decision"):
        manager_decision = {"manager": case["manager"], "decision": case["decision"]}

    if kind == "restart" and manager_decision:
        # Fresh OS process resumes the paused conversation from the shared DB.
        port = _free_port()
        env = os.environ.copy()
        env["DATABASE_URL"] = f"sqlite+aiosqlite:///{EVAL_DB.as_posix()}"
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port),
             "--log-level", "warning"],
            cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(60):
                try:
                    if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                        break
                except Exception:
                    time.sleep(0.3)
            mgr_tok = httpx.post(f"{base}/auth/token",
                                 json={"employee_id": manager_decision["manager"]}, timeout=5
                                 ).json()["access_token"]
            resp = httpx.post(
                f"{base}/manager/requests/{observed['request_id']}/decision",
                json={"decision": manager_decision["decision"], "note": "eval"},
                headers=header(mgr_tok), timeout=60,
            )
            observed["decision_status"] = resp.status_code
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    elif manager_decision:
        mgr_token = await token_for(client, manager_decision["manager"])
        resp = await client.post(
            f"/manager/requests/{observed['request_id']}/decision",
            json={"decision": manager_decision["decision"], "note": "eval"},
            headers=header(mgr_token),
        )
        observed["decision_status"] = resp.status_code

    if "transcript_contains_all" in case.get("expect", {}):
        resp = await client.get(f"/chat/{thread_id}/messages", headers=header(token))
        observed["transcript"] = resp.json() if resp.status_code == 200 else []

    return observed


async def reset_db() -> None:
    """Fresh business data per case, like the pytest suite does."""
    from app.models import (
        AuditLog,
        Cab,
        Company,
        Conversation,
        Employee,
        RosterEntry,
        ShiftRequest,
    )
    from app.seed import seed_if_empty

    async with SessionLocal.begin() as session:
        for table in (ShiftRequest, AuditLog, Conversation, RosterEntry, Cab, Employee, Company):
            await session.execute(table.__table__.delete())
    await seed_if_empty()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------- main

async def main_async() -> int:
    cases = [
        json.loads(line)
        for line in (ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval") as client:
            results = []
            for case in cases:
                clear_executed()
                await reset_db()
                app.state.chat_service.llm.calls.clear()
                # tool-failure injection
                roster_service = None
                if case.get("fail_roster"):
                    from app.services import roster as roster_service

                    async def boom(*args, **kwargs):
                        raise TimeoutError("upstream timeout")

                    original = roster_service.get_roster
                    roster_service.get_roster = boom
                try:
                    observed = await run_case(case, client, app)
                finally:
                    if case.get("fail_roster") and roster_service:
                        roster_service.get_roster = original

                probes = await db_probes()
                passed, checks, notes = await evaluate(case.get("expect", {}), observed, probes)
                results.append(
                    {
                        "id": case["id"],
                        "input": {
                            "employee": observed["employee"],
                            "kind": case.get("kind", "chat"),
                            "messages": [t["input"] for t in case.get("turns", [])]
                            or [p["input"] for p in case.get("participants", [])],
                        },
                        "expected": case.get("expect", {}),
                        "actual": {
                            "reply": observed.get("reply", ""),
                            "checks": checks,
                            "executed_tools": observed.get("executed", []),
                            "status": observed.get("status"),
                        },
                        "pass": passed,
                        "notes": "; ".join(notes) if notes else "all checks passed",
                    }
                )
                mark = "PASS" if passed else "FAIL"
                print(f"[{mark}] {case['id']}" + ("" if passed else f" - {notes}"))

    out = ROOT / "evals" / "results.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{len(results)} cases passed. Results written to {out}")
    return 0 if passed == len(results) else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
