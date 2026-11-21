"""Demo flow for the 2-minute screen recording.

Usage:  python scripts/demo_flow.py [--step] [base_url]
        base_url defaults to http://127.0.0.1:8000 (docker compose up)
        --step pauses for Enter between steps, for narrating the recording.

Prints each step and the API's reply so the flow reads well on screen.
Works with a real LLM (GROQ_API_KEY in .env) or with the deterministic
offline fallback (no key) - the rules engine behaves identically.

For a clean recording with fresh roster data (the cut-off story needs the
just-seeded trips), reset first:

    docker compose exec app python scripts/reset_demo_data.py
"""
import json
import sys

import httpx

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
STEP = "--step" in sys.argv
BASE = _ARGS[0] if _ARGS else "http://127.0.0.1:8000"


def pause() -> None:
    if STEP:
        input("\n---- press Enter for the next step ----")


def show(title: str, payload: dict) -> None:
    pause()
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=60)

    print("MoveInSync Shift-Change Request Agent - demo")
    print("Health:", client.get("/health").json())

    # --- Employee (Acme); one conversation, threaded end to end ---
    rhea = client.post("/auth/token", json={"employee_id": "acme-rhea"}).json()["access_token"]
    rh = {"Authorization": f"Bearer {rhea}"}
    thread = None

    def employee_chat(message: str) -> dict:
        nonlocal thread
        result = client.post(
            "/chat", json={"message": message, "thread_id": thread}, headers=rh
        ).json()
        thread = result["thread_id"]
        return result

    show("1. Rhea asks about her roster",
         roster_reply := employee_chat("What time is my drop today?"))

    show("2. Ambiguous request -> clarifying question", employee_chat("Can I go a bit late today?"))

    # --- ask for options; pick a concrete bookable slot for the next steps ---
    options = employee_chat("Any cab around 12:00 for my drop?")
    show("2b. Cab options near 12:00", options)
    import re

    # avoid the slot the employee is already on (a repeated demo would hit
    # the idempotency guard instead of the approval story)
    roster = re.search(r"drop at (\d{1,2}:\d{2})", roster_reply["reply"].lower())
    current_time = roster.group(1) if roster else None
    pairs = re.findall(r"(\d{1,2}:\d{2}) \((\d+) free", options["reply"])
    target_time = next(
        (t for t, n in pairs if int(n) >= 2 and t != current_time), None
    )
    if not target_time:
        print("\nCould not parse a bookable slot from the options reply; using 20:00.")
        target_time = "20:00"

    # --- cut-off rejection (seed guarantees a trip inside the cut-off window) ---
    show(f"3. Change request to {target_time} breaks Acme's 2-hour cut-off",
         employee_chat(f"Move my drop to {target_time} please"))

    show("4. Employee insists -> escalated to the transport manager",
         employee_chat("I know, but I really need it. Please escalate."))

    manager_token_resp = client.post("/auth/token", json={"employee_id": "acme-mgr"}).json()
    mgr = {"Authorization": f"Bearer {manager_token_resp['access_token']}"}

    pending = client.get("/manager/requests", headers=mgr).json()
    show("5. Manager's pending queue", pending)
    if not pending:
        print("\nNo pending requests - the escalation in step 4 did not happen.")
        print("Check the app logs: docker compose logs app | tail -50")
        return
    request_id = pending[0]["id"]

    show("6. Manager approves; conversation resumes", client.post(
        f"/manager/requests/{request_id}/decision",
        json={"decision": "approve", "note": "one-off exception"}, headers=mgr).json())

    show("7. Employee sees the outcome in the same conversation",
         employee_chat("thanks, and what's my status?"))

    # --- second company: Globex, different rules (30-min cut-off) ---
    arjun = client.post("/auth/token", json={"employee_id": "globex-arjun"}).json()["access_token"]
    show("8. Globex employee - 30-min cut-off instead of 120", client.post(
        "/chat", json={"message": "Move my drop to 20:00 please"},
        headers={"Authorization": f"Bearer {arjun}"}).json())

    print("\nDone. The same API, two companies, two rule sets - one agent.")


if __name__ == "__main__":
    main()
