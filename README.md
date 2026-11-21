# Shift-Change Request Agent — MoveInSync Case Study A

A chat agent that owns the shift-change conversation end to end for employee
transport: it reads only that employee's roster, checks cab options, applies
**that company's rules exactly** (in code, not in the prompt), escalates
rule-breaking requests to a human transport manager, pauses, and resumes when
the decision arrives — even if the service restarted in between.

Two client companies share the one service, with different rules:

| Company | Cut-off before trip | Cab size | Changes/day | Operating window |
|---|---|---|---|---|
| Acme Manufacturing | **120 min** | 4 seats | 1 | 06:00–22:00 |
| Globex Software | **30 min** | 6 seats | 2 | 06:30–21:30 |

## Run it

```bash
docker compose up
```

That's it. The app (FastAPI) and Postgres come up, schema is created and the
two sample companies are seeded automatically. API on http://localhost:8000 .

For real LLM replies, copy `.env.example` to `.env` and put a free
[Groq](https://console.groq.com/keys) key in it first. Without a key the
service still boots and every chat gets a graceful *"assistant unavailable,
call the transport desk"* reply — the rules engine keeps working.

### One tour of the API (with a real or fake LLM)

```bash
python scripts/demo_flow.py            # runs the whole story, print by print
```

For a repeatable recording, reset the demo data first (the cut-off story
needs the just-seeded trips, and approved changes persist in Postgres):

```bash
docker compose exec app python scripts/reset_demo_data.py
```

Manual equivalents:

```bash
# 1. demo token stand-in for the client company's IdP (identity lives ONLY here)
curl -s localhost:8000/auth/token -H 'Content-Type: application/json' \
     -d '{"employee_id": "acme-rhea"}'

# 2. chat (employee never sends identity; the JWT carries company + employee)
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $TOKEN" -d '{"message": "Move my drop to 20:00"}'

# 3. manager queue + decision (resumes the paused conversation)
curl -s localhost:8000/manager/requests -H "Authorization: Bearer $MANAGER_TOKEN"
curl -s -X POST localhost:8000/manager/requests/$REQUEST_ID/decision \
     -H 'Content-Type: application/json' -H "Authorization: Bearer $MANAGER_TOKEN" \
     -d '{"decision": "approve", "note": "one-off exception"}'
```

Other endpoints: `GET /chat/{thread_id}/messages` (transcript), `GET /my/requests`.

### Tests and evaluation

```bash
# unit/behaviour suite: fake LLM, frozen clock, no keys needed (runs in ~6s)
docker compose exec app python -m pytest tests -q
#   ...or on the host:  .venv/Scripts/python -m pytest tests -q   (Windows)

# evaluation set -> evals/results.jsonl (one line per case: id, input,
# expected, actual, pass, notes)
docker compose exec app python -m evals.run_evals

# p95 latency against the running service (real LLM; budget 5s)
python evals/measure_latency.py http://127.0.0.1:8000 30
```

The evaluator's ten-minute check needs one command (`docker compose up`), the
test suite, and the eval runner — none of the three needs an API key.

### Postgres-specific proof (why the compose path matters)

sqlite quietly skips foreign keys and its single-writer behaviour hides some
races, so the Postgres path is verified separately, live: one container
escalates a request, a **brand-new container** (fresh process, fresh
checkpointer) resumes it with the manager's decision, and the booking lands
with correct seat counts and audit rows. The last-seat race is then re-run on
real Postgres row locks:

```bash
P1=$(docker compose run --rm -T -e LLM_PROVIDER=fake app python \
     scripts/verify_postgres_phase.py phase1 | grep '^PHASE1_RESULT' | sed 's/^PHASE1_RESULT //')
docker compose run --rm -T -e LLM_PROVIDER=fake app python \
     scripts/verify_postgres_phase.py phase2 "$P1"
# REQUEST_STATUS applied | ROSTER moved | SEATS 4/4 | RACE ['applied','full']
```

---

## Design decisions

### 1. The split: the model owns the conversation, the code owns the rules

The LLM does exactly two things: decide *which tool to call with which
arguments*, and word the reply. Everything that touches correctness is plain
Python:

* `app/policy.py` — cut-off, operating window, per-day limit, per company.
  No prompt text can pass a request through it; `insist=True` is the only
  path around a rejection and it produces a *manager approval request*, never
  a booking.
* `app/services/booking.py` — seat moves, idempotency, escalation. Approval
  decisions are applied by the same booking code path as normal changes
  (`approval.decide` → `booking.move_roster_entry`).
* Tool schemas have **no identity parameters**. The agent's tool node builds a
  `ToolContext` from the verified JWT (employee_id, company_id, thread, clock)
  and passes it to the raw implementations. The model literally cannot ask for
  another company's roster — the argument does not exist to be filled.

So a hallucinating, jailbroken or just-dumb model cannot overbook a cab,
quote the wrong company's rule, or approve itself. Worst case it word-salads;
the truth still holds.

### 2. Identity comes from the token, only from the token

`app/auth.py` verifies an HS256 JWT whose claims (sub, company_id, role) are
the *only* source of identity in the whole system. `/auth/token` is a demo
stand-in for the client company's IdP; in production MoveInSync would validate
the company's IdP tokens and nothing else would change. Chat request bodies
have no identity fields (`app/schemas.py`), tool schemas have none, and a
transcript is readable only by its owner (`conversations` table + 403s).

### 3. Pause/resume that survives restarts

The conversation is a LangGraph `StateGraph` with a **Postgres checkpointer**
(`AsyncPostgresSaver`). When a rule-breaking request is escalated, the graph
reaches the `approval_gate` node and calls `interrupt(...)` — the paused state
lands in Postgres inside the checkpoint, next to the committed
`pending_approval` request row. The manager's decision endpoint later calls
`graph.ainvoke(Command(resume=decision), config)`; the graph wakes at exactly
that node, applies the decision **in code**, and the agent words the
confirmation. `tests/test_approval.py::test_resume_survives_a_real_process_restart`
escalates in one OS process, boots a brand-new `uvicorn` process on the same
DB, has *that* process deliver the decision, and asserts the booking happened
and the employee's transcript contains the confirmation.

### 4. At most 4 LLM calls per message (and how)

* The agent node keeps `llm_calls` in graph state and refuses to call the
  model once the counter reaches `LLM_MAX_CALLS_PER_TURN=4` (`app/agent/graph.py`).
  A scripted test (`test_llm_call_budget_is_hard_capped_at_four`) drives the
  model into endless tool-calling and asserts exactly 4 model calls, then a
  deterministic fallback reply.
* Typical turns cost 1 call (direct answer / clarifying question) or 2 (tool
  round trip). Even clarify→insist→confirm stays inside the budget.
* History is trimmed to the last 12 messages before each call, which also
  keeps tokens (and latency) bounded.

### 5. Reply within 5 s at p95

* Default provider is **Groq** (`llama-3.1-8b-instant`): small, fast, free.
* `timeout=3.5s, max_retries=0` on the model call — a retry would blow the
  budget; instead the employee gets the fallback message immediately.
* Max output capped at 300 tokens; tools return compact JSON with a
  ready-to-relay `say` string, so the model has nothing to invent.
* Tools are local DB reads/writes (single-digit milliseconds).
* `evals/measure_latency.py` prints p50/p90/p95/p99 over N live calls.
  With the scripted fake model the whole pipeline (HTTP + graph + tools +
  checkpoint) measures **p95 ≈ 31 ms** on this machine, so virtually all of
  the 5 s budget is available to the LLM; the 3.5 s model timeout caps the
  worst case below 5 s with a graceful fallback on timeout. Run the script
  with a Groq key for the end-to-end number on your network.

### 6. Graceful failure everywhere

Tool exceptions are caught in the tool node and become a friendly
ToolMessage; LLM timeouts/outages become a friendly reply; unhandled errors
never reach the client (stack traces go to the server log only). Scenarios
proved by tests/evals: tool timeout (E16), LLM outage (test), 4-call budget
(E15), cab full while deciding (E08), no such slot → nearest options (E07),
request stuck in approval (E09). No chat path hangs silently.

### 6b. No key? A deterministic assistant takes over

With `LLM_PROVIDER=groq` but no key configured (the default `docker compose up`),
the service boots with an **offline fallback model** instead of crashing:
plain regex intent matching emits the *same tool calls* a real model would
(`app/llm.py::OfflineFallbackModel`), tool results are relayed verbatim, and
every rule, isolation guarantee, booking, escalation and pause/resume works
identically - only the natural phrasing is lost. That keeps the evaluator's
one-command check fully functional without any account, and it is the honest
fallback for a provider outage in production.

### 7. Bonus chosen: concurrent last-seat + double submit

* **Double submit**: a request is idempotent on (employee, roster entry, date,
  new time) via a unique `idempotency_key`; a repeat returns the existing
  request instead of booking again (E13, `test_double_submit_*`).
* **Last-seat race**: the seat grab is a single conditional UPDATE —
  `UPDATE cabs SET booked_seats = booked_seats + 1 WHERE id = ? AND
  booked_seats < capacity` — one statement, atomic on Postgres (row lock,
  re-evaluated at commit) and on sqlite (single writer). Two concurrent
  submissions get exactly one `applied` and one `full` (E14,
  `test_two_employees_one_last_seat_exactly_one_wins`), and the loser's
  message offers the nearest slots.

### 8. Audit trail

Every action — request, rejection (with rule codes), escalation, manager
decision, booking, failure — lands in `audit_log` with actor, company and
timestamp. "Who asked for what and who decided" is a query, not an argument.

## Trade-offs (and what I'd do next)

1. **Postgres for everything (data + checkpointer) instead of Redis + a
   second store.** One service, one consistency story, and the checkpoint
   semantics the pause/resume needs are first-class. Redis would buy
   lower checkpoint latency we don't need at this scale.
2. **LangGraph `interrupt()` for the pause rather than a custom polling
   state machine.** The interrupt payload, the thread state and the resume
   mechanics are the library's tested surface; my code owns only the
   decision application. Cost: the resume must go through the same compiled
   graph (a manager decision is applied in the manager endpoint's process,
   which is exactly what the restart test exercises).
3. **Relay-don't-improvise phrasing.** Tools return employee-ready sentences
   (`say`) the prompt tells the model to use. Slightly repetitive prose buys
   the guarantee that a rule is never misquoted. A production version would
   let the model paraphrase everything except rule text.
4. **Demo IdP inside the service.** Convenient for evaluation, wrong for
   production — real deployments should federate the client company's IdP
   (the rest of the system already treats the JWT as the only identity).
5. **A single conditional UPDATE as the seat guard instead of
   `SELECT ... FOR UPDATE` everywhere.** Fewer round trips, and it is
   correct on both Postgres and the sqlite test path. A hotspot cab could
   still use explicit row locks under very high contention; not needed at
   case-study scale.
6. **Bonus #1 (waitlist notification) was not built**; the schema has no
   waitlist table. It would slot in as: on seat release (old cab), notify the
   oldest waitlist row for that cab *within the same company* — the
   company scoping is already enforced everywhere else.
7. **Tests use a scripted fake LLM; only the demo needs a real one.** This
   keeps the suite deterministic and free (per the rules). The trade-off:
   the suite proves the *harness* handles any tool sequence, not that a
   given model chooses well. The eval set is deterministic for the same
   reason; the live latency script exercises the real provider.

## Known limitations

* No streaming (SSE/WebSocket): replies arrive as one JSON response. The 5 s
  budget still holds, but a production UX would stream tokens.
* `create_all` instead of migrations; fine for the case study, Alembic in
  real life.
* Employees have at most one pickup and one drop per day in the sample data;
  multiple same-day trips would need the tools to take a roster-entry id.
* Times are IST by design (this platform's domain); no multi-timezone
  clients assumed.
* The manager approves through the API; there is no manager UI.

## Repository map

```
app/
  main.py            FastAPI factory, lifespan (schema, seed, checkpointer)
  config.py clock.py timeutil.py       frozen-clock-able configuration
  auth.py            JWT issue/verify + FastAPI dependencies
  policy.py          company rules, in code
  schemas.py         request/response models (no identity fields!)
  models.py          SQLAlchemy: companies, employees, cabs, roster,
                     shift_requests, audit_log, conversations
  seed.py            two companies + rosters, seeded relative to "now"
  llm.py             Groq/OpenAI/Gemini factory + scripted FakeChatModel
  agent_service.py   chat orchestration, transcript, resume-with-decision
  agent/
    state.py prompts.py tools.py graph.py
  services/          roster, cabs, booking, approval, audit
  api/               auth, chat, manager, health routers
tests/               32 tests incl. real process-restart + seat race
evals/               cases.jsonl, run_evals.py, results.jsonl, measure_latency.py
scripts/             make_token.py, demo_flow.py
docker-compose.yml   one command
docs/EVALUATION.md   method, results, known failures
```
