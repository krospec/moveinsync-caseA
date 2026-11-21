# Evaluation Report — Shift-Change Request Agent

## Method

Two deterministic layers, both runnable without an LLM key, plus one live
layer:

1. **Automated tests** (`pytest tests`, 32 tests, ~6 s): scripted fake LLM,
   frozen clock (`APP` runs on `2026-01-15 17:00 IST` so cut-off rules are
   time-independent), file-backed sqlite. Proves behaviour: cross-company
   isolation, rule enforcement, pause/resume (including a real second OS
   process), tool routing, graceful failure, budget cap, seat-race and
   double-submit idempotency, JWT handling.
2. **Evaluation set** (`python -m evals.run_evals`, 18 cases →
   `evals/results.jsonl`): each case drives the real HTTP API in-process with
   a scripted model and asserts on the reply text **and** the database state
   (roster times, seat counts, request statuses, audit rows). Output lines
   have exactly the required fields: `id`, `input`, `expected`, `actual`,
   `pass`, `notes`. Deterministic by construction: same input → same result,
   so submissions/runs compare side by side.
3. **Live latency** (`evals/measure_latency.py`): against the running server
   with a real provider; reports p50/p90/p95/p99. Needs a free Groq key in
   `.env`.

**Why a fake LLM in tests/evals:** the case study allows it, and it removes
the only nondeterministic component. The suite proves the *system* handles
any tool sequence, rule outcome, failure and restart correctly; it cannot
prove that a particular model picks good tools — that is what the live demo
and the real provider's behaviour are for. Quota errors can never cost a
passing test.

## Results

### Tests: 32/32 passed

Key scenarios (all green):

* `test_isolation.py` — tool schemas carry no identity parameters; roster
  reads scoped by employee+company even when a caller controls every
  argument; a chat message claiming a false identity gets the *token*
  identity's data; a manager of company A gets 403 on company B's request and
  cannot see it in their queue; transcripts are 403 to non-owners.
* `test_rules.py` — Acme 120-min cut-off rejects Rhea's 90-min-out drop (no
  booking, audit row written); Globex's 30-min cut-off allows the analogous
  request; operating window enforced; daily limit enforced on the second
  change; compliant changes move roster + seats correctly.
* `test_approval.py` — escalation persists `pending_approval`; manager
  approves → booking applied, seats moved, confirmation in transcript;
  reject → roster untouched; decisions are idempotent; **a brand-new OS
  process resumes the paused conversation and books the seat**.
* `test_agent_behavior.py` — correct tool routing (`EXECUTED_TOOLS` proof),
  ambiguous request → one clarifying question with zero side effects, roster
  tool timeout → friendly reply (no traceback, no hang), hard 4-call budget
  (exactly 4 model calls then fallback), LLM outage → friendly reply,
  stuck-in-approval status query answered from the status tool.
* `test_concurrency.py` — double submit → one request, one seat; concurrent
  last-seat submissions → exactly one `applied`, occupancy 4/4, loser's cab
  untouched.
* `test_jwt.py` — missing/garbage/expired/wrong-secret tokens → 401; manager
  role gate.

### Evaluation set: 18/18 passed

Latest run (`evals/results.jsonl`, regenerate with `python -m evals.run_evals`):

| Case | Proves |
|---|---|
| E01_isolation_roster | Rhea's chat cannot reach Globex data even when she asks for it by name |
| E02_isolation_manager_cross_company | Globex's manager cannot see/decide Acme's request (403, still pending) |
| E03_cutoff_rejection_acme | 120-min cut-off refusal, no booking, audit row |
| E04_cutoff_allowed_globex | same request shape allowed under Globex's 30-min rule; seats moved |
| E05_clarify_ambiguous | "Can I go a bit late?" → question, zero side effects |
| E06_ask_when_trip_unclear | tool refuses to guess pickup vs drop |
| E07_no_slot_offers_nearest | no 21:45 cab → nearest free slots offered |
| E08_cab_filled_up | full cab → "just filled up" + alternatives, no booking |
| E09_pending_status_query | stuck-in-approval → honest status from the status tool |
| E10_approve_books_and_confirms | manager approval books seats; transcript carries confirmation |
| E11_reject_leaves_unchanged | rejection leaves roster untouched; transcript says declined |
| E12_resume_after_restart | **second OS process** resumes the paused conversation; booking applied |
| E13_double_submit_idempotent | same change twice → one request, one seat |
| E14_last_seat_race | concurrent submissions for the last seat → exactly one applied (occupancy 4/4) |
| E15_llm_budget_cap | runaway model → exactly 4 LLM calls, deterministic fallback reply |
| E16_tool_failure_friendly | tool timeout → plain words, no traceback |
| E17_tool_selection | roster question → roster tool; options question → options tool |
| E18_policy_explained_from_tool | rules quoted from the policy tool, not invented |

### Latency

With the scripted fake model (isolates the platform overhead — HTTP, graph,
tools, checkpointing): `n=30, mean 29 ms, p95 31 ms` on the dev machine.
The full 5 s p95 budget is therefore available to the LLM call. Production
settings keep the worst case under budget: model timeout 3.5 s, no retries,
300-token cap, 12-message history window; on timeout the employee gets the
fallback message instead of a hang. Run `python evals/measure_latency.py
http://127.0.0.1:8000 30` with a Groq key for the end-to-end figure.

## Known failures / limitations

* No streaming UX (single JSON reply per message).
* The eval set cannot judge the real model's tool-*choice* quality —
  deterministic scripting means routing quality is asserted via
  `EXECUTED_TOOLS` only for scripted choices. (The trade-off is deliberate:
  free, reproducible, quota-proof. The live demo covers model quality.)
* Multiple same-day trips of one type would need roster-entry-id arguments in
  the tools; sample data has at most one pickup + one drop per employee-day.
* Bonus #1 (waitlist notification) was not implemented (bonus #2 was).
* The restart eval/test proves resume across two processes on sqlite; the
  same path was additionally verified **live on Postgres** with
  `scripts/verify_postgres_phase.py` (two `docker compose run` containers:
  phase 1 escalates, phase 2 — a fresh process — resumes and books;
  final state `REQUEST_STATUS applied`, seats 4/4, race `['applied','full']`).
  It is a manual proof, not wired into CI, because the automated suites run
  infra-free by design.
* Live compose verification surfaced (and fixed) three Postgres-only bugs
  that sqlite could not catch: unit-of-work insert ordering without
  relationships (FK violation mid-seed), naive-vs-aware timestamps under
  asyncpg, and FK-aware delete ordering in test/eval resets. A good reminder
  that the engine the case study asks for (Postgres) needed its own pass.

## The trade-off I'd defend in a viva

**Relay-don't-improvise.** Tools return employee-ready sentences and the
prompt instructs the model to use them. The prose is a bit repetitive, but a
rule can never be misquoted to the wrong company's employee — the exact
failure mode the transport desk suffers today. Every creative-sounding part
of the system is a sentence generator fed by facts the code verified.
