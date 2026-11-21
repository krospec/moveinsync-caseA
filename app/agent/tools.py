"""Agent tools.

Two halves, on purpose:

* STUBS - decorated with @tool, used only for their schema when we bind tools
  to the model. They contain no identity parameters: the model literally cannot
  ask for another employee's roster because no such argument exists.
* RAW implementations - plain async functions called by our own tool node with
  a ToolContext the graph builds from server-side state (JWT identity, thread,
  frozen-or-real clock). This is where "authorisation lives outside the model"
  is enforced, mechanically.

Every raw tool returns a dict with a `say` field: employee-ready wording the
model is instructed to relay. The model narrates; it never decides.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

from ..db import SessionLocal
from ..models import REQUEST_PENDING, Company
from ..services import booking, cabs, roster
from ..timeutil import shift_date


@dataclass(frozen=True)
class ToolContext:
    employee_id: str
    company_id: str
    employee_name: str
    thread_id: str
    today: str
    now: datetime


class ToolError(Exception):
    """Raised when the underlying service fails; converted to a friendly
    ToolMessage by the tool node. Never leaks a stack trace to the chat."""


TOOL_DOWN_SAY = (
    "That service isn't responding right now. Please try again in a minute - if it's "
    "urgent, call the transport desk."
)

RESOLVE_TRIP_SAY = (
    "I need one detail: do you want to change your pickup or your drop, and for which "
    "time? Tell me the new time and I'll take it from there."
)


# --------------------------------------------------------------------------
# Stubs: schemas the model sees (no identity parameters - that's the point)
# --------------------------------------------------------------------------

@tool
def get_my_roster(trip_type: Optional[str] = None, date: Optional[str] = None) -> str:
    """Show the employee's own roster entries. Optionally filter by trip_type
    ('pickup' or 'drop') and date (YYYY-MM-DD; today if omitted)."""
    raise NotImplementedError  # executed via the raw registry below


@tool
def find_cab_options(new_time: str, trip_type: Optional[str] = None, date: Optional[str] = None) -> str:
    """List this company's cabs with a free seat closest to new_time (HH:MM)
    for the given trip_type and date (today if omitted)."""
    raise NotImplementedError


@tool
def request_shift_change(
    new_time: str,
    trip_type: Optional[str] = None,
    date: Optional[str] = None,
    insist: bool = False,
    reason: Optional[str] = None,
) -> str:
    """Request moving the employee's trip to new_time (HH:MM). Set insist=True
    ONLY when the employee explicitly wants a change that was just refused by a
    company rule, so it goes to the transport manager. reason is a short note."""
    raise NotImplementedError


@tool
def check_request_status(request_id: Optional[str] = None) -> str:
    """Status of a shift-change request (or the latest one when request_id is
    omitted) - e.g. waiting for manager approval, applied, rejected."""
    raise NotImplementedError


@tool
def get_company_policy() -> str:
    """The employee's company transport rules: cut-off minutes, changes per
    day, operating window. Use it to answer questions about what is allowed."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Raw implementations (identity comes from ToolContext, never from the model)
# --------------------------------------------------------------------------

async def _roster_raw(ctx: ToolContext, trip_type=None, date=None) -> dict:
    async with SessionLocal() as session:
        entries = await roster.get_roster(
            session,
            employee_id=ctx.employee_id,
            company_id=ctx.company_id,
            trip_date=date or ctx.today,
            trip_type=trip_type,
        )
    if not entries:
        say = f"No roster entries found for {date or ctx.today}."
    else:
        lines = [
            f"{e['date']} {e['trip_type']} at {e['scheduled_time']}, cab {e['cab_id']} "
            f"({e['route']}, {e['seats_free']} seats free)"
            for e in entries
        ]
        say = "Here is your roster: " + "; ".join(lines) + "."
    return {"ok": True, "say": say, "entries": entries}


async def _options_raw(ctx: ToolContext, new_time: str, trip_type=None, date=None) -> dict:
    async with SessionLocal() as session:
        options = await cabs.find_options(
            session,
            company_id=ctx.company_id,
            trip_date=date or ctx.today,
            trip_type=trip_type or "drop",
            target_time=new_time,
        )
    say = (
        f"Nearest cabs with free seats around {new_time}: "
        + "; ".join(f"{o['departure_time']} ({o['seats_free']} free, {o['route']})" for o in options)
        + "."
        if options
        else f"No {trip_type or 'drop'} cabs with free seats near {new_time} on {date or ctx.today}."
    )
    return {"ok": True, "say": say, "options": options}


async def _request_change_raw(
    ctx: ToolContext,
    new_time: str,
    trip_type=None,
    date=None,
    insist=False,
    reason=None,
) -> dict:
    trip_date = date or ctx.today
    trip_type = (trip_type or "").lower() or None

    async with SessionLocal() as session:
        # Disambiguate the trip when the model didn't say pickup vs drop:
        # exactly one entry -> use it; several -> ask, never guess.
        if trip_type not in ("pickup", "drop"):
            entries = await roster.get_roster(
                session, employee_id=ctx.employee_id, company_id=ctx.company_id,
                trip_date=trip_date,
            )
            if len(entries) == 1:
                trip_type = entries[0]["trip_type"]
            else:
                return {
                    "ok": False,
                    "status": "need_trip_type",
                    "say": RESOLVE_TRIP_SAY,
                }

        result = await booking.apply_change(
            session,
            employee_id=ctx.employee_id,
            company_id=ctx.company_id,
            trip_date=trip_date,
            trip_type=trip_type,
            new_time=new_time,
            thread_id=ctx.thread_id,
            reason=reason,
            insist=bool(insist),
            now=ctx.now,
        )
        # Normalise service outcomes into tool-speak.
        status = result["status"]
        if status in ("rejected_by_rule", "full", "no_such_slot", "not_found",
                      "invalid_time", "no_change", "already_done"):
            return {
                "ok": status in ("already_done", "no_change"),
                "status": status,
                "say": result["message"],
                "checks": result.get("checks"),
                "options": result.get("options"),
                "request_id": result.get("request_id"),
            }
        return {
            "ok": True,
            "status": status,  # applied | escalated
            "say": result["message"],
            "request_id": result.get("request_id"),
        }


async def _request_status_raw(ctx: ToolContext, request_id=None) -> dict:
    async with SessionLocal() as session:
        req = (
            await booking.get_request(session, request_id)
            if request_id
            else await booking.latest_request_for(session, ctx.employee_id, ctx.company_id)
        )
        if req is None or (req.employee_id != ctx.employee_id or req.company_id != ctx.company_id):
            return {"ok": True, "say": "You have no shift-change requests yet."}
        status_words = {
            REQUEST_PENDING: "waiting for your transport manager's approval",
            "applied": "applied - your schedule has been updated",
            "rejected": "declined by your transport manager",
            "failed": "approved, but the seat was gone - schedule unchanged",
            "cancelled": "cancelled",
        }
        say = (
            f"Your {req.trip_type} change to {req.requested_time} on {req.trip_date} is "
            f"{status_words.get(req.status, req.status)}."
            + (
                " The decision will appear here as soon as they make it."
                if req.status == REQUEST_PENDING
                else ""
            )
        )
        return {
            "ok": True,
            "say": say,
            "request": {
                "id": req.id,
                "status": req.status,
                "requested_time": req.requested_time,
                "trip_date": req.trip_date,
                "trip_type": req.trip_type,
            },
        }


async def _policy_raw(ctx: ToolContext) -> dict:
    async with SessionLocal() as session:
        company = await session.get(Company, ctx.company_id)
    return {
        "ok": True,
        "say": (
            f"{company.name}: changes need {company.cutoff_minutes} minutes' notice, "
            f"max {company.max_changes_per_day} per day, cabs run {company.window_start}-{company.window_end}."
        ),
        "policy": {
            "cutoff_minutes": company.cutoff_minutes,
            "max_changes_per_day": company.max_changes_per_day,
            "window_start": company.window_start,
            "window_end": company.window_end,
        },
    }


RAW = {
    "get_my_roster": _roster_raw,
    "find_cab_options": _options_raw,
    "request_shift_change": _request_change_raw,
    "check_request_status": _request_status_raw,
    "get_company_policy": _policy_raw,
}

TOOL_SCHEMAS = [get_my_roster, find_cab_options, request_shift_change, check_request_status, get_company_policy]

# Tools actually executed by the graph (test/eval instrumentation - lets tests
# PROVE which tools ran, instead of claiming it in a README).
EXECUTED_TOOLS: list[str] = []


def clear_executed() -> None:
    EXECUTED_TOOLS.clear()


def resolve_date_alias(date: str | None, ctx: ToolContext) -> str | None:
    """Allow the model to pass 'tomorrow'/'today'; the prompt prefers YYYY-MM-DD
    but we don't trust it blindly."""
    if date in (None, "", "today"):
        return ctx.today
    if date == "tomorrow":
        return shift_date(ctx.today, 1)
    return date
