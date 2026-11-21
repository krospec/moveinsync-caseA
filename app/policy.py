"""Company rules, enforced in code - never in the prompt.

The LLM may *explain* a rejection, but the only thing that can stop or allow a
change is this module. Every check returns a RuleCheck with a code and an
employee-ready message, so the agent relays facts instead of inventing them.
"""
from dataclasses import dataclass
from datetime import datetime

from . import timeutil
from .models import Company, RosterEntry


@dataclass
class RuleCheck:
    code: str
    ok: bool
    message: str  # employee-facing wording; tools relay this verbatim


def check_cutoff(company: Company, entry: RosterEntry, now: datetime) -> RuleCheck:
    """The company cut-off rule: changes to a trip must arrive at least
    `cutoff_minutes` before that trip's scheduled departure."""
    minutes_left = timeutil.minutes_until(entry.trip_date, entry.scheduled_time, now)
    if minutes_left < company.cutoff_minutes:
        return RuleCheck(
            "CUTOFF_BREACH",
            False,
            (
                f"Your company requires changes at least {company.cutoff_minutes} minutes "
                f"before the scheduled time. Your {entry.trip_type} today is at "
                f"{entry.scheduled_time}, which leaves {int(max(minutes_left, 0))} minutes - "
                "too late for a self-service change."
            ),
        )
    return RuleCheck("CUTOFF_OK", True, "")


def check_window(company: Company, requested_time: str) -> RuleCheck:
    """New time must fall inside the company's transport operating window."""
    start_h, start_m = timeutil.parse_hhmm(company.window_start)
    end_h, end_m = timeutil.parse_hhmm(company.window_end)
    new_h, new_m = timeutil.parse_hhmm(requested_time)
    inside = (start_h, start_m) <= (new_h, new_m) <= (end_h, end_m)
    if not inside:
        return RuleCheck(
            "OUTSIDE_WINDOW",
            False,
            f"Cabs run between {company.window_start} and {company.window_end} for your "
            f"company, so {requested_time} cannot be booked.",
        )
    return RuleCheck("WINDOW_OK", True, "")


def check_daily_limit(company: Company, changes_today: int) -> RuleCheck:
    if changes_today >= company.max_changes_per_day:
        return RuleCheck(
            "DAILY_LIMIT",
            False,
            f"Your company allows at most {company.max_changes_per_day} roster change"
            f"{'s' if company.max_changes_per_day > 1 else ''} per day and you have already "
            "used them. Please contact your transport manager if this is urgent.",
        )
    return RuleCheck("DAILY_LIMIT_OK", True, "")


def validate_change(
    company: Company,
    entry: RosterEntry,
    requested_time: str,
    changes_today: int,
    now: datetime,
) -> list[RuleCheck]:
    """All the checks that gate a self-service change. Any not-ok check blocks
    the change; there is no prompt-level override."""
    checks = [
        check_cutoff(company, entry, now),
        check_window(company, requested_time),
        check_daily_limit(company, changes_today),
    ]
    return checks


def blocking(checks: list[RuleCheck]) -> list[RuleCheck]:
    return [c for c in checks if not c.ok]
