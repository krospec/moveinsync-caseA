"""Small time helpers. Trip times are stored as 'HH:MM' strings and dates as
'YYYY-MM-DD' strings (IST) so the data is portable across Postgres and sqlite.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .clock import get_now
from .config import get_settings


def app_tz() -> ZoneInfo:
    return ZoneInfo(get_settings().timezone)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.strip().split(":")
    return int(hour), int(minute)


def combine(date_str: str, hhmm: str) -> datetime:
    """Trip date + 'HH:MM' -> timezone-aware datetime in the app timezone."""
    hour, minute = parse_hhmm(hhmm)
    year, month, day = (int(p) for p in date_str.split("-"))
    return datetime(year, month, day, hour, minute, tzinfo=app_tz())


def minutes_until(date_str: str, hhmm: str, now: datetime | None = None) -> float:
    """Minutes from `now` until the trip departs (negative if already gone)."""
    now = now or get_now()
    return (combine(date_str, hhmm) - now).total_seconds() / 60.0


def shift_date(date_str: str, days: int) -> str:
    year, month, day = (int(p) for p in date_str.split("-"))
    return (datetime(year, month, day) + timedelta(days=days)).strftime("%Y-%m-%d")


def add_minutes_to_hhmm(hhmm: str, minutes: int) -> str:
    hour, minute = parse_hhmm(hhmm)
    base = datetime(2000, 1, 1, hour, minute) + timedelta(minutes=minutes)
    return base.strftime("%H:%M")


def round_up_to_step(dt: datetime, step_minutes: int = 5) -> datetime:
    """Round a datetime up to the next clean boundary (used by seeding so cabs
    land on times like 17:35 instead of 17:33:41)."""
    if dt.second or dt.microsecond or dt.minute % step_minutes:
        extra = step_minutes - (dt.minute % step_minutes)
        dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=extra)
    return dt
