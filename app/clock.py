"""Single source of "now" for the whole app.

Business rules (cut-off checks) compare wall-clock time against trip times.
Tests and evals freeze this clock via APP_FROZEN_NOW so the cut-off rule is
deterministic no matter when the suite runs.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import get_settings


def get_now() -> datetime:
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    if settings.frozen_now:
        raw = datetime.fromisoformat(settings.frozen_now)
        if raw.tzinfo is None:
            raw = raw.replace(tzinfo=tz)
        return raw.astimezone(tz)
    return datetime.now(tz)


def today_str() -> str:
    return get_now().strftime("%Y-%m-%d")
