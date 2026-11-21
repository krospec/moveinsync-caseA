"""Roster reads. Identity is passed in by the caller (the API layer, from the
JWT) and is applied as a WHERE clause - the model can never widen it."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Cab, RosterEntry


async def get_roster(
    session: AsyncSession,
    *,
    employee_id: str,
    company_id: str,
    trip_date: str | None = None,
    trip_type: str | None = None,
) -> list[dict]:
    """The given employee's roster entries, always scoped to their company."""
    stmt = (
        select(RosterEntry, Cab)
        .join(Cab, Cab.id == RosterEntry.cab_id)
        .where(
            RosterEntry.employee_id == employee_id,
            RosterEntry.company_id == company_id,  # company scoping is redundant here but explicit
        )
        .order_by(RosterEntry.trip_date, RosterEntry.scheduled_time)
    )
    if trip_date:
        stmt = stmt.where(RosterEntry.trip_date == trip_date)
    if trip_type in ("pickup", "drop"):
        stmt = stmt.where(RosterEntry.trip_type == trip_type)

    rows = (await session.execute(stmt)).all()
    return [
        {
            "roster_entry_id": entry.id,
            "date": entry.trip_date,
            "trip_type": entry.trip_type,
            "scheduled_time": entry.scheduled_time,
            "cab_id": cab.id,
            "route": cab.route_name,
            "capacity": cab.capacity,
            "seats_free": cab.capacity - cab.booked_seats,
            "status": entry.status,
            "changed_from": entry.changed_from,
        }
        for entry, cab in rows
    ]
