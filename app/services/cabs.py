"""Cab option search: which of *this company's* cabs has a free seat near the
requested time."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Cab
from ..timeutil import parse_hhmm


async def find_options(
    session: AsyncSession,
    *,
    company_id: str,
    trip_date: str,
    trip_type: str,
    target_time: str,
    limit: int = 3,
) -> list[dict]:
    """Cabs with at least one free seat, closest to target_time first."""
    stmt = select(Cab).where(
        Cab.company_id == company_id,
        Cab.trip_date == trip_date,
        Cab.trip_type == trip_type,
        Cab.booked_seats < Cab.capacity,
    )
    cabs = (await session.execute(stmt)).scalars().all()
    t_h, t_m = parse_hhmm(target_time)
    ranked = sorted(
        cabs,
        key=lambda c: abs((parse_hhmm(c.departure_time)[0] * 60 + parse_hhmm(c.departure_time)[1]) - (t_h * 60 + t_m)),
    )
    return [_cab_dict(c) for c in ranked[:limit]]


async def find_cab_at_time(
    session: AsyncSession,
    *,
    company_id: str,
    trip_date: str,
    trip_type: str,
    departure_time: str,
) -> Cab | None:
    stmt = select(Cab).where(
        Cab.company_id == company_id,
        Cab.trip_date == trip_date,
        Cab.trip_type == trip_type,
        Cab.departure_time == departure_time,
    )
    return (await session.execute(stmt)).scalars().first()


def _cab_dict(c: Cab) -> dict:
    return {
        "cab_id": c.id,
        "departure_time": c.departure_time,
        "route": c.route_name,
        "seats_free": c.capacity - c.booked_seats,
        "driver": c.driver,
    }
