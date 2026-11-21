"""Sample data: two client companies with deliberately different rules.

Company   cut-off   cab size   changes/day   window
acme      120 min   4 seats    1             06:00-22:00
globex     30 min   6 seats    2             06:30-21:30

Cab times are seeded RELATIVE to "now" (the frozen clock in tests/evals, real
clock in docker). That is what makes the cut-off demos work at any hour: there
is always a trip that is inside the cut-off window and one that is not.

`booked_seats` includes a few unmodelled co-riders (real cab sharing), so a
cab with 3 booked and only 2 roster entries here has one phantom passenger.
"""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .clock import get_now
from .db import SessionLocal
from .models import Cab, Company, Employee, RosterEntry
from .timeutil import add_minutes_to_hhmm, round_up_to_step, shift_date  # noqa: F401 (add_minutes used below)

ACME = "acme"
GLOBEX = "globex"


async def seed_if_empty(session: AsyncSession | None = None) -> None:
    own = session is None
    if own:
        session = SessionLocal()
    try:
        count = (await session.execute(select(func.count(Company.id)))).scalar_one()
        if count:
            return
        await _seed(session)
        await session.commit()
    finally:
        if own:
            await session.close()




async def _seed(session: AsyncSession) -> None:
    now = get_now()
    today = now.strftime("%Y-%m-%d")
    tomorrow = shift_date(today, 1)

    def dep(minutes_ahead: int) -> str:
        """Departure time N minutes from now, rounded to a clean 5-min mark."""
        return add_minutes_to_hhmm(round_up_to_step(now).strftime("%H:%M"), minutes_ahead)

    # Models declare bare ForeignKeys without relationships, so the unit of
    # work cannot infer table order: flush parents before children. Postgres
    # enforces FKs immediately; sqlite (tests) would silently let this slide.
    session.add_all(
        [
            Company(
                id=ACME, name="Acme Manufacturing", cutoff_minutes=120,
                max_changes_per_day=1, window_start="06:00", window_end="22:00",
            ),
            Company(
                id=GLOBEX, name="Globex Software", cutoff_minutes=30,
                max_changes_per_day=2, window_start="06:30", window_end="21:30",
            ),
        ]
    )
    await session.flush()

    session.add_all(
        [
            Employee(id="acme-rhea", company_id=ACME, name="Rhea Kapoor"),
            Employee(id="acme-sam", company_id=ACME, name="Sam Fernandes"),
            Employee(id="acme-neha", company_id=ACME, name="Neha Verma"),
            Employee(id="acme-dev", company_id=ACME, name="Dev Iyer"),
            Employee(id="acme-tara", company_id=ACME, name="Tara Shah"),
            Employee(id="acme-mgr", company_id=ACME, name="Anita Rao", role="manager"),
            Employee(id="globex-arjun", company_id=GLOBEX, name="Arjun Mehta"),
            Employee(id="globex-priya", company_id=GLOBEX, name="Priya Nair"),
            Employee(id="globex-mgr", company_id=GLOBEX, name="Farhan Qureshi", role="manager"),
        ]
    )

    # --- acme cabs (drops today) ---
    cabs = [
        # Rhea's drop: 90 min away -> inside Acme's 120-min cut-off (reject demo).
        Cab(id="ACME-D1", company_id=ACME, trip_date=today, trip_type="drop",
            departure_time=dep(90), route_name="Whitefield -> EcoSpace",
            capacity=4, booked_seats=2, driver="Ravi"),
        # The contested cab: 1 seat free -> concurrency race target.
        Cab(id="ACME-D2", company_id=ACME, trip_date=today, trip_type="drop",
            departure_time=dep(180), route_name="Whitefield -> EcoSpace",
            capacity=4, booked_seats=3, driver="Suresh"),
        # Fallback slot with plenty of room.
        Cab(id="ACME-D3", company_id=ACME, trip_date=today, trip_type="drop",
            departure_time=dep(210), route_name="Whitefield -> EcoSpace",
            capacity=4, booked_seats=0, driver="Mohan"),
        # Completely full cab -> "cab filled up while deciding" demo.
        Cab(id="ACME-DF", company_id=ACME, trip_date=today, trip_type="drop",
            departure_time=dep(240), route_name="Hebbal -> Manyata",
            capacity=4, booked_seats=4, driver="Kiran"),
        Cab(id="ACME-P1", company_id=ACME, trip_date=today, trip_type="pickup",
            departure_time=dep(50), route_name="BTM -> EcoSpace",
            capacity=4, booked_seats=3, driver="Ravi"),
        # Tomorrow's pickups: cut-off is far away, so changes are allowed.
        Cab(id="ACME-PT1", company_id=ACME, trip_date=tomorrow, trip_type="pickup",
            departure_time="08:30", route_name="HSR -> EcoSpace",
            capacity=4, booked_seats=0, driver="Suresh"),
        Cab(id="ACME-PT2", company_id=ACME, trip_date=tomorrow, trip_type="pickup",
            departure_time="09:15", route_name="HSR -> EcoSpace",
            capacity=4, booked_seats=1, driver="Mohan"),
        # --- globex cabs ---
        # Arjun's drop: 20 min away -> inside Globex's 30-min cut-off.
        Cab(id="GLOB-D1", company_id=GLOBEX, trip_date=today, trip_type="drop",
            departure_time=dep(20), route_name="Outer Ring Rd -> Campus 2",
            capacity=6, booked_seats=5, driver="Imran"),
        # Full cab for Priya.
        Cab(id="GLOB-D2", company_id=GLOBEX, trip_date=today, trip_type="drop",
            departure_time=dep(150), route_name="Outer Ring Rd -> Campus 1",
            capacity=6, booked_seats=6, driver="Joseph"),
        # Roomy fallback.
        Cab(id="GLOB-D3", company_id=GLOBEX, trip_date=today, trip_type="drop",
            departure_time=dep(180), route_name="Outer Ring Rd -> Campus 1",
            capacity=6, booked_seats=1, driver="Vikram"),
        Cab(id="GLOB-P1", company_id=GLOBEX, trip_date=tomorrow, trip_type="pickup",
            departure_time="09:00", route_name="Indiranagar -> Campus 2",
            capacity=6, booked_seats=2, driver="Imran"),
    ]
    session.add_all(cabs)
    await session.flush()

    session.add_all(
        [
            RosterEntry(id="RE-RH-D", employee_id="acme-rhea", company_id=ACME,
                        trip_date=today, trip_type="drop", scheduled_time=dep(90),
                        cab_id="ACME-D1"),
            RosterEntry(id="RE-RH-P", employee_id="acme-rhea", company_id=ACME,
                        trip_date=today, trip_type="pickup", scheduled_time=dep(50),
                        cab_id="ACME-P1"),
            RosterEntry(id="RE-RH-PT", employee_id="acme-rhea", company_id=ACME,
                        trip_date=tomorrow, trip_type="pickup", scheduled_time="08:30",
                        cab_id="ACME-PT1"),
            RosterEntry(id="RE-SA-D", employee_id="acme-sam", company_id=ACME,
                        trip_date=today, trip_type="drop", scheduled_time=dep(180),
                        cab_id="ACME-D2"),
            RosterEntry(id="RE-SA-P", employee_id="acme-sam", company_id=ACME,
                        trip_date=today, trip_type="pickup", scheduled_time=dep(50),
                        cab_id="ACME-P1"),
            RosterEntry(id="RE-NE-D", employee_id="acme-neha", company_id=ACME,
                        trip_date=today, trip_type="drop", scheduled_time=dep(180),
                        cab_id="ACME-D2"),
            RosterEntry(id="RE-DEV-D", employee_id="acme-dev", company_id=ACME,
                        trip_date=today, trip_type="drop", scheduled_time=dep(240),
                        cab_id="ACME-DF"),
            RosterEntry(id="RE-TA-D", employee_id="acme-tara", company_id=ACME,
                        trip_date=today, trip_type="drop", scheduled_time=dep(240),
                        cab_id="ACME-DF"),
            RosterEntry(id="RE-AR-D", employee_id="globex-arjun", company_id=GLOBEX,
                        trip_date=today, trip_type="drop", scheduled_time=dep(20),
                        cab_id="GLOB-D1"),
            RosterEntry(id="RE-AR-PT", employee_id="globex-arjun", company_id=GLOBEX,
                        trip_date=tomorrow, trip_type="pickup", scheduled_time="09:00",
                        cab_id="GLOB-P1"),
            RosterEntry(id="RE-PR-D", employee_id="globex-priya", company_id=GLOBEX,
                        trip_date=today, trip_type="drop", scheduled_time=dep(150),
                        cab_id="GLOB-D2"),
        ]
    )
