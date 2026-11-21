"""Data model. Times are 'HH:MM' strings, dates 'YYYY-MM-DD' (IST).

Security note: every row that belongs to a client company carries company_id,
and every service function filters on it. Company/employee identity never
comes from the LLM or the request body - only from the JWT.
"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text

# All timestamps are timezone-aware (TIMESTAMP WITH TIME ZONE): the app clock
# is IST-aware, and asyncpg refuses naive columns fed with aware datetimes.
TS = DateTime(timezone=True)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

# Request lifecycle:
#   applied          - rule-compliant change, booked immediately
#   pending_approval - breaks a rule, employee insisted, waiting on manager
#   rejected         - manager said no
#   failed           - manager approved but the seat was taken meanwhile
#   cancelled        - employee withdrew (not used in flows yet, kept for completeness)
REQUEST_APPLIED = "applied"
REQUEST_PENDING = "pending_approval"
REQUEST_REJECTED = "rejected"
REQUEST_FAILED = "failed"
REQUEST_CANCELLED = "cancelled"


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    # Changes to a trip must arrive at least this many minutes before its
    # scheduled departure. This is THE rule enforced in code, per company.
    cutoff_minutes: Mapped[int] = mapped_column(Integer)
    max_changes_per_day: Mapped[int] = mapped_column(Integer, default=1)
    window_start: Mapped[str] = mapped_column(String(5), default="06:00")
    window_end: Mapped[str] = mapped_column(String(5), default="22:00")


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20), default="employee")  # employee | manager


class Cab(Base):
    __tablename__ = "cabs"

    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(40), ForeignKey("companies.id"), index=True)
    trip_date: Mapped[str] = mapped_column(String(10), index=True)
    trip_type: Mapped[str] = mapped_column(String(10))  # pickup | drop
    departure_time: Mapped[str] = mapped_column(String(5), index=True)
    route_name: Mapped[str] = mapped_column(String(160))
    capacity: Mapped[int] = mapped_column(Integer)
    booked_seats: Mapped[int] = mapped_column(Integer, default=0)
    driver: Mapped[str] = mapped_column(String(80), default="")


class RosterEntry(Base):
    __tablename__ = "roster_entries"

    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(60), ForeignKey("employees.id"), index=True)
    company_id: Mapped[str] = mapped_column(String(40), ForeignKey("companies.id"), index=True)
    trip_date: Mapped[str] = mapped_column(String(10), index=True)
    trip_type: Mapped[str] = mapped_column(String(10))
    scheduled_time: Mapped[str] = mapped_column(String(5))
    cab_id: Mapped[str] = mapped_column(String(60), ForeignKey("cabs.id"))
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled | changed
    changed_from: Mapped[str | None] = mapped_column(String(5), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class ShiftRequest(Base):
    __tablename__ = "shift_requests"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(60), ForeignKey("employees.id"), index=True)
    company_id: Mapped[str] = mapped_column(String(40), ForeignKey("companies.id"), index=True)
    thread_id: Mapped[str] = mapped_column(String(60), index=True)  # conversation that raised it
    roster_entry_id: Mapped[str] = mapped_column(String(60), ForeignKey("roster_entries.id"))
    trip_date: Mapped[str] = mapped_column(String(10))
    trip_type: Mapped[str] = mapped_column(String(10))
    old_time: Mapped[str] = mapped_column(String(5))
    requested_time: Mapped[str] = mapped_column(String(5))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    decision_by: Mapped[str | None] = mapped_column(String(60), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    # Same employee + same entry + same new time = same key. Double submits
    # collapse onto one row instead of creating two bookings.
    idempotency_key: Mapped[str | None] = mapped_column(String(140), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS)
    updated_at: Mapped[datetime] = mapped_column(TS)


class AuditLog(Base):
    """Clean record of who asked for what and who decided - the desk's paper
    trail, except it can't be skipped by a busy executive."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(TS)
    actor: Mapped[str] = mapped_column(String(80))  # employee id / manager id / "system"
    company_id: Mapped[str] = mapped_column(String(40), index=True)
    action: Mapped[str] = mapped_column(String(60))
    request_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    details: Mapped[str] = mapped_column(Text, default="{}")


class Conversation(Base):
    """Maps chat threads to their owner so a transcript is only ever readable
    by the employee who created it (and threads survive reconnects)."""

    __tablename__ = "conversations"

    thread_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(60), ForeignKey("employees.id"), index=True)
    company_id: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[datetime] = mapped_column(TS)
    last_active_at: Mapped[datetime] = mapped_column(TS)
