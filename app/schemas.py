"""Request/response schemas. Note what is *absent*: no employee_id, no
company_id - identity cannot be smuggled in through a request body."""
from pydantic import BaseModel, Field


class TokenIn(BaseModel):
    employee_id: str = Field(min_length=1, max_length=60, description="Demo IdP: employee id, e.g. acme-rhea")


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    employee_id: str
    company_id: str
    role: str
    name: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=60, description="Omit to start a new conversation")


class ChatOut(BaseModel):
    reply: str
    thread_id: str
    status: str  # ok | pending_approval | error
    request_id: str | None = None


class MessageOut(BaseModel):
    role: str
    content: str


class DecisionIn(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    note: str | None = Field(default=None, max_length=500)


class RequestOut(BaseModel):
    id: str
    employee_id: str
    company_id: str
    trip_type: str
    trip_date: str
    old_time: str
    requested_time: str
    status: str
    reason: str | None = None
    decision_by: str | None = None
    decision_note: str | None = None
