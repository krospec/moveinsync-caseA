"""Chat orchestration: one service object owns the compiled graph.

Responsibilities:
* conversation ownership (a thread belongs to exactly one employee)
* per-turn state assembly (identity from the JWT, clock from app.clock)
* graceful degradation when the graph or the checkpointer fails
* transcript reads from the checkpointer (so history survives restarts)
* resuming a paused conversation with a manager's decision
"""
import json
import logging
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command

from .agent.graph import build_graph
from .clock import get_now
from .config import Settings, get_settings
from .db import SessionLocal
from .llm import build_chat_model
from .models import Company, Conversation

log = logging.getLogger("shift.chat")

GENERIC_ERROR_REPLY = (
    "Something went wrong on our side and I couldn't complete that. Please try again in a "
    "moment - if it's urgent, call the transport desk."
)


class ConversationNotOwned(Exception):
    pass


def _last_ai_reply(messages: list[BaseMessage]) -> str:
    """The reply for THIS turn only: messages after the last HumanMessage.
    Without that bound, an interrupted turn would surface wording from an
    earlier turn."""
    last_human = -1
    for i, m in enumerate(messages or []):
        if type(m).__name__ == "HumanMessage":
            last_human = i
    turn = list((messages or [])[last_human + 1:])

    for msg in reversed(turn):
        if isinstance(msg, AIMessage):
            content = (msg.content or "").strip()
            if isinstance(content, list):  # some providers return content blocks
                content = " ".join(str(part) for part in content).strip()
            if content:
                return content
    # Interrupted turn (paused for approval): no final AI wording exists yet -
    # relay the tool's employee-facing message instead of a generic error.
    from langchain_core.messages import ToolMessage

    for msg in reversed(turn):
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
                if isinstance(payload, dict) and payload.get("say"):
                    return str(payload["say"])
            except (ValueError, TypeError):
                pass
    return GENERIC_ERROR_REPLY


class ChatService:
    def __init__(self, settings: Settings | None = None, llm=None, checkpointer=None):  # noqa: ANN001
        self.settings = settings or get_settings()
        self.llm = llm
        if self.llm is None:
            self.llm = build_chat_model(self.settings)
        self.graph = build_graph(self.llm, self.settings, checkpointer)

    # ------------------------------------------------------------------ chat

    async def handle_message(self, *, identity, message: str, thread_id: str | None) -> dict:
        thread_id = thread_id or uuid.uuid4().hex
        now = get_now()
        await self._register_conversation(identity, thread_id, now)

        state_in = {
            "messages": [HumanMessage(content=message)],
            "employee_id": identity.employee_id,
            "company_id": identity.company_id,
            "employee_name": identity.name,
            "thread_id": thread_id,
            "today": now.strftime("%Y-%m-%d"),
            "now_iso": now.isoformat(),
            "now_hhmm": now.strftime("%H:%M"),
            "llm_calls": 0,
            "needs_approval": False,
            "pending_approval_id": None,
            "decision": None,
        }
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        try:
            state_in["company_name"] = await self._company_name(identity.company_id)
            final = await self.graph.ainvoke(state_in, config)
        except Exception:
            log.exception("graph invocation failed")
            return {
                "reply": GENERIC_ERROR_REPLY,
                "thread_id": thread_id,
                "status": "error",
                "request_id": None,
            }
        return {
            "reply": _last_ai_reply(final.get("messages", [])),
            "thread_id": thread_id,
            "status": "pending_approval" if final.get("needs_approval") else "ok",
            "request_id": final.get("pending_approval_id"),
        }

    # --------------------------------------------------------------- manager

    async def resume_decision(self, *, thread_id: str, decision: dict) -> dict:
        """Wake a paused conversation with the manager's decision.

        The graph resumes at approval_gate, applies the decision in plain code,
        and the agent words the final confirmation. Works on any process:
        the paused state lives in the checkpointer (Postgres).
        """
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        try:
            final = await self.graph.ainvoke(Command(resume=decision), config)
        except Exception as exc:  # no checkpoint / not paused / persistence error
            log.exception("resume failed for thread %s", thread_id)
            raise RuntimeError(f"could not resume conversation {thread_id}") from exc
        return {
            "reply": _last_ai_reply(final.get("messages", [])),
            "thread_id": thread_id,
            "status": "ok",
        }

    # ------------------------------------------------------------- transcript

    async def transcript(self, *, thread_id: str, identity) -> list[dict]:
        await self._assert_owner(identity, thread_id)
        snap = await self.graph.checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        if snap is None:
            return []
        messages = snap.checkpoint.get("channel_values", {}).get("messages", [])
        out = []
        for m in messages:
            if isinstance(m, AIMessage):
                content = (m.content or "").strip()
                if content and not getattr(m, "tool_calls", None):
                    out.append({"role": "assistant", "content": content})
            elif type(m).__name__ == "HumanMessage":
                out.append({"role": "employee", "content": (m.content or "").strip()})
        return out

    # ---------------------------------------------------------------- helpers

    async def _company_name(self, company_id: str) -> str:
        async with SessionLocal() as session:
            company = await session.get(Company, company_id)
            return company.name if company else company_id

    async def _register_conversation(self, identity, thread_id: str, now: datetime) -> None:
        async with SessionLocal() as session:
            conv = await session.get(Conversation, thread_id)
            if conv is None:
                session.add(
                    Conversation(
                        thread_id=thread_id,
                        employee_id=identity.employee_id,
                        company_id=identity.company_id,
                        created_at=now,
                        last_active_at=now,
                    )
                )
                await session.commit()
            elif conv.employee_id != identity.employee_id:
                raise ConversationNotOwned(thread_id)

    async def _assert_owner(self, identity, thread_id: str) -> None:
        async with SessionLocal() as session:
            conv = await session.get(Conversation, thread_id)
            if conv is None or conv.employee_id != identity.employee_id:
                raise ConversationNotOwned(thread_id)
