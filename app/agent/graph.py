"""The conversation graph.

    START -> agent -> (tool calls?) -> tools -> (escalated?) -> approval_gate
                 ^                                  |no              |
                 |                                  v                v
                 +-------------------------------- agent <- apply_decision

Why a hand-rolled StateGraph instead of a prebuilt ReAct loop:

* The LLM-call budget (max 4 per employee message) is enforced *in the agent
  node*, by counter - the model cannot talk its way past it.
* Escalation is deterministic: only the tools node, reading a tool result of
  status "escalated" (which only the booking engine can produce), routes to the
  approval gate. The model cannot trigger a pause, and cannot approve itself.
* Manager decision handling is a plain code node (apply_decision), so an
  approval books a seat even if the model is behaving badly.
"""
import logging
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..config import Settings
from ..llm import json_dumps
from ..models import Employee
from ..db import SessionLocal
from ..services import approval
from ..services.approval import NotAllowed
from .prompts import TEMPLATE
from .state import AgentState
from .tools import EXECUTED_TOOLS, RAW, TOOL_SCHEMAS, TOOL_DOWN_SAY, ToolContext, resolve_date_alias

log = logging.getLogger("shift.agent")

BUDGET_MSG = (
    "I wasn't able to finish that within my step budget. Please rephrase it as one "
    "clear change - for example: 'Move my drop today to 20:00'."
)
LLM_DOWN_MSG = (
    "I'm having trouble reaching the assistant service right now. Please try again in a "
    "moment - if it's urgent, call the transport desk."
)
UNKNOWN_TOOL_SAY = "I don't have that capability; please rephrase your request."


def build_graph(llm, settings: Settings, checkpointer=None):  # noqa: ANN001
    bound = llm.bind_tools(TOOL_SCHEMAS)

    # -- nodes ---------------------------------------------------------------

    async def agent_node(state: AgentState) -> dict:
        calls = state.get("llm_calls", 0)
        if calls >= settings.llm_max_calls_per_turn:
            # Hard budget stop: answer without another model call.
            return {"messages": [AIMessage(content=BUDGET_MSG)]}
        system = SystemMessage(
            TEMPLATE.format(
                company_name=state.get("company_name", ""),
                employee_name=state.get("employee_name", ""),
                employee_id=state.get("employee_id", ""),
                company_id=state.get("company_id", ""),
                today=state.get("today", ""),
                now_hhmm=state.get("now_hhmm", ""),
            )
        )
        window = list(state["messages"][-settings.history_window:])
        try:
            resp = await bound.ainvoke([system] + window)
        except Exception:
            # Timeout, quota, provider outage -> graceful, never a hang.
            log.exception("LLM call failed")
            resp = AIMessage(LLM_DOWN_MSG)
        return {"messages": [resp], "llm_calls": calls + 1}

    def route_from_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    async def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        ctx = ToolContext(
            employee_id=state["employee_id"],
            company_id=state["company_id"],
            employee_name=state.get("employee_name", ""),
            thread_id=state.get("thread_id", ""),
            today=state.get("today", ""),
            now=datetime.fromisoformat(state["now_iso"]),
        )
        messages: list[ToolMessage] = []
        needs_approval = False
        pending_id = None
        for call in last.tool_calls:
            name = call["name"]
            args = dict(call.get("args") or {})
            EXECUTED_TOOLS.append(name)
            fn = RAW.get(name)
            if fn is None:
                result = {"ok": False, "say": UNKNOWN_TOOL_SAY}
            else:
                if args.get("date") is not None:
                    args["date"] = resolve_date_alias(args["date"], ctx)
                try:
                    result = await fn(ctx, **args)
                except Exception:
                    log.exception("tool %s failed", name)
                    result = {"ok": False, "error": "tool_failed", "say": TOOL_DOWN_SAY}
            messages.append(
                ToolMessage(
                    content=json_dumps(result),
                    name=name,
                    tool_call_id=call.get("id") or name,
                )
            )
            if result.get("status") == "escalated" and result.get("request_id"):
                needs_approval = True
                pending_id = result["request_id"]
        return {
            "messages": messages,
            "needs_approval": needs_approval,
            "pending_approval_id": pending_id,
        }

    def route_from_tools(state: AgentState) -> str:
        return "approval_gate" if state.get("needs_approval") else "agent"

    async def approval_gate(state: AgentState) -> dict:
        """Pauses the conversation. The interrupt payload is persisted in the
        checkpointer, so this survives a restart. On resume, `interrupt()`
        returns the manager's decision dict."""
        decision = interrupt(
            {"type": "manager_approval", "request_id": state.get("pending_approval_id")}
        )
        return {"decision": decision}

    async def apply_decision(state: AgentState) -> dict:
        decision = state.get("decision") or {}
        say = "Something went wrong while applying the decision. Please contact the transport desk."
        ok = False
        async with SessionLocal() as session:
            manager = await session.get(Employee, decision.get("decided_by") or "")
            if manager is None:
                say = "The deciding manager could not be found; the request stays pending."
            else:
                try:
                    outcome = await approval.decide(
                        session,
                        request_id=state.get("pending_approval_id") or "",
                        manager=manager,
                        decision=decision.get("decision", ""),
                        note=decision.get("note"),
                        now=datetime.fromisoformat(state["now_iso"]),
                    )
                    say = outcome["message"]
                    ok = outcome.get("status") in ("applied", "rejected")
                except NotAllowed:
                    log.warning("decision blocked for manager %s", manager.id)
        tool_msg = ToolMessage(
            content=json_dumps({"ok": ok, "say": say, "status": decision.get("decision")}),
            name="manager_decision",
            tool_call_id="manager_decision",
        )
        return {
            "messages": [tool_msg],
            "needs_approval": False,
            "pending_approval_id": None,
            "decision": None,
            "llm_calls": 0,  # fresh budget for the resumed turn's final wording
        }

    # -- wiring ---------------------------------------------------------------

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("apply_decision", apply_decision)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", route_from_tools, {"agent": "agent", "approval_gate": "approval_gate"})
    builder.add_edge("approval_gate", "apply_decision")
    builder.add_edge("apply_decision", "agent")

    return builder.compile(checkpointer=checkpointer)
