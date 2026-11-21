"""Agent state.

`messages` uses LangGraph's append reducer; every other field is authoritative
server-side state - written by the graph nodes, never by the model.

Identity fields (employee_id / company_id) are copied from the verified JWT at
invocation time. Tools read them from here; the tool schemas the LLM sees have
no identity parameters at all.
"""
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # server-side identity + context (from JWT, not the model)
    employee_id: str
    company_id: str
    employee_name: str
    company_name: str
    thread_id: str
    today: str          # YYYY-MM-DD in the app timezone
    now_iso: str        # full ISO timestamp; tools derive the clock from this
    now_hhmm: str       # current wall-clock time, HH:MM (for the prompt)

    # loop control
    llm_calls: int                      # budget: max 4 per employee message
    needs_approval: bool                # a tool escalated -> pause for the manager
    pending_approval_id: Optional[str]  # shift request waiting on the manager
    decision: Optional[dict]            # manager's resume payload (approve/reject)
