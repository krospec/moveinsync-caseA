"""Agent behaviour: tool selection, clarifying questions, graceful failures,
LLM-call budget."""
from app.agent.graph import BUDGET_MSG, LLM_DOWN_MSG
from app.agent.tools import EXECUTED_TOOLS
from app.agent_service import ChatService
from app.llm import FailingChatModel
from tests.conftest import TOMORROW, auth_token, chat


async def test_roster_question_routes_to_roster_tool(client, fake_llm):
    token = await auth_token(client, "acme-rhea")
    fake_llm.script_tool_call("get_my_roster", {"trip_type": "pickup", "date": TOMORROW})
    result = await chat(client, token, "What time is my pickup tomorrow?")

    assert EXECUTED_TOOLS == ["get_my_roster"]  # exactly this tool, nothing else
    assert "08:30" in result["reply"]


async def test_options_question_routes_to_options_tool(client, fake_llm):
    token = await auth_token(client, "acme-rhea")
    fake_llm.script_tool_call("find_cab_options", {"new_time": "20:00"})
    result = await chat(client, token, "Is there a cab around 20:00 for my drop?")

    assert EXECUTED_TOOLS == ["find_cab_options"]
    assert "20:00" in result["reply"] and "20:30" in result["reply"]  # nearest slots listed


async def test_ambiguous_request_gets_a_question_not_a_guess(client, fake_llm):
    """"Can I go a bit late?" names no trip and no time -> ask, don't act."""
    from langchain_core.messages import AIMessage

    token = await auth_token(client, "acme-rhea")
    fake_llm.script(AIMessage(content="Sure I can help - which trip do you mean, and what time?"))
    result = await chat(client, token, "Can I go a bit late today?")

    assert "?" in result["reply"]
    assert EXECUTED_TOOLS == []  # no tool ran; nothing was guessed
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import ShiftRequest

    async with SessionLocal() as session:
        assert (await session.execute(select(func.count()).select_from(ShiftRequest))).scalar_one() == 0


async def test_tool_failure_becomes_a_friendly_reply(client, fake_llm, monkeypatch):
    """Roster service is down -> the employee gets plain words, never a trace,
    never a hang."""
    from app.services import roster as roster_service

    async def boom(*args, **kwargs):
        raise TimeoutError("upstream timeout")

    monkeypatch.setattr(roster_service, "get_roster", boom)
    token = await auth_token(client, "acme-rhea")
    fake_llm.script_tool_call("get_my_roster", {})
    result = await chat(client, token, "What's my roster today?")

    assert "try again" in result["reply"].lower()
    assert "Traceback" not in result["reply"]
    assert result["status"] == "ok"
    assert EXECUTED_TOOLS == ["get_my_roster"]


async def test_llm_call_budget_is_hard_capped_at_four(client, fake_llm):
    """Script a model that keeps calling tools: after the 4th LLM call the
    graph answers for it - no 5th model call, employee still gets a reply."""
    token = await auth_token(client, "acme-sam")
    for _ in range(4):
        fake_llm.script_tool_call("get_company_policy", {})
    result = await chat(client, token, "What are my company's rules?")

    assert fake_llm.calls and len(fake_llm.calls) == 4  # not 5
    assert result["reply"] == BUDGET_MSG


async def test_llm_outage_becomes_friendly_message(client, app):
    """Provider completely down -> graceful message, no hang, no 500."""
    inner = app.state.chat_service
    original = app.state.chat_service
    service = ChatService(settings=inner.settings, llm=FailingChatModel(), checkpointer=inner.graph.checkpointer)
    app.state.chat_service = service
    try:
        token = await auth_token(client, "acme-sam")
        result = await chat(client, token, "hello")
        assert result["reply"] == LLM_DOWN_MSG
        assert result["status"] == "ok"
    finally:
        app.state.chat_service = original


async def test_request_status_tool_reports_pending(client, fake_llm):
    """'Is my request stuck?' -> the status tool, not a hallucinated answer."""
    from tests.test_approval import _escalate

    ctx = await _escalate(client, fake_llm, "acme-rhea")
    token = ctx["token"]
    before = len(EXECUTED_TOOLS)  # escalate flow already ran its tools
    fake_llm.script_tool_call("check_request_status", {})
    result = await chat(client, token, "Is my request still stuck in approval?", thread_id=ctx["thread_id"])

    assert EXECUTED_TOOLS[before:] == ["check_request_status"]
    assert "waiting" in result["reply"].lower() or "manager" in result["reply"].lower()


def test_boots_without_api_key_via_fallback_model():
    """No GROQ_API_KEY -> app still boots with the deterministic offline
    assistant instead of crashing; chats keep working through the same tools."""
    from app.config import Settings
    from app.llm import OfflineFallbackModel, build_chat_model

    settings = Settings(llm_provider="groq", groq_api_key=None)
    assert isinstance(build_chat_model(settings), OfflineFallbackModel)
