"""LLM provider factory.

Default is Groq (free tier, OpenAI-compatible endpoint, very fast small
models - which is what keeps us inside the 5s p95 budget). Gemini and OpenAI
are one env var away; tests and evals use the scripted FakeChatModel so a
quota error can never cost a passing test.
"""
import json
import logging
import re

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from .config import Settings


class FakeChatModel(BaseChatModel):
    """Scripted model for tests/evals.

    Queue pre-canned AIMessages; each request pops the next one. Responses may
    carry tool_calls exactly like a real model would emit. Every call is
    recorded (tool names, message count) so tests can *assert which tools the
    agent chose*, which is one of the required test scenarios.
    """

    calls: list[dict] = Field(default_factory=list, exclude=True)
    _queue: list[AIMessage] = PrivateAttr(default_factory=list)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._queue = list(responses or [])

    # -- scripting helpers -------------------------------------------------
    def script(self, *responses: AIMessage) -> "FakeChatModel":
        self._queue.extend(responses)
        return self

    def script_tool_call(self, tool_name: str, args: dict, *, content: str = "") -> "FakeChatModel":
        return self.script(
            AIMessage(content=content, tool_calls=[{"name": tool_name, "args": args, "id": f"call_{len(self._queue)}"}])
        )

    # -- BaseChatModel plumbing -------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        self._bound_tool_names = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        tool_calls = [
            tc["name"] for m in messages if isinstance(m, AIMessage) and m.tool_calls for tc in m.tool_calls
        ]
        self.calls.append({"n_messages": len(messages), "prior_tool_calls": tool_calls})
        if self._queue:
            return ChatResult(generations=[ChatGeneration(message=self._queue.pop(0))])
        # Queue exhausted: behave like a well-trained assistant - relay the
        # last tool result's employee-facing wording. Deterministic, and it
        # exercises exactly the behaviour the system prompt demands.
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._relay_last_tool(messages)))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @staticmethod
    def _relay_last_tool(messages: list[BaseMessage]) -> str:
        from langchain_core.messages import ToolMessage

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                    if isinstance(payload, dict) and payload.get("say"):
                        return str(payload["say"])
                except (ValueError, TypeError):
                    pass
                return str(msg.content)
        return "I'm sorry - I couldn't process that. Please rephrase your request in one line."


class FailingChatModel(FakeChatModel):
    """Model whose provider is down - used to test graceful degradation."""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise TimeoutError("llm provider timed out")


class OfflineFallbackModel(BaseChatModel):
    """No API key (or provider down at boot): a tiny deterministic assistant.

    Regex intent matching -> the SAME tool calls a real model would make, and
    tool results are relayed verbatim - so rules, isolation, bookings,
    escalation and pause/resume all keep working without any LLM. Plain code
    is the better tool for this slice, which is exactly the case study's
    point: the model earns its place in the conversation, not in the rules.
    """

    calls: list[dict] = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "offline-rule-based"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    _TIME_RE = r"\b((?:[01]?\d|2[0-3]):[0-5]\d)\b"
    _INSIST_WORDS = ("insist", "really need", "still want", "escalate", "don't care",
                     "do not care", "override", "regardless", "anyway", "manager")
    _CHANGE_WORDS = ("move", "change", "reschedule", "shift", "make it", "go late",
                     "go early", "late", "leave early", "wfh", "leave tomorrow")
    _OPTIONS_WORDS = ("any cab", "cab around", "cab at", "options", "free seat",
                      "seats free", "availability", "available")
    _ROSTER_WORDS = ("roster", "schedule", "my drop", "my pickup", "what time is my",
                     "when is my", "which cab")
    _STATUS_WORDS = ("status", "stuck", "approved yet", "approval", "my request")
    _POLICY_WORDS = ("rule", "policy", "cut-off", "cutoff", "how many changes",
                     "allowed", "notice")

    @staticmethod
    def _human_texts(messages: list[BaseMessage]) -> list[str]:
        out = []
        for m in messages:
            if type(m).__name__ == "HumanMessage":
                text = m.content if isinstance(m.content, str) else str(m.content)
                out.append(text)
        return out

    def _route(self, history: list[str]) -> AIMessage:
        last = history[-1].lower()
        prior = history[-2] if len(history) > 1 else ""

        # reuse the most recent time in the conversation (insist follow-ups
        # like "I really need it, escalate" carry no new time)
        times = re.findall(self._TIME_RE, prior) if prior else []
        times += re.findall(self._TIME_RE, last)
        time = times[-1] if times else None

        trip = "pickup" if "pickup" in last or "pick up" in last else \
               "drop" if "drop" in last else None
        trip = trip or ("pickup" if "pickup" in prior.lower() or "pick up" in prior.lower() else
                        "drop" if "drop" in prior.lower() else None)
        date = "tomorrow" if "tomorrow" in last or "tomorrow" in prior.lower() else None

        def call(name: str, args: dict) -> AIMessage:
            return AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": f"offline_{len(history)}"}
            ])

        if any(w in last for w in self._INSIST_WORDS):
            if time:
                return call("request_shift_change", {
                    "new_time": time, "trip_type": trip, "date": date,
                    "insist": True, "reason": history[-1][:200],
                })
            return AIMessage(content="Which trip should I escalate, and to what time?")
        if time and any(w in last for w in self._CHANGE_WORDS):
            return call("request_shift_change", {
                "new_time": time, "trip_type": trip, "date": date,
            })
        if time and any(w in last for w in self._OPTIONS_WORDS):
            return call("find_cab_options", {"new_time": time, "trip_type": trip, "date": date})
        if any(w in last for w in self._ROSTER_WORDS):
            return call("get_my_roster", {"trip_type": trip, "date": date})
        if any(w in last for w in self._STATUS_WORDS):
            return call("check_request_status", {})
        if any(w in last for w in self._POLICY_WORDS):
            return call("get_company_policy", {})
        if time:
            return call("find_cab_options", {"new_time": time, "trip_type": trip, "date": date})
        if any(w in last for w in self._CHANGE_WORDS):
            return AIMessage(content=(
                "Sure I can help - which trip do you mean (pickup or drop), and what "
                "time should it be?"
            ))
        return AIMessage(content=(
            "I can help with your transport roster, cab options, shift changes and "
            "request status. What would you like to do?"
        ))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        self.calls.append({"n_messages": len(messages)})
        from langchain_core.messages import ToolMessage

        # After tools ran this turn, relay the employee-facing wording.
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                    if isinstance(payload, dict) and payload.get("say"):
                        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(payload["say"])))])
                except (ValueError, TypeError):
                    pass
                break
            if type(msg).__name__ == "HumanMessage":
                break  # reached this turn's user message: no tool result yet
        history = self._human_texts(messages)
        if not history:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Could you rephrase that?"))])
        return ChatResult(generations=[ChatGeneration(message=self._route(history))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _unavailable_model(reason: str) -> "OfflineFallbackModel":
    """No API key configured: boot anyway with the deterministic offline
    assistant. Rules, isolation, bookings, escalation and pause/resume keep
    working; only the natural phrasing is lost."""
    logging.getLogger("shift.llm").warning("LLM unavailable, using offline fallback: %s", reason)
    return OfflineFallbackModel()


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = (settings.llm_provider or "").lower()

    if provider == "fake":
        return FakeChatModel()

    if provider in ("groq", "openai"):
        from langchain_openai import ChatOpenAI

        if provider == "groq":
            api_key = settings.groq_api_key
            base_url = settings.groq_base_url
            model = settings.llm_model or "llama-3.1-8b-instant"
            if not api_key:
                return _unavailable_model(
                    "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                    "and put it in .env (or set LLM_PROVIDER=fake for a no-LLM run)."
                )
        else:
            api_key = settings.openai_api_key
            base_url = None
            model = settings.llm_model or "gpt-4o-mini"
            if not api_key:
                return _unavailable_model("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # a retry would blow the 5s p95; fail over to the fallback message
            max_tokens=settings.llm_max_output_tokens,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install langchain-google-genai to use LLM_PROVIDER=gemini") from exc
        if not settings.gemini_api_key:
            return _unavailable_model("GEMINI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            model=settings.llm_model or "gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def json_dumps(obj) -> str:  # noqa: ANN001
    return json.dumps(obj, ensure_ascii=False, default=str)
