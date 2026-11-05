"""LLM team manager. Runs only in the GK agent.

maybe_trigger() is called every tick and returns at once. After each goal
and every TRIGGER_EVERY_S of game time it starts ONE background thread that
asks the LLM for a strategy and writes it with shared.write_strategy().

- invalid or late LLM answer -> keep the current strategy
- LLM unavailable (error) -> deterministic rules
- hysteresis: at most one switch per MIN_SWITCH_S, unless a goal happened
Env ENABLE_MANAGER=0 disables it. Independent of team_config enable_llm.
"""

import json
import os
import re
import threading
import time

import match_memory
import shared
import strategies

MODEL_ID = "us.amazon.nova-lite-v1:0"
TRIGGER_EVERY_S = 30.0
MIN_SWITCH_S = 20.0
LATE_S = 10.0              # wall-clock limit for one LLM answer
HALF_TIME_S = 150.0
LATE_GAME_S = 230.0

SYSTEM_PROMPT = (
    "You are the manager of a 5v5 robot football team. Pick the team strategy "
    "for the next 30 seconds. Answer ONLY one JSON object: "
    '{"strategy": "<NAME>", "reason": "<max 12 words>"}'
)

_lock = threading.Lock()
_state = {"busy": False, "last_t": None, "last_goals": 0, "last_switch_t": -1e9,
          "match": None, "thread": None, "last_result": None}

# Tests replace this with a fake: fn(prompt:str) -> str. None = real Strands Agent.
LLM_CALL = None


def enabled() -> bool:
    return os.environ.get("ENABLE_MANAGER", "1") != "0"


def reset() -> None:
    _state.update(busy=False, last_t=None, last_goals=0, last_switch_t=-1e9,
                  match=None, thread=None, last_result=None)


def _llm(prompt: str) -> str:
    if LLM_CALL is not None:
        return LLM_CALL(prompt)
    from botocore.config import Config
    from strands import Agent
    from strands.models import BedrockModel
    model = BedrockModel(
        model_id=MODEL_ID, temperature=0.2, max_tokens=150,
        boto_client_config=Config(connect_timeout=2, read_timeout=LATE_S, retries={"total_max_attempts": 1, "mode": "standard"}),
    )
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
    return str(agent(prompt))


def build_prompt(st: dict, current: str) -> str:
    names = "\n".join(f"- {n}: {strategies.DESCRIPTIONS[n]}" for n in strategies.NAMES)
    return (f"Match report:\n{match_memory.report()}\n\n"
            f"Current strategy: {current}\n"
            f"Score us {st['score'][0]} - {st['score'][1]} them, game time {st['t']:.0f}s of about 300s.\n"
            f"Strategies:\n{names}\n"
            'Answer JSON only: {"strategy": "<NAME>", "reason": "..."}')


def parse_answer(text) -> tuple[str, str] | None:
    """(strategy, reason) from the LLM text, or None when invalid."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = strategies.normalize(obj.get("strategy"))
    if name is None:
        return None
    return name, str(obj.get("reason", ""))[:120]


def rules(st: dict) -> tuple[str, str] | None:
    """Deterministic choice when the LLM is unavailable. None = keep current."""
    us, them = st["score"]
    t = st["t"]
    if us < them and t > HALF_TIME_S:
        return "ALL_OUT", "rules: losing after half-time"
    if us > them and t > LATE_GAME_S:
        return "LOW_BLOCK", "rules: leading late"
    if st.get("conceded_counters", 0) >= 2:
        return "LOW_BLOCK", "rules: conceded 2+ from counters"
    if us == them and t < 10:
        return "BALANCED", "rules: kick-off"
    return None


def _decide(st: dict, goal: bool, kickoff: bool, current: str) -> tuple[str, str, str] | None:
    """(strategy, reason, via) or None. Runs in the background thread."""
    if kickoff:
        r = rules(st)
        return (r[0], r[1], "rules") if r else None
    t0 = time.monotonic()
    try:
        text = _llm(build_prompt(st, current))
    except Exception as e:  # noqa: BLE001  LLM unavailable
        r = rules(st)
        _state["last_result"] = f"llm error {e!r}"[:200]
        return (r[0], r[1], "rules") if r else None
    if time.monotonic() - t0 > LATE_S:
        _state["last_result"] = "llm late"
        return None
    ans = parse_answer(text)
    if ans is None:
        _state["last_result"] = f"llm invalid: {str(text)[:100]}"
        return None
    return ans[0], ans[1], "llm"


def _run(st: dict, goal: bool, kickoff: bool) -> None:
    try:
        current = shared.current().get("strategy", strategies.DEFAULT)
        res = _decide(st, goal, kickoff, current)
        if res is None:
            return
        name, reason, via = res
        if name == current and not kickoff:
            _state["last_result"] = f"keep {name} ({via})"
            return
        if not (goal or kickoff) and st["t"] - _state["last_switch_t"] < MIN_SWITCH_S:
            _state["last_result"] = f"hysteresis: skip {name}"
            return
        shared.write_strategy(name, f"{via}: {reason}", by="manager-gk")
        _state["last_switch_t"] = st["t"]
        _state["last_result"] = f"switch {name} ({via})"
    except Exception as e:  # noqa: BLE001
        _state["last_result"] = f"error {e!r}"[:200]
    finally:
        _state["busy"] = False


def maybe_trigger(game_state: dict, team_id: int, role: str = "GK") -> str | None:
    """Start a background decision when due. Returns the trigger reason or None. Never blocks."""
    try:
        if role != "GK" or not enabled():
            return None
        st = match_memory.stats()
        t = float(st["t"] or 0.0)
        match = match_memory.MEMORY.resets
        kickoff = _state["match"] != match
        goal = st["goals"] > _state["last_goals"]
        due = _state["last_t"] is None or t - _state["last_t"] >= TRIGGER_EVERY_S or t < _state["last_t"]
        if not (kickoff or goal or due):
            return None
        with _lock:
            if _state["busy"]:
                return None
            _state["busy"] = True
        if kickoff:
            _state["last_switch_t"] = -1e9
        _state.update(match=match, last_t=t, last_goals=st["goals"])
        why = "kickoff" if kickoff else ("goal" if goal else "timer")
        th = threading.Thread(target=_run, args=(dict(st), goal, kickoff), daemon=True, name="afc-manager")
        _state["thread"] = th
        th.start()
        return why
    except Exception:  # noqa: BLE001
        _state["busy"] = False
        return None
