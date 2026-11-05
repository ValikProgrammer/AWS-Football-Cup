"""Tests for strategies, shared (SSM), match_memory, manager, tactic orders and the handler path."""

import asyncio
import copy
import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from test_helpers import mock_agentcore, GAME_STATE, _FakeApp  # noqa: E402
mock_agentcore()

import manager  # noqa: E402
import match_memory  # noqa: E402
import norm  # noqa: E402
import orders  # noqa: E402
import policy  # noqa: E402
import shared  # noqa: E402
import strategies  # noqa: E402
from parsing import VALID_COMMANDS  # noqa: E402

PARAMS = policy.load_params()
ROLES = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}


@pytest.fixture(autouse=True)
def _reset_norm():
    norm.reset()
    orders.TRACKER.__init__()
    yield


class FakeSSM:
    """Fake boto3 SSM client."""

    def __init__(self, value=None, exc=None):
        self.value = value
        self.exc = exc
        self.gets = 0
        self.puts = []

    def get_parameter(self, Name):
        self.gets += 1
        if self.exc is not None:
            raise self.exc
        return {"Parameter": {"Name": Name, "Value": self.value}}

    def put_parameter(self, **kw):
        if self.exc is not None:
            raise self.exc
        self.puts.append(kw)
        self.value = kw["Value"]
        return {"Version": len(self.puts)}


def rec(name, seq=1):
    return json.dumps({"strategy": name, "reason": "t", "seq": seq, "ts": 0, "by": "test"})


# ---------------------------------------------------------------- strategies

def test_presets_apply_and_copy():
    base = dict(PARAMS)
    assert strategies.apply(PARAMS, "BALANCED") == PARAMS
    hp = strategies.apply(PARAMS, "HIGH_PRESS")
    assert hp["press_bodies"] == 2 and hp["line_height"] > PARAMS["line_height"]
    lb = strategies.apply(PARAMS, "LOW_BLOCK")
    assert lb["press_bodies"] == 1 and lb["line_height"] < PARAMS["line_height"]
    ao = strategies.apply(PARAMS, "ALL_OUT")
    assert ao["shot_bonus"] > PARAMS["shot_bonus"] and ao["shoot_range"] > PARAMS["shoot_range"]
    assert strategies.apply(PARAMS, "POSSESSION")["pass_bonus"] > PARAMS["pass_bonus"]
    assert strategies.apply(PARAMS, "COUNTER")["def_max_x"] < PARAMS["def_max_x"]
    assert PARAMS == base                       # input not changed
    extra_keys = {"through_ahead", "press_max_x"}
    for name, over in strategies.PRESETS.items():
        assert set(over) <= set(PARAMS) | extra_keys, name
        assert name in strategies.DESCRIPTIONS


@pytest.mark.parametrize("name", ["NOPE", None, 42, ""])
def test_unknown_preset_is_balanced(name):
    assert strategies.apply(PARAMS, name) == PARAMS


def test_normalize_names():
    assert strategies.normalize("high press") == "HIGH_PRESS"
    assert strategies.normalize("lowblock") == "LOW_BLOCK"
    assert strategies.normalize("all-out") == "ALL_OUT"
    assert strategies.normalize("x") is None


@pytest.mark.parametrize("name", strategies.NAMES)
@pytest.mark.parametrize("pid", list(ROLES))
def test_every_preset_valid_commands(name, pid):
    c = policy.decide(GAME_STATE, 0, pid, ROLES[pid], strategies.apply(PARAMS, name))[0].command
    assert c["commandType"] in VALID_COMMANDS and c["playerId"] == pid


def _opp_holds_high():
    gs = copy.deepcopy(GAME_STATE)
    gs["players"][9]["position"] = {"x": 30, "y": 5}      # away agentId_4 holds in their half
    gs["ball"].update(position={"x": 30, "y": 5, "z": 0}, possessionAgentId="agentId_4", isFree=False)
    gs["players"][4]["position"] = {"x": 27, "y": 5}      # home FWD2 right next to it
    return gs


def test_low_block_does_not_press_high():
    gs = _opp_holds_high()
    types = lambda p: [policy.decide(gs, 0, i, ROLES[i], p)[0].command["commandType"] for i in ROLES]
    pressing = {"PRESS_BALL", "SLIDE_TACKLE"}   # a close presser tackles instead of pressing
    assert pressing & set(types(PARAMS))
    assert not pressing & set(types(strategies.apply(PARAMS, "LOW_BLOCK")))


# ---------------------------------------------------------------- shared

def test_shared_read_uses_cache(monkeypatch):
    monkeypatch.setenv("SHARED_STATE", "1")
    fake = FakeSSM(rec("COUNTER"))
    shared.set_client(fake)
    assert shared.read_strategy() == "COUNTER"
    assert shared.read_strategy() == "COUNTER"
    assert fake.gets == 1 and shared.source() == "shared"
    fake.value = rec("LOW_BLOCK", 2)
    shared._state["fetched"] = -1e9                  # cache expired
    assert shared.read_strategy() == "LOW_BLOCK" and fake.gets == 2


def test_shared_survives_exceptions_and_timeouts(monkeypatch):
    from botocore.exceptions import ReadTimeoutError, ConnectTimeoutError
    monkeypatch.setenv("SHARED_STATE", "1")
    assert shared.read_strategy() == "BALANCED"      # guard client raises -> default
    fake = FakeSSM(rec("HIGH_PRESS"))
    shared.set_client(fake)
    shared._state.update(fetched=-1e9, retry_after=0.0)
    assert shared.read_strategy() == "HIGH_PRESS"
    for exc in (ReadTimeoutError(endpoint_url="x"), ConnectTimeoutError(endpoint_url="x"),
                RuntimeError("AccessDenied"), KeyError("Parameter")):
        fake.exc = exc
        shared._state.update(fetched=-1e9, retry_after=0.0)
        assert shared.read_strategy() == "HIGH_PRESS"   # last known value
    n = fake.gets
    shared._state["fetched"] = -1e9                     # backoff: no new call yet
    assert shared.read_strategy() == "HIGH_PRESS" and fake.gets == n
    fake.exc = None
    fake.value = "not json"
    shared._state.update(fetched=-1e9, retry_after=0.0)
    assert shared.read_strategy() == "HIGH_PRESS"
    fake.value = rec("NOT_A_PRESET")
    shared._state.update(fetched=-1e9, retry_after=0.0)
    assert shared.read_strategy() == "HIGH_PRESS"


def test_shared_slow_client_is_bounded_by_config():
    import botocore.config  # noqa: F401
    shared.set_client(None)
    try:
        c = shared._get_client()
        cfg = c.meta.config
        assert cfg.connect_timeout <= 0.5 and cfg.read_timeout <= 0.5
        assert cfg.retries["total_max_attempts"] == 1
    finally:
        shared.set_client(None)


def test_shared_disabled(monkeypatch):
    monkeypatch.setenv("SHARED_STATE", "0")
    fake = FakeSSM(rec("ALL_OUT"))
    shared.set_client(fake)
    assert shared.read_strategy() == "BALANCED"
    assert shared.write_strategy("COUNTER", "x") is False
    assert shared.read_strategy() == "COUNTER" and shared.source() == "local"
    assert fake.gets == 0 and fake.puts == []


def test_shared_write_increments_seq(monkeypatch):
    monkeypatch.setenv("SHARED_STATE", "1")
    fake = FakeSSM(rec("BALANCED", 5))
    shared.set_client(fake)
    shared.read_strategy()
    assert shared.write_strategy("LOW_BLOCK", "leading", by="t") is True
    v = json.loads(fake.puts[-1]["Value"])
    assert v["strategy"] == "LOW_BLOCK" and v["seq"] == 6 and v["by"] == "t"
    assert fake.puts[-1]["Name"] == "/afc/team/strategy" and fake.puts[-1]["Overwrite"] is True
    fake.exc = RuntimeError("throttled")
    assert shared.write_strategy("ALL_OUT", "x") is False
    assert shared.read_strategy() == "ALL_OUT"           # local value still updates


# ---------------------------------------------------------------- match memory

def _tick(tick, t, holder, bx, home=0, away=0):
    """holder: ('home'|'away', idx) or None; ball at (bx, 0)."""
    gs = copy.deepcopy(GAME_STATE)
    gs.update(tick=tick, gameTime=t, score={"home": home, "away": away})
    gs["ball"]["position"] = {"x": bx, "y": 0, "z": 0}
    if holder is None:
        gs["ball"].update(isFree=True, possessionAgentId=None)
    else:
        team, idx = holder
        gs["ball"].update(isFree=False, possessionAgentId=f"agentId_{idx}")
        for p in gs["players"]:
            if p["teamCode"] == team and p["agentId"] == f"agentId_{idx}":
                p["position"] = {"x": bx, "y": 0}
    return gs


def test_memory_detects_goals_and_counter():
    mm = match_memory
    tick = 1
    for i in range(20):                                   # home build-up with 3 holders
        mm.update(_tick(tick, tick * 0.5, ("home", 2 + i // 7), -20 + 3 * i), 0)
        tick += 1
    mm.update(_tick(tick, tick * 0.5, None, 50, home=1), 0)
    tick += 1
    st = mm.stats()
    assert st["goals"] == 1 and st["score"] == (1, 0)
    g = mm.MEMORY.goals[-1]
    assert g["who"] == "us" and g["holder"] == "us" and g["passes"] == 2
    for i in range(5):                                    # we lose it high, they break fast
        mm.update(_tick(tick, tick * 0.5, ("home", 3), 30), 0)
        tick += 1
    for i in range(6):
        mm.update(_tick(tick, tick * 0.5, ("away", 4), 30 - 12 * i), 0)
        tick += 1
    mm.update(_tick(tick, tick * 0.5, None, -50, home=1, away=1), 0)
    g = mm.MEMORY.goals[-1]
    assert g["who"] == "them" and g["counter"] and g["passes"] == 0
    assert mm.stats()["losses"]["att"] >= 1
    rep = mm.report()
    lines = rep.splitlines()
    assert 3 <= len(lines) <= 12 and all(len(l) < 120 for l in lines)
    assert "THEM scored: counter" in rep
    mm.update(_tick(tick + 1, tick * 0.5, None, -50, home=1, away=1), 1)   # other team = reset
    assert mm.stats()["goals"] == 0
    mm.update(_tick(500, 200, None, 0, home=1, away=0), 0)
    mm.update(_tick(3, 1, None, 0), 0)                    # tick goes back -> new match
    assert mm.stats()["goals"] == 0 and mm.stats()["score"] == (0, 0)


def test_memory_joins_mid_match_without_fake_goals():
    match_memory.update(_tick(900, 140, None, 0, home=2, away=3), 0)
    assert match_memory.stats()["goals"] == 0


def test_memory_never_raises():
    match_memory.update({"players": "bad", "ball": 3, "score": "x"}, 0)
    match_memory.update(None, 0)
    assert isinstance(match_memory.report(), str)


# ---------------------------------------------------------------- manager

def _run_trigger(gs, team=0):
    match_memory.update(gs, team)
    why = manager.maybe_trigger(gs, team, "GK")
    th = manager._state["thread"]
    if why and th is not None:
        th.join(5)
    return why


@pytest.fixture
def mgr(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGER", "1")
    calls = []

    def set_llm(answer, delay=0.0):
        def fake(prompt):
            calls.append(prompt)
            if delay:
                time.sleep(delay)
            if isinstance(answer, Exception):
                raise answer
            return answer
        monkeypatch.setattr(manager, "LLM_CALL", fake)

    set_llm('{"strategy": "BALANCED", "reason": "x"}')
    return set_llm, calls


def test_manager_disabled_and_not_gk(monkeypatch):
    assert manager.maybe_trigger(GAME_STATE, 0, "GK") is None     # ENABLE_MANAGER=0 by conftest
    monkeypatch.setenv("ENABLE_MANAGER", "1")
    assert manager.maybe_trigger(GAME_STATE, 0, "MID") is None


def test_manager_valid_answer_and_bad_answers(mgr):
    set_llm, calls = mgr
    assert _run_trigger(_tick(1, 1, None, 0)) == "kickoff"         # kickoff: rules, no LLM
    assert calls == [] and shared.current()["strategy"] == "BALANCED"
    set_llm('Sure! {"strategy": "counter", "reason": "they press high"} ok')
    assert _run_trigger(_tick(2, 40, None, 0)) == "timer"
    assert len(calls) == 1 and "Strategies:" in calls[0] and "COUNTER" in calls[0]
    assert shared.current()["strategy"] == "COUNTER"
    assert shared.current()["by"] == "manager-gk"
    for bad in ("no json here", '{"strategy": "WIN_NOW"}', '{"strategy": 3}', "[1,2]", None):
        set_llm(bad)
        t = 40 + 31 * (len(calls))
        assert _run_trigger(_tick(len(calls) + 10, t, None, 0)) == "timer"
        assert shared.current()["strategy"] == "COUNTER", bad


def test_manager_hysteresis_and_goal_bypass(mgr):
    set_llm, calls = mgr
    _run_trigger(_tick(1, 1, None, 0))
    set_llm('{"strategy": "HIGH_PRESS", "reason": "a"}')
    _run_trigger(_tick(2, 31, None, 0))
    assert shared.current()["strategy"] == "HIGH_PRESS"
    manager._state["last_t"] = 0                                    # force the timer early
    set_llm('{"strategy": "POSSESSION", "reason": "b"}')
    _run_trigger(_tick(3, 40, None, 0))                             # 9 s after switch: skip
    assert shared.current()["strategy"] == "HIGH_PRESS"
    assert "hysteresis" in manager._state["last_result"]
    assert _run_trigger(_tick(4, 42, None, 0, away=1)) == "goal"    # a goal bypasses it
    assert shared.current()["strategy"] == "POSSESSION"


def test_manager_rules_when_llm_unavailable(mgr):
    set_llm, _ = mgr
    _run_trigger(_tick(1, 1, None, 0))
    set_llm(RuntimeError("AccessDenied"))
    _run_trigger(_tick(2, 160, None, 0, home=0, away=1))
    assert shared.current()["strategy"] == "ALL_OUT"
    assert "rules" in shared.current()["reason"]
    st = {"score": (2, 1), "t": 250, "conceded_counters": 0}
    assert manager.rules(st)[0] == "LOW_BLOCK"
    assert manager.rules({"score": (1, 2), "t": 100, "conceded_counters": 2})[0] == "LOW_BLOCK"
    assert manager.rules({"score": (1, 1), "t": 100, "conceded_counters": 0}) is None


def test_manager_never_blocks_and_late_is_ignored(mgr, monkeypatch):
    set_llm, _ = mgr
    _run_trigger(_tick(1, 1, None, 0))
    monkeypatch.setattr(manager, "LATE_S", 0.05)
    set_llm('{"strategy": "ALL_OUT", "reason": "late"}', delay=0.3)
    gs = _tick(2, 40, None, 0)
    match_memory.update(gs, 0)
    t0 = time.perf_counter()
    assert manager.maybe_trigger(gs, 0, "GK") == "timer"
    assert (time.perf_counter() - t0) < 0.05                        # returns at once
    assert manager.maybe_trigger(_tick(3, 80, None, 0), 0, "GK") is None   # one thread at a time
    manager._state["thread"].join(5)
    assert shared.current()["strategy"] == "BALANCED"
    assert manager._state["last_result"] == "llm late"


def test_parse_answer():
    assert manager.parse_answer('{"strategy":"LOW_BLOCK","reason":"r"}') == ("LOW_BLOCK", "r")
    assert manager.parse_answer('```json\n{"strategy": "all out"}\n```') == ("ALL_OUT", "")
    assert manager.parse_answer("{bad") is None


# ---------------------------------------------------------------- orders

@pytest.mark.parametrize("text,want", [
    ("cuonter", "counter"), ("posession", "possession"), ("hihgpress", "highpress"),
    ("alout", "allout"), ("high press", "highpress"), ("low block", "lowblock"),
    ("park the bus", "lowblock"), ("all out!", "allout"), ("balanced", "balanced"),
    ("press", "press"), ("pres", "press"), ("pas", "pass"), ("mark", "mark"), ("hello", None),
])
def test_tactic_order_words(text, want):
    assert orders.match_order(text) == want


def test_strategy_for():
    assert orders.strategy_for("highpress") == "HIGH_PRESS"
    assert orders.strategy_for("press") is None and orders.strategy_for(None) is None
    for w, name in orders.STRATEGY_WORDS.items():
        assert name in strategies.PRESETS


# ---------------------------------------------------------------- handler

class Ctx:
    session_id = "sess-123"


def _handler(pid, label, monkeypatch):
    monkeypatch.delenv("ENABLE_LLM", raising=False)
    import agent_base
    from fallback import FallbackConfig

    class BoomAgent:
        def __init__(self, *a, **k):
            raise AssertionError("LLM must not be called")

    monkeypatch.setattr(agent_base, "Agent", BoomAgent)
    return agent_base, agent_base.create_invoke_handler(
        _FakeApp(), object(), pid, label, lambda gs, t, p: [{"commandType": "SET_STANCE"}], FallbackConfig())


def _call(handler, gs, team, pid, ctx, capsys):
    payload = {"prompt": json.dumps({"gameState": gs, "teamId": team, "myPlayers": [pid]})}

    async def run():
        return [x async for x in handler(payload, ctx)]

    out = asyncio.run(run())
    recs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    return json.loads(out[0]), recs[-1]


def test_handler_logs_shared_strategy(monkeypatch, capsys):
    import agent_base as _ab
    monkeypatch.setattr(_ab, "team_flag", lambda key, default: True)   # features on
    monkeypatch.setenv("SHARED_STATE", "1")
    fake = FakeSSM(rec("HIGH_PRESS"))
    shared.set_client(fake)
    ab, h = _handler(2, "MID", monkeypatch)
    ab._CTX_LOGGED = False
    cmds, r = _call(h, GAME_STATE, 1, 2, Ctx(), capsys)
    assert cmds[0]["teamId"] == 1 and cmds[0]["playerId"] == 2
    assert r["source"] == "policy" and r["strategy"] == "HIGH_PRESS" and r["strategy_source"] == "shared"
    assert r["session_id"] == "sess-123" and r["ctx_type"] == "Ctx"
    _, r2 = _call(h, dict(GAME_STATE, tick=151), 1, 2, Ctx(), capsys)
    assert "ctx_type" not in r2 and fake.gets == 1                 # cached


def test_handler_coach_order_overrides(monkeypatch, capsys):
    monkeypatch.setenv("SHARED_STATE", "1")
    shared.set_client(FakeSSM(rec("ALL_OUT")))
    _, h = _handler(1, "DEF", monkeypatch)
    gs = dict(GAME_STATE, teamChat=[{"text": "cuonter", "teamId": 0}])
    _, r = _call(h, gs, 0, 1, None, capsys)
    assert r["source"] == "policy" and r["strategy"] == "COUNTER" and r["strategy_source"] == "coach"
    assert r["order"] is None and r["session_id"] is None


def test_handler_default_source_and_fallback_to_plain_policy(monkeypatch, capsys):
    ab, h = _handler(3, "FWD1", monkeypatch)
    _, r = _call(h, GAME_STATE, 0, 3, None, capsys)
    assert r["strategy"] == "BALANCED" and r["strategy_source"] == "default"

    def boom(*a, **k):
        raise RuntimeError("strategy bug")

    monkeypatch.setattr(ab._strategies, "apply", boom)
    cmds, r = _call(h, dict(GAME_STATE, tick=151), 0, 3, None, capsys)
    assert r["source"] == "policy" and "strategy bug" in r["strategy_error"]
    assert cmds[0]["playerId"] == 3


def test_handler_gk_triggers_manager_without_waiting(monkeypatch, capsys):
    import agent_base as _ab
    monkeypatch.setattr(_ab, "team_flag", lambda key, default: True)   # features on
    monkeypatch.setenv("ENABLE_MANAGER", "1")
    monkeypatch.setattr(manager, "LLM_CALL", lambda p: (time.sleep(0.5), '{"strategy":"COUNTER"}')[1])
    _, h = _handler(0, "GK", monkeypatch)
    t0 = time.perf_counter()
    _, r = _call(h, dict(GAME_STATE, gameTime=5.0), 0, 0, None, capsys)
    assert r["manager_trigger"] == "kickoff"
    _, r = _call(h, dict(GAME_STATE, tick=151, gameTime=40.0), 0, 0, None, capsys)
    assert r["manager_trigger"] in ("timer", None)
    assert time.perf_counter() - t0 < 0.4                          # the LLM sleep did not block
    if manager._state["thread"] is not None:
        manager._state["thread"].join(5)
