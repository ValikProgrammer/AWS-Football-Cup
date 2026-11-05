"""Tests for norm, orders, policy and the policy path in agent_base."""

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

import norm  # noqa: E402
import orders  # noqa: E402
import policy  # noqa: E402
from parsing import VALID_COMMANDS  # noqa: E402

PARAMS = policy.load_params()
ROLES = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}


@pytest.fixture(autouse=True)
def _reset():
    norm.reset()
    yield


def scaled(gs, k):
    gs = copy.deepcopy(gs)
    for p in gs["players"]:
        p["position"] = {a: v * k for a, v in p["position"].items()}
    gs["ball"]["position"] = {a: v * k for a, v in gs["ball"]["position"].items()}
    return gs


def mirrored(gs):
    """Same situation seen with the teams swapped (away attacks -x)."""
    gs = copy.deepcopy(gs)
    for p in gs["players"]:
        p["teamCode"] = "away" if p["teamCode"] == "home" else "home"
        p["position"] = {"x": -p["position"]["x"], "y": -p["position"]["y"]}
    b = gs["ball"]["position"]
    gs["ball"]["position"] = {"x": -b["x"], "y": -b["y"], "z": b.get("z", 0)}
    gs["score"] = {"home": gs["score"]["away"], "away": gs["score"]["home"]}
    return gs


@pytest.mark.parametrize("text,want", [
    ("shot", "shoot"), ("shooot", "shoot"), ("atack", "attack"), ("defnd", "defend"),
    ("pres", "press"), ("pas", "pass"), ("SHOOT!", "shoot"), ("mark", "mark"),
    ("hello", None), ("", None),
])
def test_order_typos(text, want):
    assert orders.match_order(text) == want


def test_order_tracker_timer_and_team_filter():
    tr = orders.OrderTracker()
    gs = {"tick": 100, "teamChat": [{"text": "press", "teamId": 0}, {"text": "defend", "teamId": 1}]}
    assert tr.update(gs, 0, 20) == "press"
    gs2 = dict(gs, tick=115)
    assert tr.update(gs2, 0, 20) == "press"          # same message: no restart
    gs3 = dict(gs, tick=125)
    assert tr.update(gs3, 0, 20) is None             # expired
    assert orders.OrderTracker().update({"tick": 1, "teamChat": ["atack"]}, 1, 5) == "attack"


@pytest.mark.parametrize("team_id", [0, 1])
@pytest.mark.parametrize("pid", list(ROLES))
def test_valid_commands_every_role(team_id, pid):
    gs = GAME_STATE if team_id == 0 else mirrored(GAME_STATE)
    cands = policy.decide(gs, team_id, pid, ROLES[pid], PARAMS)
    assert cands
    c = cands[0].command
    assert c["commandType"] in VALID_COMMANDS
    assert c["playerId"] == pid and c["teamId"] == team_id
    assert all(cands[i].score >= cands[i + 1].score for i in range(len(cands) - 1))


@pytest.mark.parametrize("team_id", [0, 1])
def test_mirror_gives_same_command_types(team_id):
    a = [policy.decide(GAME_STATE, 0, p, ROLES[p], PARAMS)[0].command["commandType"] for p in ROLES]
    norm.reset()
    b = [policy.decide(mirrored(GAME_STATE), 1, p, ROLES[p], PARAMS)[0].command["commandType"] for p in ROLES]
    assert a == b


def test_opponent_same_number_holds_ball():
    gs = copy.deepcopy(GAME_STATE)
    # away agentId_3 holds the ball; home agentId_3 is far away
    gs["ball"]["position"] = {"x": 30, "y": -12, "z": 0}
    gs["ball"]["possessionAgentId"] = "agentId_3"
    cands = policy.decide(gs, 0, 3, "FWD1", PARAMS)
    assert cands[0].command["commandType"] not in ("PASS", "SHOOT")
    assert all(c.command["commandType"] not in ("PASS", "SHOOT") for c in cands)


def test_scale_invariance():
    a = [policy.decide(GAME_STATE, 0, p, ROLES[p], PARAMS)[0].command["commandType"] for p in ROLES]
    norm.reset()
    small = scaled(GAME_STATE, 1 / 8)
    b = [policy.decide(small, 0, p, ROLES[p], PARAMS)[0].command["commandType"] for p in ROLES]
    assert a == b


@pytest.mark.parametrize("bx", [30, 5, -20])
def test_def_stays_in_own_half_when_defending(bx):
    gs = copy.deepcopy(GAME_STATE)
    gs["ball"]["possessionAgentId"] = "agentId_4"
    gs["ball"]["isFree"] = False
    gs["players"][9]["position"] = {"x": bx, "y": 5}      # away agentId_4 holds
    gs["ball"]["position"] = {"x": bx, "y": 5, "z": 0}
    gs["players"][1]["position"] = {"x": 10, "y": 0}      # DEF caught high
    for t in (0, 1):
        norm.reset()
        g = gs if t == 0 else mirrored(gs)
        c = policy.decide(g, t, 1, "DEF", PARAMS)[0].command
        assert c["commandType"] == "MOVE_TO", c
        x = c["parameters"]["target_x"]
        assert (x <= 0) if t == 0 else (x >= 0), c


def test_shoot_near_goal_and_pass_far():
    gs = copy.deepcopy(GAME_STATE)
    gs["ball"]["possessionAgentId"] = "agentId_3"
    gs["players"][3]["position"] = {"x": 42, "y": 2}
    gs["players"][6]["position"] = {"x": 10, "y": -3}
    gs["ball"]["position"] = {"x": 42, "y": 2, "z": 0}
    c = policy.decide(gs, 0, 3, "FWD1", PARAMS)[0].command
    assert c["commandType"] == "SHOOT", c


def test_gk_distributes():
    gs = copy.deepcopy(GAME_STATE)
    gs["ball"]["possessionAgentId"] = "agentId_0"
    gs["ball"]["position"] = {"x": -50, "y": 0, "z": 0}
    c = policy.decide(gs, 0, 0, "GK", PARAMS)[0].command
    assert c["commandType"] == "GK_DISTRIBUTE"


def test_timing_under_5ms():
    policy.decide(GAME_STATE, 0, 2, "MID", PARAMS)
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        policy.decide(GAME_STATE, i % 2, i % 5, ROLES[i % 5], PARAMS, "press")
    assert (time.perf_counter() - t0) / n * 1000 < 5.0


def test_handler_policy_source(monkeypatch, capsys):
    monkeypatch.delenv("ENABLE_LLM", raising=False)
    import agent_base
    from fallback import FallbackConfig

    class BoomAgent:
        def __init__(self, *a, **k):
            raise AssertionError("LLM must not be called")

    monkeypatch.setattr(agent_base, "Agent", BoomAgent)
    handler = agent_base.create_invoke_handler(
        _FakeApp(), object(), 2, "MID", lambda gs, t, p: [{"commandType": "SET_STANCE"}], FallbackConfig())
    payload = {"prompt": json.dumps({"gameState": GAME_STATE, "teamId": 1, "myPlayers": [2]})}

    async def run():
        return [x async for x in handler(payload, None)]

    out = asyncio.run(run())
    cmds = json.loads(out[0])
    assert len(cmds) == 1 and cmds[0]["teamId"] == 1 and cmds[0]["playerId"] == 2
    recs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert recs[-1]["source"] == "policy" and "top" in recs[-1]
