"""Coach C1/C6 force the carrier; 'nothing needed' decisions go to the engine within a cap."""
import glob
import json
import os

import agent_base
import norm
import policy
from policy import Candidate
from test_rules import _state, P

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_coach_shoot_makes_carrier_shoot_in_their_half():
    # FWD1 in their half, corridor blocked by a defender -> normally no shot
    gs = _state((25, 3), "agentId_3", {("home", "agentId_3"): (25, 3), ("away", "agentId_1"): (32, 3),
                                       ("away", "agentId_0"): (53, 0)})
    norm.reset()
    c = policy.decide(gs, 0, 3, "FWD1", P, "shoot")[0]
    assert c.command["commandType"] == "SHOOT" and c.reason.startswith("coach shoot"), c.reason
    assert c.command["parameters"]["aim_location"] in ("CENTER", "BL", "BR")


def test_coach_shoot_not_from_own_half():
    gs = _state((-20, 0), "agentId_2", {("home", "agentId_2"): (-20, 0)})
    norm.reset()
    assert not policy.decide(gs, 0, 2, "MID", P, "shoot")[0].reason.startswith("coach shoot")


def test_coach_pass_makes_carrier_pass():
    gs = _state((14, -5), "agentId_3", {("home", "agentId_3"): (14, -5)})
    norm.reset()
    c = policy.decide(gs, 0, 3, "FWD1", P, "pass")[0]
    assert c.command["commandType"] == "PASS", c.reason


def _cand(ok=True):
    return [Candidate({"commandType": "MOVE_TO", "parameters": {"target_x": 1, "target_y": 1, "sprint": False}},
                      0.4, "shape", ok)]


def test_engine_share_capped_from_first_tick():
    agent_base._ENGINE_RECENT.clear()
    first5 = [agent_base._pick_command(_cand(), "MID")[1] for _ in range(5)]
    assert sum(first5) <= 2                       # 3 of 5 stay active: fitness check passes
    more = [agent_base._pick_command(_cand(), "MID")[1] for _ in range(10)]
    assert sum(more) <= 4


def test_gk_and_non_engine_candidates_never_cleared():
    agent_base._ENGINE_RECENT.clear()
    assert not any(agent_base._pick_command(_cand(), "GK")[1] for _ in range(6))
    agent_base._ENGINE_RECENT.clear()
    assert not any(agent_base._pick_command(_cand(ok=False), "MID")[1] for _ in range(6))


def test_fitness_check_on_real_scrimmage_ticks():
    pls = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(FX, "real_def_*.json")))]
    for pid, role in {1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}.items():
        agent_base._ENGINE_RECENT.clear()
        norm.reset()
        active = 0
        for pl in pls[-5:]:
            cmd, _ = agent_base._pick_command(policy.decide(pl["gameState"], pl["teamId"], pid, role, P), role)
            active += cmd["commandType"] not in ("CLEAR_OVERRIDE", "SET_STANCE", "RESET")
        assert active > 2.5, (role, active)
