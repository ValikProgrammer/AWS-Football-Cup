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
    assert c.command["commandType"] == "SHOOT" and c.reason.startswith("coach C1"), c.reason
    assert c.command["parameters"]["aim_location"] in ("CENTER", "BL", "BR")


def test_coach_c1_shoots_from_anywhere():
    # own half and from beside the post: C1 still shoots (coach's choice)
    for ball, holder, pos in (((-20, 0), "agentId_2", (-20, 0)), ((50, 25), "agentId_3", (50, 25))):
        gs = _state(ball, holder, {("home", holder): pos})
        norm.reset()
        c = policy.decide(gs, 0, int(holder[-1]), "MID" if holder.endswith("2") else "FWD1", P, "shoot")[0]
        assert c.command["commandType"] == "SHOOT" and c.command["parameters"]["aim_location"] == "CENTER", c.reason


def test_coach_c1_gk_still_distributes():
    gs = _state((-50, 0), "agentId_0", {("home", "agentId_0"): (-50, 0)})
    norm.reset()
    assert policy.decide(gs, 0, 0, "GK", P, "shoot")[0].command["commandType"] == "GK_DISTRIBUTE"


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


def _dribble():
    return [Candidate({"commandType": "MOVE_TO", "parameters": {"target_x": 1, "target_y": 1, "sprint": True}},
                      0.3, "dribble fwd space=0.30", True)]


def test_carrier_priority_still_passes_fitness_check():
    agent_base._ENGINE_RECENT.clear()
    first5 = [agent_base._pick_command(_dribble(), "FWD1")[1] for _ in range(5)]
    assert sum(first5) <= 2
    agent_base._ENGINE_RECENT.clear()
    many = [agent_base._pick_command(_dribble(), "FWD1")[1] for _ in range(20)]
    assert sum(many[-10:]) <= 5 and sum(many) / len(many) < 0.5


def test_dribble_in_their_half_is_engine_ok():
    gs = _state((22, 0), "agentId_3", {("home", "agentId_3"): (22, 0), ("away", "agentId_1"): (30, 0)})
    norm.reset()
    cands = policy.decide(gs, 0, 3, "FWD1", P)
    drib = [c for c in cands if c.reason.startswith("dribble")]
    assert drib and all(c.engine_ok for c in drib)        # x = 22/55 = 0.4 > 0.3


def test_s7_code_maps_to_shoot_on_sight():
    import orders
    assert orders.match_order("S7") == "shootonsight"
    assert orders.match_order("ы7") == "shootonsight"
    assert orders.strategy_for("shootonsight") == "SHOOT_ON_SIGHT"


def test_shoot_on_sight_carrier_in_their_half_shoots_center():
    import strategies
    params = strategies.apply(P, "SHOOT_ON_SIGHT")
    # FWD1 in their half, corridor blocked by a defender: BALANCED would not shoot
    gs = _state((25, 3), "agentId_3", {("home", "agentId_3"): (25, 3), ("away", "agentId_1"): (32, 3),
                                       ("away", "agentId_0"): (53, 0)})
    norm.reset()
    assert policy.decide(gs, 0, 3, "FWD1", P)[0].command["commandType"] != "SHOOT"
    norm.reset()
    c = policy.decide(gs, 0, 3, "FWD1", params)[0].command
    assert c["commandType"] == "SHOOT" and c["parameters"]["aim_location"] == "CENTER"


def test_shoot_on_sight_not_from_own_half():
    import strategies
    params = strategies.apply(P, "SHOOT_ON_SIGHT")
    gs = _state((-20, 0), "agentId_2", {("home", "agentId_2"): (-20, 0)})
    norm.reset()
    assert policy.decide(gs, 0, 2, "MID", params)[0].command["commandType"] != "SHOOT"
