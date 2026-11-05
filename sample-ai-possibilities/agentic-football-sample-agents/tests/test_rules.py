"""Hard rules: open goal -> SHOOT CENTER; close presser -> SLIDE_TACKLE."""
import copy

import norm
import policy
from test_helpers import GAME_STATE

P = policy.load_params()


def _state(ball_xy, holder, players):
    gs = copy.deepcopy(GAME_STATE)
    gs["ball"]["position"] = {"x": ball_xy[0], "y": ball_xy[1], "z": 0}
    gs["ball"]["possessionAgentId"] = holder
    gs["ball"]["isFree"] = False
    for p in gs["players"]:
        key = (p["teamCode"], p["agentId"])
        if key in players:
            p["position"] = {"x": players[key][0], "y": players[key][1]}
    return gs


def _top(gs, team, pid, role):
    norm.reset()
    return policy.decide(gs, team, pid, role, P)[0]


def test_open_goal_shoots_center():
    gs = _state((40, 0), "agentId_3", {("home", "agentId_3"): (40, 0), ("away", "agentId_0"): (52, 25),
                                       ("away", "agentId_1"): (20, -20), ("away", "agentId_4"): (35, 20)})
    c = _top(gs, 0, 3, "FWD1")
    assert c.command["commandType"] == "SHOOT", c.reason
    assert c.command["parameters"]["aim_location"] == "CENTER"
    assert c.reason.startswith("open goal")


def test_gk_on_line_is_not_open_goal():
    gs = _state((40, 0), "agentId_3", {("home", "agentId_3"): (40, 0), ("away", "agentId_0"): (53, 0)})
    c = _top(gs, 0, 3, "FWD1")
    assert not c.reason.startswith("open goal"), c.reason


def test_no_open_goal_shot_from_own_half():
    gs = _state((-10, 0), "agentId_2", {("home", "agentId_2"): (-10, 0), ("away", "agentId_0"): (52, 30),
                                        ("away", "agentId_1"): (-30, 20), ("away", "agentId_2"): (-20, -25),
                                        ("away", "agentId_3"): (-40, 10), ("away", "agentId_4"): (-35, -10)})
    c = _top(gs, 0, 2, "MID")
    assert not c.reason.startswith("open goal"), c.reason


def test_open_goal_mirrored_for_away_team():
    # away attacks -x: same picture mirrored
    gs = _state((-40, 0), "agentId_3", {("away", "agentId_3"): (-40, 0), ("home", "agentId_0"): (-52, 25),
                                        ("home", "agentId_1"): (-20, 20), ("home", "agentId_4"): (-35, -20),
                                        ("home", "agentId_3"): (-14, -5)})
    c = _top(gs, 1, 3, "FWD1")
    assert c.command["parameters"].get("aim_location") == "CENTER", c.reason


def test_tackles_off_by_default():
    gs = _state((10, 5), "agentId_2", {("away", "agentId_2"): (10, 5), ("home", "agentId_2"): (8.5, 4)})
    assert P["tackle_dist"] < 0
    assert _top(gs, 0, 2, "MID").command["commandType"] != "SLIDE_TACKLE"


def test_close_presser_slide_tackles_carrier():
    # tackles enabled: away MID (agentId_2) carries the ball at (10, 5); our MID 2 m away
    gs = _state((10, 5), "agentId_2", {("away", "agentId_2"): (10, 5), ("home", "agentId_2"): (8.5, 4)})
    norm.reset()
    c = policy.decide(gs, 0, 2, "MID", {**P, "tackle_dist": 0.07})[0]
    assert c.command["commandType"] == "SLIDE_TACKLE", c.reason
    assert c.command["parameters"]["target_player_id"] == 2
    assert c.command["parameters"]["distance"] > 0


def test_far_presser_does_not_tackle():
    gs = _state((10, 5), "agentId_2", {("away", "agentId_2"): (10, 5), ("home", "agentId_2"): (-5, 4)})
    c = _top(gs, 0, 2, "MID")
    assert c.command["commandType"] != "SLIDE_TACKLE", c.reason
