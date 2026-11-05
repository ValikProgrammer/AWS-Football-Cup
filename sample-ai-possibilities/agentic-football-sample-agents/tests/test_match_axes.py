"""Real MATCH payloads: ball axes, field scale and shooting.

In matches the ball is {x, y=height, z=lateral}; players are {x, y=lateral}.
Scrimmages and test_helpers.GAME_STATE send the ball as {x, y=lateral, z=0}.
Fixtures tests/fixtures/match_*.json are payloads copied from a real match
(we were team 0, home, attacking +x; field about x +-6.4, y +-3.6).
"""

import asyncio
import copy
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from test_helpers import mock_agentcore, GAME_STATE, _FakeApp  # noqa: E402
mock_agentcore()

import norm  # noqa: E402
import policy  # noqa: E402
import state  # noqa: E402

FX = os.path.join(os.path.dirname(__file__), "fixtures")
P = policy.load_params()
ROLES = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}


def _load(name):
    with open(os.path.join(FX, name)) as f:
        return json.load(f)


HOLD = "match_hold_t4852.json"     # home agentId_3 dribbles; away agentId_3 is 1.1 m away
SHOT = "match_shot_t6153.json"     # home agentId_3 beside the post, real shot tick
FREE = "match_free_t6052.json"     # loose ball behind the goal line


def _player(gs, team, idx):
    return next(p for p in gs["players"] if p["teamCode"] == team and p["agentId"] == f"agentId_{idx}")


# ------------------------------------------------------------------ canonical_ball

def test_game_state_unchanged():
    before = copy.deepcopy(GAME_STATE)
    norm.reset()
    out = norm.canonical_ball(GAME_STATE)
    assert out == before and GAME_STATE == before
    assert not norm._state["z_lateral"]


def test_scrimmage_fixtures_unchanged():
    for f in sorted(glob.glob(os.path.join(FX, "real_*.json"))):
        gs = json.load(open(f))["gameState"]
        norm.reset()
        assert norm.canonical_ball(gs) == gs, f


def test_match_lateral_from_z():
    raw = _load(HOLD)["gameState"]
    keep = copy.deepcopy(raw)
    norm.reset()
    c = norm.canonical_ball(raw)
    assert raw == keep                          # input not mutated
    rp, cp = raw["ball"]["position"], c["ball"]["position"]
    assert cp["x"] == rp["x"] and cp["y"] == rp["z"] and cp["z"] == rp["y"]
    rv, cv = raw["ball"]["velocity"], c["ball"]["velocity"]
    assert cv["y"] == rv["z"] and cv["z"] == rv["y"]
    assert norm._state["z_lateral"]             # |z| = 2.49 > 0.5: locked


def test_idempotent():
    for gs in (GAME_STATE, _load(HOLD)["gameState"], _load(SHOT)["gameState"], _load(FREE)["gameState"]):
        norm.reset()
        a = norm.canonical_ball(gs)
        b = norm.canonical_ball(a)
        assert a == b
        norm.reset()                            # also with a fresh lock
        assert norm.canonical_ball(a) == a


def test_ball_without_position():
    gs = copy.deepcopy(_load(FREE)["gameState"])
    del gs["ball"]["position"]
    norm.reset()
    assert norm.canonical_ball(gs) is gs
    for pid, role in ROLES.items():
        assert policy.decide(gs, 0, pid, role, P)


def test_lock_holds_when_y_looks_nearer():
    """After the lock, z stays lateral even when (x, y=height) is nearer a player."""
    norm.reset()
    norm.canonical_ball(_load(HOLD)["gameState"])
    gs = copy.deepcopy(_load(SHOT)["gameState"])
    me = _player(gs, "home", 3)
    gs["ball"]["position"] = {"x": me["position"]["x"], "y": me["position"]["y"], "z": 3.0}
    assert norm.canonical_ball(gs)["ball"]["position"]["y"] == 3.0


# ------------------------------------------------------------------ holder

def test_holder_is_me_after_canonical():
    raw = _load(HOLD)["gameState"]
    norm.reset()
    c = norm.canonical_ball(raw)
    b, pl = c["ball"], c["players"]
    assert state.holder_is_me(b, pl, 0, 3)          # true holder: home agentId_3
    assert not state.holder_is_me(b, pl, 1, 3)      # same index, other team
    # the bug this fixes: with y = height the opponent looked nearer
    assert state.holder_is_me(raw["ball"], raw["players"], 1, 3)


def test_free_ball_fixture():
    raw = _load(FREE)["gameState"]
    norm.reset()
    c = norm.canonical_ball(raw)
    assert c["ball"]["position"]["y"] == raw["ball"]["position"]["z"]
    assert state.get_possession_info(c["ball"], c["players"], 0)[1] == "free"
    for pid, role in ROLES.items():
        cmd = policy.decide(raw, 0, pid, role, P)[0].command
        assert cmd["commandType"] not in ("SHOOT", "PASS"), cmd


def test_handler_uses_canonical_ball(monkeypatch, capsys):
    monkeypatch.delenv("ENABLE_LLM", raising=False)
    import agent_base
    from fallback import FallbackConfig

    norm.reset()
    handler = agent_base.create_invoke_handler(
        _FakeApp(), object(), 3, "FWD1", lambda gs, t, p: [{"commandType": "SET_STANCE"}], FallbackConfig())
    payload = {"prompt": json.dumps(_load(HOLD))}

    async def run():
        return [x async for x in handler(payload, None)]

    cmds = json.loads(asyncio.run(run())[0])
    assert cmds[0]["playerId"] == 3 and cmds[0]["teamId"] == 0
    recs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    reason = recs[-1]["reason"]
    # on-ball decision (pass / shoot / dribble), not a defending one
    assert reason.split()[0] in ("pass", "shoot", "dribble", "open", "clearance"), reason


# ------------------------------------------------------------------ field scale

def test_scale_on_match_payload():
    norm.reset()
    sx, sy = norm.update_scale(norm.canonical_ball(_load(SHOT)["gameState"]))
    assert abs(sx - 6.4) <= 0.3 and abs(sy - 3.6) <= 0.3, (sx, sy)


def test_scale_on_match_sequence():
    norm.reset()
    for name in (HOLD, FREE, SHOT):
        sx, sy = norm.update_scale(norm.canonical_ball(_load(name)["gameState"]))
    assert abs(sx - 6.4) <= 0.3 and abs(sy - 3.6) <= 0.3, (sx, sy)


def test_scale_ignores_ball_behind_line():
    norm.reset()
    gs = norm.canonical_ball(_load(FREE)["gameState"])
    assert gs["ball"]["position"]["x"] > 6.45          # ball behind the goal line
    sx, _ = norm.update_scale(gs)
    assert sx <= 6.7, sx


def test_scale_on_scrimmage():
    norm.reset()
    sx, _ = norm.update_scale(norm.canonical_ball(GAME_STATE))
    assert 50 <= sx <= 55, sx
    norm.reset()
    for f in sorted(glob.glob(os.path.join(FX, "real_*.json"))):
        sx, _ = norm.update_scale(norm.canonical_ball(json.load(open(f))["gameState"]))
    assert 50 <= sx <= 55, sx


# ------------------------------------------------------------------ shooting

def _shot_state(me_xy, gk_xy, clear=True):
    """SHOT fixture with home agentId_3 holding at me_xy and the away GK at gk_xy (raw)."""
    gs = copy.deepcopy(_load(SHOT)["gameState"])
    _player(gs, "home", 3)["position"] = {"x": me_xy[0], "y": me_xy[1]}
    _player(gs, "away", 0)["position"] = {"x": gk_xy[0], "y": gk_xy[1]}
    if clear:   # other opponents and our mates far from the shot
        for i, y in zip((1, 2, 3, 4), (-3.0, -2.0, 2.0, 3.0)):
            _player(gs, "away", i)["position"] = {"x": -3.0, "y": y}
        for i, y in zip((1, 2, 4), (-1.0, 1.0, -2.5)):
            _player(gs, "home", i)["position"] = {"x": -2.0, "y": y}
    gs["ball"]["position"] = {"x": me_xy[0] + 0.1, "y": 0.143, "z": me_xy[1]}
    gs["ball"]["possessionAgentId"] = "agentId_3"
    gs["ball"]["isFree"] = False
    return gs


def _decide(gs):
    norm.reset()
    policy.decide(_load(SHOT)["gameState"], 0, 3, "FWD1", P)   # warm scale + axis lock
    return policy.decide(gs, 0, 3, "FWD1", P)


def _shots(cands):
    return [c for c in cands if c.command["commandType"] == "SHOOT"]


def test_shot_angle():
    assert policy.shot_angle_deg((0.375, 0.0), 0.155) > 25
    assert policy.shot_angle_deg((0.984, 0.25), 0.155) < 12


def test_no_shot_beside_post():
    # lateral 1.6 (outside the goal half-width 1.0), 0.15 from the line; GK off its line
    cands = _decide(_shot_state((6.3, 1.6), (3.0, -2.5)))
    assert not _shots(cands), [c.reason for c in _shots(cands)]


def test_central_shot_gk_off_line_center():
    # 3.8 raw out on the centre line (dg = 0.59 <= gk_off_max_d), GK out at the side
    cands = _decide(_shot_state((2.6, 0.0), (3.5, -2.8)))
    top = cands[0]
    assert top.command["commandType"] == "SHOOT", top.reason
    assert top.command["parameters"] == {"aim_location": "CENTER", "power": 1.0}


def test_far_central_gk_off_line_no_shot():
    # 5 raw out: beyond gk_off_max_d, and not in the normal shoot range
    assert not _shots(_decide(_shot_state((1.4, 0.0), (3.5, -2.8))))


def test_gk_on_centre_line_aims_bottom_corner_away():
    me = (3.8, 0.5)
    # GK just on the -y side (attacker's right) of the shooter-goal line -> BL
    s = _shots(_decide(_shot_state(me, (5.76, 0.0))))
    assert s and s[0].command["parameters"]["aim_location"] == "BL", [c.reason for c in s]
    # GK just on the +y side (attacker's left) -> BR
    s = _shots(_decide(_shot_state(me, (5.76, 0.29))))
    assert s and s[0].command["parameters"]["aim_location"] == "BR", [c.reason for c in s]


def test_gk_aside_aims_center():
    s = _shots(_decide(_shot_state((3.8, 0.5), (5.9, -0.9))))
    assert s and s[0].command["parameters"]["aim_location"] == "CENTER", [c.reason for c in s]


def _all_fixture_states():
    for f in sorted(glob.glob(os.path.join(FX, "*.json"))):
        pl = json.load(open(f))
        yield os.path.basename(f), pl["gameState"]


def _give_ball(gs, team, idx):
    gs = copy.deepcopy(gs)
    me = _player(gs, team, idx)["position"]
    bp = gs["ball"].get("position") or {}
    if "z" in bp and abs(bp.get("z") or 0) > 1e-6:      # match layout: z = lateral
        gs["ball"]["position"] = {"x": me["x"], "y": 0.143, "z": me["y"]}
    else:
        gs["ball"]["position"] = {"x": me["x"], "y": me["y"]}
    gs["ball"]["possessionAgentId"] = f"agentId_{idx}"
    gs["ball"]["isFree"] = False
    return gs


def test_never_top_corners():
    seen = 0
    for name, gs in _all_fixture_states():
        for tid, team in ((0, "home"), (1, "away")):
            for pid, role in ROLES.items():
                for g in (gs, _give_ball(gs, team, pid)):
                    norm.reset()
                    for c in policy.decide(g, tid, pid, role, P):
                        if c.command["commandType"] == "SHOOT":
                            seen += 1
                            aim = c.command["parameters"]["aim_location"]
                            assert aim in ("CENTER", "BL", "BR"), (name, tid, pid, c.reason)
    assert seen > 0
