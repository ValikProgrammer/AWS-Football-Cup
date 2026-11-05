"""Controller factories for the simulator.

A factory is called as factory(team_id, seed) and returns
fn(payload, team_id, pid) -> [command dict], where payload is
{"gameState": {...}, "teamId": t, "myPlayers": [pid]}.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(HERE, "..", "lib"))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

ROLES = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}


def _gs(payload):
    return payload.get("gameState", payload) if isinstance(payload, dict) else {}


def _cmd(ct, pid, tid, params, duration=0):
    return {"commandType": ct, "playerId": pid, "teamId": tid, "parameters": params, "duration": duration}


def _field(gs):
    """Rough half-length from positions (baselines should work on any scale)."""
    xs = [abs(p["position"]["x"]) for p in gs.get("players", [])]
    return max(max(xs, default=1.0) * 1.08, 1e-3)


def _split(gs, team_id):
    code = "home" if team_id == 0 else "away"
    mates = {int(p["agentId"].rsplit("_", 1)[-1]): p for p in gs["players"] if p["teamCode"] == code}
    opps = {int(p["agentId"].rsplit("_", 1)[-1]): p for p in gs["players"] if p["teamCode"] != code}
    return mates, opps


def _holder(gs, team_id):
    """Return ("me"|"mate"|"opp"|None, holder_pid)."""
    ball = gs["ball"]
    if ball.get("isFree") or not ball.get("possessionAgentId"):
        return None, None
    pid = int(ball["possessionAgentId"].rsplit("_", 1)[-1])
    bp = ball["position"]
    cands = [p for p in gs["players"] if p["agentId"] == ball["possessionAgentId"]]
    cands.sort(key=lambda p: (p["position"]["x"] - bp["x"]) ** 2 + (p["position"]["y"] - bp["y"]) ** 2)
    if not cands:
        return None, None
    mine = cands[0]["teamCode"] == ("home" if team_id == 0 else "away")
    return ("mate" if mine else "opp"), pid


# ---------------------------------------------------------------- stock
def stock(team_id, seed):
    from fallback import build_fallback, GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG
    fns = {i: build_fallback(c) for i, c in enumerate([GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG])}

    def fn(payload, tid, pid):
        return fns[pid](_gs(payload), tid, pid)
    return fn


# ---------------------------------------------------------------- aggressive
def aggressive(team_id, seed):
    def fn(payload, tid, pid):
        gs = _gs(payload)
        L = _field(gs)
        d = 1.0 if tid == 0 else -1.0
        mates, opps = _split(gs, tid)
        me = mates[pid]["position"]
        who, hp = _holder(gs, tid)
        if who == "mate" and hp == pid:
            if pid == 0:
                fwd = max((q for q in mates if q != 0), key=lambda q: mates[q]["position"]["x"] * d)
                return [_cmd("GK_DISTRIBUTE", pid, tid, {"target_player_id": fwd, "method": "KICK"})]
            if abs(d * L - me["x"]) < 0.65 * L:
                return [_cmd("SHOOT", pid, tid, {"aim_location": "BL" if pid % 2 else "TR", "power": 1.0})]
            return [_cmd("MOVE_TO", pid, tid, {"target_x": d * L * 0.8, "target_y": 0, "sprint": True})]
        if who == "mate":
            return [_cmd("MOVE_TO", pid, tid, {"target_x": d * L * 0.6, "target_y": (pid - 2.5) * 0.15 * L,
                                                "sprint": True})]
        if pid == 0:
            return [_cmd("CLEAR_OVERRIDE", pid, tid, {})]
        return [_cmd("PRESS_BALL", pid, tid, {"intensity": 1.0}, 3)]
    return fn


# ---------------------------------------------------------------- defensive
def defensive(team_id, seed):
    def fn(payload, tid, pid):
        gs = _gs(payload)
        L = _field(gs)
        d = 1.0 if tid == 0 else -1.0
        mates, opps = _split(gs, tid)
        me = mates[pid]["position"]
        bp = gs["ball"]["position"]
        who, hp = _holder(gs, tid)
        if who == "mate" and hp == pid:
            if pid == 0:
                return [_cmd("GK_DISTRIBUTE", pid, tid, {"target_player_id": 4, "method": "KICK"})]
            if pid in (3, 4) and abs(d * L - me["x"]) < 0.4 * L:
                return [_cmd("SHOOT", pid, tid, {"aim_location": "BR", "power": 0.85})]
            far = max((q for q in mates if q not in (0, pid)), key=lambda q: mates[q]["position"]["x"] * d)
            return [_cmd("PASS", pid, tid, {"target_player_id": far, "type": "AERIAL"})]
        if pid == 0:
            return [_cmd("CLEAR_OVERRIDE", pid, tid, {})]
        if pid in (3, 4):
            return [_cmd("MOVE_TO", pid, tid, {"target_x": d * L * 0.15, "target_y": (pid - 3.5) * 0.4 * L,
                                                "sprint": False})]
        # deep block: DEF + MID mark the two most advanced opponents, near ball -> press
        if who is None or who == "opp":
            dball = math.hypot(me["x"] - bp["x"], me["y"] - bp["y"])
            if dball < 0.12 * L and (bp["x"] * d) < -0.3 * L:
                return [_cmd("PRESS_BALL", pid, tid, {"intensity": 0.8}, 3)]
            danger = sorted((q for q in opps if q != 0), key=lambda q: opps[q]["position"]["x"] * d)
            tgt = danger[pid - 1] if len(danger) >= pid else danger[0]
            if opps[tgt]["position"]["x"] * d < 0:
                return [_cmd("MARK", pid, tid, {"target_player_id": tgt, "tightness": "TIGHT"}, 3)]
        return [_cmd("MOVE_TO", pid, tid, {"target_x": -d * L * (0.7 if pid == 1 else 0.5),
                                            "target_y": bp["y"] * 0.3, "sprint": False})]
    return fn


# ---------------------------------------------------------------- random_legal
def random_legal(team_id, seed):
    rng = random.Random(seed * 31 + team_id)
    types = ["MOVE_TO", "PASS", "SHOOT", "SLIDE_TACKLE", "PRESS_BALL", "MARK", "INTERCEPT",
             "FOLLOW_PLAYER", "SET_STANCE", "CLEAR_OVERRIDE", "GK_DISTRIBUTE"]

    def fn(payload, tid, pid):
        gs = _gs(payload)
        L = _field(gs)
        ct = rng.choice(types)
        p = {}
        if ct == "MOVE_TO":
            p = {"target_x": rng.uniform(-L, L), "target_y": rng.uniform(-0.6 * L, 0.6 * L), "sprint": rng.random() < 0.5}
        elif ct == "PASS":
            p = {"target_player_id": rng.randrange(5), "type": rng.choice(["GROUND", "AERIAL", "THROUGH"])}
        elif ct == "SHOOT":
            p = {"aim_location": rng.choice(["TL", "TR", "BL", "BR", "CENTER"]), "power": rng.random()}
        elif ct in ("SLIDE_TACKLE", "MARK"):
            p = {"target_player_id": rng.randrange(5)}
        elif ct == "PRESS_BALL":
            p = {"intensity": rng.random()}
        elif ct == "INTERCEPT":
            p = {"aggressive": rng.random() < 0.5}
        elif ct == "FOLLOW_PLAYER":
            p = {"target_player_id": rng.randrange(5), "target_team": rng.choice(["HOME", "AWAY"]), "distance": 3.0}
        elif ct == "SET_STANCE":
            p = {"stance": rng.randrange(3)}
        elif ct == "GK_DISTRIBUTE":
            p = {"target_player_id": rng.randrange(1, 5), "method": rng.choice(["THROW", "KICK"])}
        return [_cmd(ct, pid, tid, p)]
    return fn


# ---------------------------------------------------------------- policy
_POLICY = None


def policy_available():
    global _POLICY
    if _POLICY is None:
        try:
            import policy as _p  # noqa
            _POLICY = _p
        except Exception as e:  # module missing or broken
            _POLICY = False
            print(f"[baselines] policy unavailable: {e!r}", file=sys.stderr)
    return bool(_POLICY)


def load_params(path=None):
    with open(path or os.path.join(LIB_DIR, "params.json")) as f:
        return json.load(f)


class PolicyFactory:
    """Picklable factory: policy(params)(team_id, seed) -> fn."""

    def __init__(self, params=None):
        self.params = dict(params) if params is not None else load_params()

    def __call__(self, team_id, seed):
        if not policy_available():
            return stock(team_id, seed)
        decide = _POLICY.decide
        params = self.params

        def fn(payload, tid, pid):
            cands = decide(_gs(payload), tid, pid, ROLES[pid], params, None)
            return [cands[0].command] if cands else []
        return fn


def policy(params=None):
    return PolicyFactory(params)


FACTORIES = {"stock": stock, "aggressive": aggressive, "defensive": defensive, "random_legal": random_legal}
