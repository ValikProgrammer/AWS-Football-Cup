"""Deterministic utility-based football policy (no network, no LLM).

decide(game_state, team_id, pid, role, params, order) -> list[Candidate]
All geometry runs in the normalized frame from norm.py: +x = opponent goal.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import norm
from norm import dist, corridor, nearest
from state import _player_idx, _is_my_team, _resolve_holder

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Candidate:
    command: dict
    score: float
    reason: str


def load_params(path: str | None = None) -> dict:
    with open(path or os.path.join(_HERE, "params.json")) as f:
        return json.load(f)


def _cmd(ct, pid, tid, params, duration=0):
    return {"commandType": ct, "playerId": pid, "teamId": tid, "parameters": params, "duration": duration}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _apply_context(p: dict, gs: dict, team_id: int, order: str | None) -> dict:
    p = dict(p)
    sc = gs.get("score") or {}
    mine = sc.get("home", 0) if team_id == 0 else sc.get("away", 0)
    theirs = sc.get("away", 0) if team_id == 0 else sc.get("home", 0)
    t = gs.get("gameTime") or 0
    try:
        diff, t = int(mine) - int(theirs), float(t)
    except (TypeError, ValueError):
        diff, t = 0, 0.0
    if diff > 0:
        p["risk_weight"] *= p["lead_risk_mult"]
    elif diff < 0 and t > p["half_time_s"]:
        p["shoot_range"] += p["lose_shot_boost"]
        p["line_height"] += 0.1
    p["early"] = t < p["early_time_s"]
    p["no_fwd_sprint"] = False
    p["use_mark"] = False
    if order == "shoot":
        p["shot_bonus"] += 0.2
        p["shoot_range"] += 0.15
    elif order == "attack":
        p["line_height"] += 0.15
        p["forward_pass_weight"] += 0.15
    elif order == "defend":
        p["line_height"] -= 0.15
        p["no_fwd_sprint"] = True
    elif order == "press":
        p["press_bodies"] = 2
    elif order == "mark":
        p["use_mark"] = True
    elif order == "pass":
        p["pass_bonus"] += 0.15
    return p


def decide(game_state: dict, team_id: int, pid: int, role: str, params: dict,
           order: str | None = None) -> list[Candidate]:
    gs = game_state or {}
    p = _apply_context(params, gs, team_id, order)
    sx, sy = norm.update_scale(gs)
    fr = norm.Frame(sx, sy, team_id)
    players = gs.get("players") or []
    ball = gs.get("ball") or {}

    mates, opps = {}, []
    me_raw = None
    opp_gk = None
    opp_by_pid = []
    for pl in players:
        q = fr.to_norm(pl.get("position"))
        i = _player_idx(pl)
        if _is_my_team(pl, team_id):
            mates[i] = q
            if i == pid:
                me_raw = pl
        else:
            opps.append(q)
            opp_by_pid.append((i, q))
            if i == 0:
                opp_gk = q
    if me_raw is None:
        return [Candidate(_cmd("CLEAR_OVERRIDE", pid, team_id, {}), 0.0, "me not found")]
    me = mates[pid]
    stamina = me_raw.get("stamina", 1.0)
    sprint = bool(isinstance(stamina, (int, float)) and stamina >= p["stamina_sprint_min"])
    b = fr.to_norm(ball.get("position"))

    holder = None if ball.get("isFree") is True else _resolve_holder(ball, players)
    ctx = dict(p=p, fr=fr, pid=pid, tid=team_id, role=role, me=me, mates=mates, opps=opps,
               opp_gk=opp_gk, opp_by_pid=opp_by_pid, ball=b, sprint=sprint)

    if holder is not None and _is_my_team(holder, team_id):
        if _player_idx(holder) == pid:
            cands = _on_ball(ctx)
        else:
            cands = _attack_off_ball(ctx, _player_idx(holder))
    elif holder is not None:
        hq = fr.to_norm(holder.get("position"))
        cands = _defend(ctx, hq, _player_idx(holder))
    else:
        cands = _transition(ctx)
    if not cands:
        cands = [Candidate(_cmd("CLEAR_OVERRIDE", pid, team_id, {}), 0.0, "default")]
    cands.sort(key=lambda c: -c.score)
    return cands


# ---------------------------------------------------------------- ATTACK

def _pass_type(d, corr, recv_open, ahead, p):
    if d > p["aerial_dist"] or corr < p["pass_open_min"]:
        return "AERIAL"
    if ahead > 0.15 and recv_open > p["pass_open_min"]:
        return "THROUGH"
    return "GROUND"


def _on_ball(c) -> list[Candidate]:
    p, fr, pid, tid, me, opps = c["p"], c["fr"], c["pid"], c["tid"], c["me"], c["opps"]
    out = []
    pressure, _ = nearest(me, opps)
    risk_base = p["risk_weight"] * (1.0 - me[0]) / 2.0   # losing it near my goal costs more

    if c["role"] == "GK" or pid == 0:
        for t, q in c["mates"].items():
            if t == pid:
                continue
            d = dist(me, q)
            op, _ = nearest(q, opps)
            corr = corridor(me, q, opps)
            s = 2.0 * min(op, 0.3) + 1.0 * min(corr, 0.2) + 0.2 * (q[0] + 1) / 2
            method = "THROW" if d < 0.5 else "KICK"
            out.append(Candidate(_cmd("GK_DISTRIBUTE", pid, tid, {"target_player_id": t, "method": method}),
                                 s, f"gk_dist {t} open={op:.2f} corr={corr:.2f}"))
        return out

    # SHOOT
    goal = (1.0, 0.0)
    dg = dist(me, goal)
    blockers = [o for o in opps if o is not c["opp_gk"]]
    corr = corridor(me, goal, blockers)
    gk_off_line = c["opp_gk"] is not None and c["opp_gk"][0] < 1.0 - p["gk_far_line"]
    can = (dg <= p["shoot_range"] and corr >= p["shot_corridor"]) or (gk_off_line and me[0] > 0)
    if can:
        gk_y = c["opp_gk"][1] if c["opp_gk"] is not None else 0.0
        side = "R" if gk_y > 0 else "L"   # +y is the attacker's left
        aim = ("B" if dg < 0.3 else "T") + side
        succ = _clamp(1.0 - dg / (p["shoot_range"] + 0.3), 0.1, 0.9) * (0.6 + 0.4 * min(corr / 0.15, 1.0))
        if gk_off_line:
            succ = max(succ, 0.6)
        s = 1.0 * succ + p["shot_bonus"]
        out.append(Candidate(_cmd("SHOOT", pid, tid, {"aim_location": aim, "power": 0.9 if dg < 0.3 else 1.0}),
                             s, f"shoot d={dg:.2f} corr={corr:.2f}"))

    # PASS
    pb = p["pass_bonus"] + (p["pass_bonus_mid"] if c["role"] == "MID" else 0.0)
    for t, q in c["mates"].items():
        if t == pid or t == 0:
            continue
        d = dist(me, q)
        pc = corridor(me, q, opps)
        op, _ = nearest(q, opps)
        succ = _clamp(0.35 + 1.5 * min(pc, op, 0.3) - 0.25 * d, 0.05, 0.95)
        ahead = q[0] - me[0]
        value = 0.3 + p["forward_pass_weight"] * ahead + 0.15 * (q[0] + 1) / 2
        s = value * succ - risk_base * (1 - succ) + pb
        ptype = _pass_type(d, pc, op, ahead, p)
        out.append(Candidate(_cmd("PASS", pid, tid, {"target_player_id": t, "type": ptype}),
                             s, f"pass {t} {ptype} succ={succ:.2f} val={value:.2f}"))

    # DRIBBLE
    st = p["dribble_step"]
    for name, ang in (("fwd", 0.0), ("left", 0.6), ("right", -0.6)):
        tx = me[0] + st * math.cos(ang)
        ty = _clamp(me[1] + st * math.sin(ang), -fr.half_w * 0.9, fr.half_w * 0.9)
        space, _ = nearest((tx, ty), opps)
        succ = _clamp(space * 3.0, 0.05, 0.85)
        value = 0.2 + 0.4 * (min(tx, 1.0) + 1) / 2
        s = value * succ - risk_base * (1 - succ)
        rx, ry = fr.to_raw(tx, ty)
        out.append(Candidate(_cmd("MOVE_TO", pid, tid, {"target_x": rx, "target_y": ry, "sprint": c["sprint"]}),
                             s, f"dribble {name} space={space:.2f}"))

    # CLEARANCE under pressure in own third
    if me[0] < p["clear_zone"] and pressure < p["pressure_radius"]:
        fwd = max(((t, q) for t, q in c["mates"].items() if t not in (pid, 0)),
                  key=lambda tq: tq[1][0], default=None)
        if fwd:
            out.append(Candidate(_cmd("PASS", pid, tid, {"target_player_id": fwd[0], "type": "AERIAL"}),
                                 0.6, f"clearance to {fwd[0]} pressure={pressure:.2f}"))
    return out


def _attack_off_ball(c, holder_pid) -> list[Candidate]:
    p, fr, pid, tid, me, role = c["p"], c["fr"], c["pid"], c["tid"], c["me"], c["role"]
    out = [Candidate(_cmd("CLEAR_OVERRIDE", pid, tid, {}), 0.4, "engine support")]
    if role in ("FWD1", "FWD2") and me[0] < c["ball"][0] + 0.1:
        lane = 0.3 if role == "FWD1" else -0.3
        if abs(me[1]) > 0.1:          # keep the flank the player is already on
            lane = math.copysign(0.3, me[1])
        tx = _clamp(c["ball"][0] + 0.3 + p["line_height"] * 0.5, -0.5, 0.85)
        ty = lane * fr.half_w / 0.64
        space, _ = nearest((tx, ty), c["opps"])
        rx, ry = fr.to_raw(tx, ty)
        spr = c["sprint"] and not p["no_fwd_sprint"]
        out.append(Candidate(_cmd("MOVE_TO", pid, tid, {"target_x": rx, "target_y": ry, "sprint": spr}),
                             0.35 + space, f"support run space={space:.2f}"))
    if role == "GK" or pid == 0:
        return [_gk_line(c, 0.5, "gk hold line (attack)")]
    return out


# ---------------------------------------------------------------- DEFEND

def _gk_line(c, score, reason) -> Candidate:
    p, fr, b = c["p"], c["fr"], c["ball"]
    g = (-1.0, 0.0)
    d = dist(g, b)
    k = min(p["gk_line_offset"] / d, 1.0) if d > 1e-6 else 0.0
    tx, ty = g[0] + (b[0] - g[0]) * k, g[1] + (b[1] - g[1]) * k
    rx, ry = fr.to_raw(tx, ty)
    return Candidate(_cmd("MOVE_TO", c["pid"], c["tid"], {"target_x": rx, "target_y": ry, "sprint": False}),
                     score, reason)


def _gk_block(c, target) -> Candidate | None:
    """GK leaves the line only when the ball is close and no teammate is near."""
    p, b = c["p"], c["ball"]
    if dist(b, (-1.0, 0.0)) < p["gk_leave_dist"]:
        mate_d = min((dist(q, target) for t, q in c["mates"].items() if t != c["pid"]), default=9.0)
        if mate_d > p["gk_mate_near"]:
            return Candidate(_cmd("INTERCEPT", c["pid"], c["tid"], {"aggressive": True}, 3), 0.9,
                             f"gk come out mate_d={mate_d:.2f}")
    return None


def _defend(c, hq, holder_pid) -> list[Candidate]:
    p, fr, pid, tid, me, role = c["p"], c["fr"], c["pid"], c["tid"], c["me"], c["role"]
    if role == "GK" or pid == 0:
        return [x for x in (_gk_block(c, hq), _gk_line(c, 0.5, "gk on ball-goal line")) if x]

    outfield = sorted((t for t in c["mates"] if t != 0), key=lambda t: (dist(c["mates"][t], hq), t))
    rank = outfield.index(pid) if pid in outfield else 9
    is_def = role == "DEF"
    def_ok = lambda x: (not is_def) or x <= p["def_max_x"]
    out = []

    def hold_own_half(score, why):
        tx = min(p["def_max_x"] - 0.1, max(hq[0] - 0.4, -0.8))
        ty = hq[1] * 0.5
        rx, ry = fr.to_raw(tx, ty)
        return Candidate(_cmd("MOVE_TO", pid, tid, {"target_x": rx, "target_y": ry, "sprint": c["sprint"]}),
                         score, why)

    if is_def and me[0] > p["def_max_x"]:
        out.append(hold_own_half(0.95, "def recover to own half"))

    high_ball = hq[0] > p["line_height"] + 0.2
    press_allowed = not (p["early"] and high_ball)
    second = (rank == 1 and p["press_bodies"] >= 2 and dist(me, hq) < p["second_press_dist"])
    if (rank == 0 or second) and press_allowed and def_ok(hq[0]):
        inten = p["press_intensity"] * (0.7 if p["early"] else 1.0)
        out.append(Candidate(_cmd("PRESS_BALL", pid, tid, {"intensity": round(inten, 2)}, 3),
                             0.9 if rank == 0 else 0.8, f"press rank={rank}"))
    elif rank == 1 and not p["use_mark"] and def_ok(hq[0]):
        out.append(Candidate(_cmd("INTERCEPT", pid, tid, {"aggressive": False}, 3), 0.7, "intercept lane"))
    else:
        # deterministic assignment: non-pressers by pid -> most dangerous unmarked opponents
        markers = sorted(t for t in outfield if outfield.index(t) >= 1)
        threats = sorted(((t, q) for t, q in c["opp_by_pid"] if t != holder_pid and t != 0),
                         key=lambda tq: (dist(tq[1], (-1.0, 0.0)), tq[0]))
        if markers and threats and pid in markers:
            t, q = threats[markers.index(pid) % len(threats)]
            if def_ok(q[0]):
                out.append(Candidate(_cmd("MARK", pid, tid, {"target_player_id": t, "tightness": "TIGHT"}, 3),
                                     0.6, f"mark {t} danger_d={dist(q, (-1.0, 0.0)):.2f}"))
    if is_def:
        out.append(hold_own_half(0.3, "def hold own half"))
    else:
        out.append(Candidate(_cmd("CLEAR_OVERRIDE", pid, tid, {}), 0.2, "engine shape"))
    return out


# ---------------------------------------------------------------- TRANSITION

def _transition(c) -> list[Candidate]:
    p, fr, pid, tid, me, b = c["p"], c["fr"], c["pid"], c["tid"], c["me"], c["ball"]
    if c["role"] == "GK" or pid == 0:
        return [x for x in (_gk_block(c, b), _gk_line(c, 0.5, "gk line (loose ball)")) if x]
    outfield = sorted((t for t in c["mates"] if t != 0), key=lambda t: (dist(c["mates"][t], b), t))
    out = [Candidate(_cmd("CLEAR_OVERRIDE", pid, tid, {}), 0.4, "engine recover shape")]
    if outfield and outfield[0] == pid:
        out.append(Candidate(_cmd("INTERCEPT", pid, tid, {"aggressive": True}, 3), 0.9, "nearest: chase ball"))
        rx, ry = fr.to_raw(*b)
        out.append(Candidate(_cmd("MOVE_TO", pid, tid, {"target_x": rx, "target_y": ry, "sprint": c["sprint"]}),
                             0.85, "nearest: run to ball"))
    elif c["role"] == "DEF" and me[0] > p["def_max_x"] + p["line_height"]:
        rx, ry = fr.to_raw(p["def_max_x"] - 0.1, b[1] * 0.5)
        out.append(Candidate(_cmd("MOVE_TO", pid, tid, {"target_x": rx, "target_y": ry, "sprint": c["sprint"]}),
                             0.6, "def recover"))
    return out
