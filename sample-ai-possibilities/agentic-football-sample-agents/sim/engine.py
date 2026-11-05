"""Simplified 5v5 football world for offline testing and parameter tuning.

This is NOT the AWS engine (that one is closed). It is a world with similar
meaning: same payload format, same command names, similar scale.

Team 0 (home) attacks +x, team 1 (away) attacks -x. Player ids 0..4, 0 = GK.

Controller factory contract:
    factory(team_id: int, seed: int) -> fn(game_state, team_id, pid) -> [command dict]
game_state is the full request payload: {"gameState": {...}, "teamId": t, "myPlayers": [pid]}.
"""

from __future__ import annotations

import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(HERE, "..", "lib"))
CONFIG_PATH = os.path.join(HERE, "config.json")

# Constants multiplied by field_scale (lengths and speeds).
SPATIAL_KEYS = [
    "half_length", "half_width", "goal_half_width", "player_speed", "sprint_speed",
    "control_radius", "pickup_speed_max", "pass_arrive_speed", "pass_speed_max",
    "through_lead", "shot_speed_base", "shot_speed_power", "intercept_radius",
    "tackle_range", "press_tackle_range", "gk_catch_radius", "gk_catch_speed_ref",
    "mark_tight", "mark_loose", "ai_shoot_dist",
]
# Constants per metre: divided by field_scale.
INV_SPATIAL_KEYS = ["pass_err_dist", "shot_err_dist"]

# Physics constants that domain randomization changes by default.
DEFAULT_RANDOM_KEYS = [
    "player_speed", "sprint_speed", "dribble_factor", "stamina_drain", "ball_friction",
    "pass_err_base", "pass_err_dist", "shot_err_base", "shot_err_dist", "intercept_radius",
    "intercept_prob", "tackle_prob", "press_tackle_prob", "gk_catch_prob", "gk_catch_radius",
    "receive_prob", "shot_speed_base",
]

ROLES = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}

# Base formation as fractions of (L, W) in the team's own attack frame.
FORMATION = {0: (-0.93, 0.0), 1: (-0.55, 0.0), 2: (-0.25, 0.0), 3: (0.05, -0.3), 4: (0.05, 0.3)}
KICKOFF = {0: (-0.93, 0.0), 1: (-0.5, 0.0), 2: (-0.02, 0.0), 3: (-0.15, -0.35), 4: (-0.15, 0.35)}


def load_config(overrides=None, path=None):
    with open(path or CONFIG_PATH) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    if overrides:
        cfg.update(overrides)
    s = float(cfg.get("field_scale", 1.0))
    if s != 1.0:
        for k in SPATIAL_KEYS:
            cfg[k] = cfg[k] * s
        for k in INV_SPATIAL_KEYS:
            cfg[k] = cfg[k] / s
    return cfg


def randomize(cfg, rng, keys=None, lo=0.8, hi=1.2):
    """Domain randomization: multiply chosen constants by U(lo, hi)."""
    out = dict(cfg)
    for k in (keys or DEFAULT_RANDOM_KEYS):
        if k in out:
            out[k] = out[k] * rng.uniform(lo, hi)
    out["sprint_speed"] = max(out["sprint_speed"], out["player_speed"] * 1.05)
    return out


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


class Player:
    __slots__ = ("team", "pid", "x", "y", "vx", "vy", "stamina", "sprinting",
                 "order", "receive_until", "down_until", "next_tackle", "stance", "next_think",
                 "ai_target")

    def __init__(self, team, pid, x, y):
        self.team, self.pid = team, pid
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.stamina = 1.0
        self.sprinting = False
        self.order = {"type": "AI"}
        self.receive_until = -1.0
        self.down_until = -1.0
        self.next_tackle = 0.0
        self.stance = 0
        self.next_think = 0.0
        self.ai_target = None


class Match:
    def __init__(self, home_fn, away_fn, seed=0, cfg=None):
        self.cfg = cfg or load_config()
        c = self.cfg
        self.rng = random.Random(seed)
        self.fns = [home_fn, away_fn]
        self.L, self.W, self.GW = c["half_length"], c["half_width"], c["goal_half_width"]
        self.S = float(c.get("field_scale", 1.0))  # length unit for fixed margins
        self.t = 0.0
        self.tick = 0
        self.score = [0, 0]
        self.players = [[Player(t, i, 0, 0) for i in range(5)] for t in (0, 1)]
        self.all = self.players[0] + self.players[1]
        self.owner = None          # Player with the ball
        self.last_touch = None     # Player
        self.bx = self.by = 0.0
        self.bvx = self.bvy = 0.0
        self.flight = None         # dict while a pass/shot is moving
        self.protect_until = 0.0
        self.stats = {
            "shots": [0, 0], "on_target": [0, 0], "passes": [0, 0], "passes_ok": [0, 0],
            "tackles_won": [0, 0], "interceptions": [0, 0], "saves": [0, 0],
            "possession_s": [0.0, 0.0], "errors": [0, 0], "outs": 0,
            "cmds": [{}, {}],
        }
        self.all_rev = self.players[1] + self.players[0]
        self.all_fwd = self.all
        self.kickoff(self.rng.randrange(2))

    # ------------------------------------------------------------------ geometry
    def attack_dir(self, team):
        return 1.0 if team == 0 else -1.0

    def opp_goal_x(self, team):
        return self.L * self.attack_dir(team)

    def own_goal_x(self, team):
        return -self.L * self.attack_dir(team)

    def clamp_in(self, x, y, margin=0.3):
        margin *= self.S
        return (max(-self.L + margin, min(self.L - margin, x)),
                max(-self.W + margin, min(self.W - margin, y)))

    # ------------------------------------------------------------------ restarts
    def kickoff(self, team):
        for t in (0, 1):
            d = self.attack_dir(t)
            for p in self.players[t]:
                fx, fy = KICKOFF[p.pid]
                p.x, p.y = fx * self.L * d, fy * self.W * d
                p.vx = p.vy = 0.0
                p.receive_until = p.down_until = -1.0
        taker = self.players[team][2]
        taker.x, taker.y = -0.5 * self.S * self.attack_dir(team), 0.0
        self.give(taker, protect=1.5)

    def give(self, p, protect=None):
        self.owner = p
        self.last_touch = p
        self.flight = None
        self.bx, self.by = p.x, p.y
        self.bvx = self.bvy = 0.0
        p.receive_until = -1.0
        self.protect_until = self.t + (self.cfg["possession_protect"] if protect is None else protect)

    def restart_out(self, x, y, goal_line):
        self.stats["outs"] += 1
        lt = self.last_touch.team if self.last_touch else 0
        other = 1 - lt
        if goal_line:
            side = 1 if x > 0 else -1
            defending = 0 if side < 0 else 1  # team whose goal is on this side
            if lt != defending:  # attackers touched last -> goal kick
                gk = self.players[defending][0]
                gk.x, gk.y = side * (self.L - 5.0 * self.S), 0.0
                self.give(gk, protect=1.5)
                return
        bx, by = self.clamp_in(x, y, margin=0.5)
        taker = min(self.players[other][1:], key=lambda p: (p.x - bx) ** 2 + (p.y - by) ** 2)
        taker.x, taker.y = bx, by
        self.give(taker, protect=1.5)

    # ------------------------------------------------------------------ payload
    def payload(self, team, pid):
        return {"gameState": self.state_dict, "teamId": team, "myPlayers": [pid]}

    def build_state(self):
        players = []
        for p in self.all:
            players.append({
                "agentId": f"agentId_{p.pid}", "teamCode": "home" if p.team == 0 else "away",
                "position": {"x": round(p.x, 3), "y": round(p.y, 3)},
                "velocity": {"x": round(p.vx, 3), "y": round(p.vy, 3)},
                "orientation": 0, "stamina": round(p.stamina, 3),
                "speed": round(math.hypot(p.vx, p.vy), 3), "isSprinting": p.sprinting,
            })
        o = self.owner
        self.state_dict = {
            "tick": self.tick, "gameTime": round(self.t, 2), "playMode": "OPEN_PLAY",
            "modeTeamId": None,
            "score": {"home": self.score[0], "away": self.score[1]},
            "ball": {
                "position": {"x": round(self.bx, 3), "y": round(self.by, 3), "z": 0},
                "velocity": {"x": round(self.bvx, 3), "y": round(self.bvy, 3), "z": 0},
                "isFree": o is None,
                "possessionAgentId": f"agentId_{o.pid}" if o else None,
            },
            "players": players,
            "teamChat": [],
        }

    # ------------------------------------------------------------------ decisions
    def decide_all(self):
        self.build_state()
        pending = []
        for t in (0, 1):
            fn = self.fns[t]
            for pid in range(5):
                try:
                    cmds = fn(self.payload(t, pid), t, pid)
                except Exception:
                    self.stats["errors"][t] += 1
                    cmds = []
                cmd = None
                for c in cmds or []:
                    if isinstance(c, dict) and c.get("playerId", pid) in (pid, str(pid)):
                        cmd = c
                        break
                pending.append((self.players[t][pid], cmd))
        self.rng.shuffle(pending)
        for p, cmd in pending:
            self.apply(p, cmd)

    def apply(self, p, cmd):
        if not cmd:
            p.order = {"type": "AI"}
            return
        ct = str(cmd.get("commandType", "")).upper()
        prm = cmd.get("parameters") or {}
        if not isinstance(prm, dict):
            prm = {}
        cc = self.stats["cmds"][p.team]
        cc[ct] = cc.get(ct, 0) + 1
        has = self.owner is p
        if ct == "MOVE_TO":
            tx, ty = self.clamp_in(_f(prm.get("target_x"), p.x), _f(prm.get("target_y"), p.y))
            p.order = {"type": "MOVE", "x": tx, "y": ty, "sprint": bool(prm.get("sprint", False))}
        elif ct == "PASS":
            if has:
                self.do_pass(p, int(_f(prm.get("target_player_id"), 2)), str(prm.get("type", "GROUND")).upper())
            p.order = {"type": "AI"}
        elif ct == "GK_DISTRIBUTE":
            if has:
                m = str(prm.get("method", "THROW")).upper()
                self.do_pass(p, int(_f(prm.get("target_player_id"), 1)), "THROW" if m == "THROW" else "AERIAL")
            p.order = {"type": "AI"}
        elif ct == "SHOOT":
            if has:
                self.do_shot(p, str(prm.get("aim_location", "CENTER")).upper(),
                             max(0.0, min(1.0, _f(prm.get("power"), 0.8))))
            p.order = {"type": "AI"}
        elif ct == "SLIDE_TACKLE":
            tgt = prm.get("target_player_id")
            p.order = {"type": "SLIDE", "target": None if tgt is None else int(_f(tgt, 0))}
        elif ct == "PRESS_BALL":
            p.order = {"type": "PRESS", "intensity": max(0.0, min(1.0, _f(prm.get("intensity"), 0.7)))}
        elif ct == "MARK":
            tgt = prm.get("target_player_id")
            if tgt is None:
                p.order = {"type": "AI"}
            else:
                tight = str(prm.get("tightness", "TIGHT")).upper()
                p.order = {"type": "MARK", "target": int(_f(tgt, 0)),
                           "gap": self.cfg["mark_tight"] if tight == "TIGHT" else self.cfg["mark_loose"]}
        elif ct == "INTERCEPT":
            p.order = {"type": "INTERCEPT", "aggressive": bool(prm.get("aggressive", True))}
        elif ct == "FOLLOW_PLAYER":
            team_s = str(prm.get("target_team", "")).upper()
            if team_s in ("HOME", "AWAY"):
                tt = 0 if team_s == "HOME" else 1
            else:
                tt = 1 - p.team
            p.order = {"type": "FOLLOW", "target": int(_f(prm.get("target_player_id"), 0)) % 5,
                       "team": tt, "dist": _f(prm.get("distance"), 3.0 * self.cfg["field_scale"])}
        elif ct == "SET_STANCE":
            p.stance = int(_f(prm.get("stance"), 0))
            p.order = {"type": "AI"}
        elif ct == "RESET":
            for q in self.players[p.team]:
                q.order = {"type": "AI"}
        else:  # CLEAR_OVERRIDE and unknown
            p.order = {"type": "AI"}

    # ------------------------------------------------------------------ kicks
    def kick(self, p, tx, ty, speed, sigma, kind, receiver=None, aerial=False):
        ang = math.atan2(ty - p.y, tx - p.x) + self.rng.gauss(0.0, sigma)
        self.owner = None
        self.last_touch = p
        self.bx, self.by = p.x, p.y
        self.bvx, self.bvy = speed * math.cos(ang), speed * math.sin(ang)
        self.flight = {"kind": kind, "team": p.team, "kicker": p, "t0": self.t, "receiver": receiver,
                       "aerial": aerial, "sx": p.x, "sy": p.y, "dist": math.hypot(tx - p.x, ty - p.y),
                       "rolled": set()}

    def do_pass(self, p, tpid, ptype):
        c = self.cfg
        tpid = tpid % 5
        r = self.players[p.team][tpid]
        if r is p:
            return
        tx, ty = r.x, r.y
        if ptype == "THROUGH":
            tx += self.attack_dir(p.team) * c["through_lead"] + r.vx * 0.5
            ty += r.vy * 0.5
        tx, ty = self.clamp_in(tx, ty, 1.0)
        d = math.hypot(tx - p.x, ty - p.y)
        sigma = c["pass_err_base"] + c["pass_err_dist"] * d
        if ptype == "AERIAL":
            sigma *= 1.5
        elif ptype == "THROW":
            sigma *= 0.6
            d = min(d, 30.0 * c["field_scale"])
        speed = min(c["pass_speed_max"], c["ball_friction"] * d + c["pass_arrive_speed"])
        self.stats["passes"][p.team] += 1
        self.kick(p, tx, ty, speed, sigma, "pass", receiver=r, aerial=(ptype == "AERIAL"))
        r.receive_until = self.t + 4.0

    def do_shot(self, p, aim, power):
        c = self.cfg
        gx = self.opp_goal_x(p.team)
        left = self.attack_dir(p.team)  # +y is left for team 0
        side = 0.0
        if aim.endswith("L"):
            side = 0.75
        elif aim.endswith("R"):
            side = -0.75
        ty = side * left * self.GW
        d = math.hypot(gx - p.x, ty - p.y)
        sigma = c["shot_err_base"] + c["shot_err_power"] * power * power + c["shot_err_dist"] * d
        if aim.startswith("T"):
            sigma *= 1.2
        speed = c["shot_speed_base"] + c["shot_speed_power"] * power
        self.stats["shots"][p.team] += 1
        self.kick(p, gx, ty, speed, sigma, "shot")
        self.flight["top"] = aim.startswith("T")
        # on-target estimate: straight line crossing goal line inside the posts
        if abs(self.bvx) > 1e-6:
            yc = p.y + self.bvy * (gx - p.x) / self.bvx
            if abs(yc) < self.GW:
                self.stats["on_target"][p.team] += 1

    # ------------------------------------------------------------------ movement targets
    def nearest(self, team, x, y, skip_gk=False):
        best, bd = None, 1e18
        for q in self.players[team]:
            if skip_gk and q.pid == 0:
                continue
            d = (q.x - x) ** 2 + (q.y - y) ** 2
            if d < bd:
                best, bd = q, d
        return best

    def ball_path_point(self, p):
        """Point on the current ball path nearest to p (for receive/intercept)."""
        f = self.cfg["ball_friction"]
        sp = math.hypot(self.bvx, self.bvy)
        if sp < 0.5 * self.S:
            return self.bx, self.by
        reach = sp / f
        ux, uy = self.bvx / sp, self.bvy / sp
        s = max(0.0, min(reach, (p.x - self.bx) * ux + (p.y - self.by) * uy))
        return self.bx + ux * s, self.by + uy * s

    def formation_spot(self, p, has_ball_team):
        d = self.attack_dir(p.team)
        fx, fy = FORMATION[p.pid]
        x = fx * self.L * d
        y = fy * self.W * d
        if p.pid == 0:
            gx = self.own_goal_x(p.team)
            # stand on the line ball -> goal centre
            k = 3.0 * self.cfg["field_scale"]
            dx, dy = self.bx - gx, self.by
            n = math.hypot(dx, dy) or 1.0
            return gx + dx / n * k, max(-self.GW, min(self.GW, dy / n * k))
        x += self.bx * 0.45
        y += self.by * 0.3
        if has_ball_team:
            x += 0.12 * self.L * d
        if p.stance == 1:
            x += 0.08 * self.L * d
        elif p.stance == 2:
            x -= 0.08 * self.L * d
        return self.clamp_in(x, y, 1.0)

    def target_for(self, p):
        """Return (tx, ty, sprint) for an off-ball (or carrier) player this step."""
        o = self.order_type(p)
        order = p.order
        owner = self.owner
        my_ball = owner is not None and owner.team == p.team
        if p is owner:
            if o == "MOVE":
                return order["x"], order["y"], order["sprint"]
            return self.ai_carrier(p)
        if p.receive_until > self.t and self.flight and self.flight.get("receiver") is p:
            x, y = self.ball_path_point(p)
            return x, y, True
        if o == "MOVE":
            return order["x"], order["y"], order["sprint"]
        if o == "PRESS":
            if owner is not None and owner.team != p.team:
                return owner.x, owner.y, order["intensity"] >= 0.5
            if owner is None:
                x, y = self.ball_path_point(p)
                return x, y, order["intensity"] >= 0.5
            return self.ai_offball(p, my_ball)
        if o == "SLIDE":
            tgt = self.players[1 - p.team][order["target"] % 5] if order["target"] is not None else owner
            if tgt is None:
                return self.ai_offball(p, my_ball)
            return tgt.x, tgt.y, True
        if o == "MARK":
            tgt = self.players[1 - p.team][order["target"] % 5]
            gx = self.own_goal_x(p.team)
            dx, dy = gx - tgt.x, -tgt.y
            n = math.hypot(dx, dy) or 1.0
            g = min(order["gap"], n)
            return tgt.x + dx / n * g, tgt.y + dy / n * g, True
        if o == "INTERCEPT":
            if owner is not None and owner.team != p.team:
                gx = self.own_goal_x(p.team)
                return (owner.x * 0.7 + gx * 0.3), owner.y * 0.7, order["aggressive"]
            if owner is None:
                x, y = self.ball_path_point(p)
                return x, y, order["aggressive"]
            return self.ai_offball(p, my_ball)
        if o == "FOLLOW":
            tgt = self.players[order["team"]][order["target"]]
            dx, dy = p.x - tgt.x, p.y - tgt.y
            n = math.hypot(dx, dy) or 1.0
            g = order["dist"]
            return tgt.x + dx / n * g, tgt.y + dy / n * g, n > g * 2
        return self.ai_offball(p, my_ball)

    def order_type(self, p):
        return p.order.get("type", "AI")

    def ai_offball(self, p, my_ball):
        owner = self.owner
        if not my_ball:
            # chase if nearest field player of my team (GK only near own goal)
            chaser = self.nearest(p.team, self.bx, self.by, skip_gk=True)
            gx = self.own_goal_x(p.team)
            near_goal = abs(self.bx - gx) < 0.18 * self.L and abs(self.by) < 0.35 * self.W
            if p is chaser or (p.pid == 0 and near_goal and owner is None):
                if owner is None:
                    x, y = self.ball_path_point(p)
                    return x, y, True
                return owner.x, owner.y, True
        x, y = self.formation_spot(p, my_ball)
        return x, y, False

    def ai_carrier(self, p):
        c = self.cfg
        gx = self.opp_goal_x(p.team)
        if self.t >= p.next_think:
            p.next_think = self.t + c["ai_think_interval"]
            dgoal = math.hypot(gx - p.x, p.y)
            if dgoal < c["ai_shoot_dist"] and p.pid != 0:
                self.do_shot(p, self.rng.choice(["BL", "BR", "TL", "TR"]), 0.85)
                return p.x, p.y, False
            opp = self.players[1 - p.team]
            pressure = min(math.hypot(q.x - p.x, q.y - p.y) for q in opp)
            if p.pid == 0 or pressure < 3.0 * c["field_scale"]:
                d = self.attack_dir(p.team)
                mates = [q for q in self.players[p.team] if q is not p and q.pid != 0]
                best, bs = None, -1e9
                for q in mates:
                    space = min(math.hypot(o.x - q.x, o.y - q.y) for o in opp)
                    lane = min(_seg_dist(o.x, o.y, p.x, p.y, q.x, q.y) for o in opp)
                    s = ((q.x - p.x) * d * 0.3 + min(space, 10.0 * self.S) + min(lane, 5.0 * self.S) * 1.5) / self.S
                    if s > bs:
                        best, bs = q, s
                if best is not None:
                    self.do_pass(p, best.pid, "GROUND")
                    return p.x, p.y, False
        return gx, p.y * 0.8, False

    # ------------------------------------------------------------------ step
    def move_players(self, dt):
        c = self.cfg
        for p in self.all:
            if p.down_until > self.t:
                p.vx = p.vy = 0.0
                p.sprinting = False
                p.stamina = min(1.0, p.stamina + c["stamina_recover"] * dt)
                continue
            tx, ty, sprint = self.target_for(p)
            sprint = sprint and p.stamina > 0.1
            v = c["sprint_speed"] if sprint else c["player_speed"]
            v *= c["stamina_speed_min"] + (1 - c["stamina_speed_min"]) * p.stamina
            if self.owner is p:
                v *= c["dribble_factor"]
            dx, dy = tx - p.x, ty - p.y
            n = math.hypot(dx, dy)
            step = v * dt
            if n <= step or n < 1e-6:
                nx, ny = tx, ty
                moving = n > 0.05 * self.S
            else:
                nx, ny = p.x + dx / n * step, p.y + dy / n * step
                moving = True
            nx, ny = self.clamp_in(nx, ny, 0.2)
            p.vx, p.vy = (nx - p.x) / dt, (ny - p.y) / dt
            p.x, p.y = nx, ny
            p.sprinting = sprint and moving
            if p.sprinting:
                p.stamina = max(0.0, p.stamina - c["stamina_drain"] * dt)
            else:
                p.stamina = min(1.0, p.stamina + c["stamina_recover"] * dt)

    def tackles(self):
        c = self.cfg
        o = self.owner
        if o is None or self.t < self.protect_until:
            return
        for p in self.players[1 - o.team]:
            if p.down_until > self.t or p.next_tackle > self.t:
                continue
            d = math.hypot(p.x - o.x, p.y - o.y)
            ot = self.order_type(p)
            if ot == "SLIDE" and (p.order["target"] is None or p.order["target"] % 5 == o.pid):
                if d <= c["tackle_range"]:
                    p.order = {"type": "AI"}
                    if self.rng.random() < c["tackle_prob"] * (1.0 - 0.6 * d / c["tackle_range"]):
                        self.stats["tackles_won"][p.team] += 1
                        o.down_until = self.t + 0.5
                        self.give(p)
                        return
                    p.down_until = self.t + c["slide_recover"]
                continue
            if d <= c["press_tackle_range"]:
                p.next_tackle = self.t + c["tackle_interval"]
                k = p.order.get("intensity", 0.7) if ot == "PRESS" else 0.6
                if self.rng.random() < c["press_tackle_prob"] * (0.5 + k) * (1.0 - 0.5 * d / c["press_tackle_range"]):
                    self.stats["tackles_won"][p.team] += 1
                    o.down_until = self.t + 0.3
                    self.give(p)
                    return

    def move_ball(self, dt):
        c = self.cfg
        o = self.owner
        if o is not None:
            self.bx, self.by = o.x, o.y
            self.bvx, self.bvy = o.vx, o.vy
            self.stats["possession_s"][o.team] += dt
            return
        ox, oy = self.bx, self.by
        decay = math.exp(-c["ball_friction"] * dt)
        self.bx += self.bvx * dt
        self.by += self.bvy * dt
        self.bvx *= decay
        self.bvy *= decay
        sp = math.hypot(self.bvx, self.bvy)
        # goal / out
        if abs(self.bx) > self.L:
            side = 1.0 if self.bx > 0 else -1.0
            xl = side * self.L
            frac = (xl - ox) / (self.bx - ox) if self.bx != ox else 0.0
            yc = oy + (self.by - oy) * frac
            if abs(yc) < self.GW:
                scorer = 0 if side > 0 else 1
                self.score[scorer] += 1
                self.kickoff(1 - scorer)
                return
            self.restart_out(xl, yc, goal_line=True)
            return
        if abs(self.by) > self.W:
            self.restart_out(self.bx, self.by, goal_line=False)
            return
        fl = self.flight
        if fl is not None and sp < c["pickup_speed_max"] * 0.5:
            self.flight = fl = None
        if fl is not None:
            travelled = math.hypot(self.bx - fl["sx"], self.by - fl["sy"])
            for p in self.all:
                if p in fl["rolled"] or p.down_until > self.t:
                    continue
                if p is fl["kicker"] and self.t - fl["t0"] < c["kicker_immunity"]:
                    continue
                dseg = _seg_dist(p.x, p.y, ox, oy, self.bx, self.by)
                if fl["kind"] == "shot" and p.pid == 0 and p.team != fl["team"]:
                    if dseg < c["gk_catch_radius"]:
                        fl["rolled"].add(p)
                        pc = c["gk_catch_prob"] * math.exp(-sp / c["gk_catch_speed_ref"])
                        pc *= (1.0 - 0.5 * dseg / c["gk_catch_radius"])
                        if fl.get("top"):
                            pc *= 0.8
                        r = self.rng.random()
                        if r < pc:
                            self.stats["saves"][p.team] += 1
                            self.give(p, protect=1.0)
                            return
                        if r < pc + (1 - pc) * c["gk_parry_prob"]:
                            self.stats["saves"][p.team] += 1
                            ang = self.rng.uniform(-1.2, 1.2) + (0.0 if p.team == 0 else math.pi)
                            v = sp * 0.35
                            self.bx, self.by = p.x, p.y
                            self.bvx, self.bvy = v * math.cos(ang), v * math.sin(ang)
                            self.last_touch = p
                            self.flight = None
                            return
                    continue
                if dseg >= c["intercept_radius"]:
                    continue
                if fl["aerial"] and fl["dist"] * 0.65 > travelled > 2.0:
                    continue
                fl["rolled"].add(p)
                if p.team == fl["team"]:
                    if p is fl["receiver"] or dseg < c["control_radius"]:
                        pr = c["receive_prob"] * (1.0 - 0.3 * dseg / c["intercept_radius"])
                        if self.rng.random() < pr:
                            if fl["kind"] == "pass":
                                self.stats["passes_ok"][p.team] += 1
                            self.give(p)
                            return
                else:
                    pi = c["intercept_prob"] * (1.0 - dseg / c["intercept_radius"])
                    if fl["kind"] == "shot":
                        pi *= 0.5
                    if self.rng.random() < pi:
                        self.stats["interceptions"][p.team] += 1
                        self.give(p)
                        return
            return
        # slow free ball: nearest player within control radius takes it
        best, bd = None, c["control_radius"]
        for p in self.all:
            if p.down_until > self.t:
                continue
            d = math.hypot(p.x - self.bx, p.y - self.by)
            if d < bd:
                best, bd = p, d
        if best is not None:
            if self.flight is None and self.last_touch is not None and best.team == self.last_touch.team \
                    and self.last_touch is not best and best.receive_until > self.t:
                self.stats["passes_ok"][best.team] += 1
            self.give(best)

    def step(self):
        dt = self.cfg["dt"]
        # alternate update order so neither team is favoured by iteration order
        self.all = self.all_rev if self.tick % 2 else self.all_fwd
        self.move_players(dt)
        self.tackles()
        self.move_ball(dt)
        self.t += dt
        self.tick += 1

    def run(self):
        c = self.cfg
        n_steps = int(round(c["match_time"] / c["dt"]))
        every = max(1, int(round(c["decision_interval"] / c["dt"])))
        for i in range(n_steps):
            if i % every == 0:
                self.decide_all()
            self.step()
        st = self.stats
        st["possession_s"] = [round(x, 1) for x in st["possession_s"]]
        return {"home": self.score[0], "away": self.score[1], "stats": st}


def run_match(home_fn_factory, away_fn_factory, seed, config_overrides=None,
              randomize_keys=None, cfg=None):
    """Play one match. randomize_keys: None = no randomization, True = default keys, list = those keys."""
    cfg = cfg or load_config(config_overrides)
    if randomize_keys:
        rr = random.Random(seed * 7919 + 13)
        cfg = randomize(cfg, rr, None if randomize_keys is True else randomize_keys)
    home = home_fn_factory(0, seed)
    away = away_fn_factory(1, seed + 1000003)
    return Match(home, away, seed=seed, cfg=cfg).run()
