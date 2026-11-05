"""In-process match memory built from consecutive payloads.

update(game_state, team_id) once per tick. All positions are in the
normalized frame of norm.py: +x = opponent goal, our goal = (-1, 0).
report() gives a short text summary for the manager LLM.

Module global: reset when the tick goes back or the score returns to 0-0
at low game time (a new match in the same process). Never raises.
"""

from collections import deque

import norm
from state import _player_idx, _is_my_team, _resolve_holder

WINDOW = 15          # ticks kept per goal summary
HISTORY = 300        # ticks kept in total
MAX_GOALS = 20


def _third(x: float) -> str:
    return "def" if x < -1 / 3 else ("att" if x > 1 / 3 else "mid")


class Memory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.team_id = None
        self.last_tick = None
        self.last_time = 0.0
        self.score = (0, 0)               # (us, them)
        self.ticks = deque(maxlen=HISTORY)
        self.goals = []                   # list of goal dicts
        self.n_ticks = 0
        self.poss = {"us": 0, "them": 0}
        self.losses = {"def": 0, "mid": 0, "att": 0}   # our losses by ball third
        self.wins = {"def": 0, "mid": 0, "att": 0}
        self.last_side = None
        self.last_change = None           # (tick index, side that won, x)
        self.press_sum = 0.0
        self.press_n = 0
        self.gk_sum = 0.0
        self.gk_n = 0
        self.resets = getattr(self, "resets", -1) + 1

    # ------------------------------------------------------------ update
    def update(self, gs: dict, team_id: int) -> None:
        try:
            self._update(gs or {}, team_id)
        except Exception:  # noqa: BLE001
            pass

    def _score(self, gs, team_id):
        sc = gs.get("score") or {}
        h, a = int(sc.get("home", 0) or 0), int(sc.get("away", 0) or 0)
        return (h, a) if team_id == 0 else (a, h)

    def _update(self, gs, team_id):
        tick = gs.get("tick")
        t = gs.get("gameTime") or 0
        t = float(t) if isinstance(t, (int, float)) else 0.0
        score = self._score(gs, team_id)
        new_match = (
            (isinstance(tick, (int, float)) and isinstance(self.last_tick, (int, float)) and tick < self.last_tick)
            or (score == (0, 0) and self.score != (0, 0) and t < 10)
            or (self.team_id is not None and team_id != self.team_id)
        )
        if new_match:
            self.reset()
        if isinstance(tick, (int, float)) and tick == self.last_tick:
            return                         # same payload again
        if self.n_ticks == 0:
            self.score = score             # joined mid-match: not new goals
        self.team_id = team_id
        self.last_tick = tick
        self.last_time = t

        sx, sy = norm.update_scale(gs)
        fr = norm.Frame(sx, sy, team_id)
        ball = gs.get("ball") or {}
        b = fr.to_norm(ball.get("position"))
        players = gs.get("players") or []
        holder = None if ball.get("isFree") is True else _resolve_holder(ball, players)
        side = hpid = None
        if holder is not None:
            side = "us" if _is_my_team(holder, team_id) else "them"
            hpid = _player_idx(holder)
        rec = {"tick": tick, "t": t, "bx": round(b[0], 2), "by": round(b[1], 2), "side": side, "hpid": hpid}
        self.ticks.append(rec)
        self.n_ticks += 1
        idx = self.n_ticks

        if side is not None:
            self.poss[side] += 1
            if self.last_side is not None and side != self.last_side:
                th = _third(b[0])
                if side == "them":
                    self.losses[th] += 1
                else:
                    self.wins[th] += 1
                self.last_change = (idx, side, b[0])
            self.last_side = side

        opps = [(_player_idx(p), fr.to_norm(p.get("position"))) for p in players if not _is_my_team(p, team_id)]
        if side == "us" and opps:
            # how high they engage: x of the opponent nearest to the ball
            q = min((q for i, q in opps if i != 0), key=lambda q: norm.dist(q, b), default=None)
            if q is not None:
                self.press_sum += q[0]
                self.press_n += 1
        for i, q in opps:
            if i == 0:
                self.gk_sum += 1.0 - q[0]       # distance from their goal line
                self.gk_n += 1

        if score != self.score:
            for who, n in (("us", score[0] - self.score[0]), ("them", score[1] - self.score[1])):
                for _ in range(max(n, 0)):
                    self._goal(who, t, idx)
            self.score = score

    def _goal(self, who: str, t: float, idx: int) -> None:
        win = [r for r in list(self.ticks)[-WINDOW - 1:-1]] or list(self.ticks)[-1:]
        held = [r for r in win if r["side"] is not None]
        mine = [r for r in held if r["side"] == who]
        holder_side = max(("us", "them"), key=lambda s: sum(r["side"] == s for r in held)) if held else None
        passes, prev = 0, None
        for r in mine:
            if prev is not None and r["hpid"] != prev:
                passes += 1
            prev = r["hpid"]
        last = mine[-1] if mine else (win[-1] if win else {"bx": 0.0, "by": 0.0})
        # shot spot from the scorer's view: distance to the goal they attack
        sx = last["bx"] if who == "us" else -last["bx"]
        shot_d = round(norm.dist((sx, last["by"]), (1.0, 0.0)), 2)
        won_x = None
        counter = False
        if self.last_change and self.last_change[1] == who and idx - self.last_change[0] <= WINDOW:
            won_x = self.last_change[2] if who == "us" else -self.last_change[2]  # scorer frame
            counter = passes <= 3
        g = {"who": who, "t": round(t, 1), "holder": holder_side, "passes": passes, "shot_d": shot_d,
             "shot_zone": "close" if shot_d < 0.3 else ("box" if shot_d < 0.5 else "long"),
             "won_third": _third(won_x) if won_x is not None else None, "counter": counter}
        self.goals.append(g)
        del self.goals[:-MAX_GOALS]

    # ------------------------------------------------------------ stats
    def stats(self) -> dict:
        tot = self.poss["us"] + self.poss["them"]
        return {
            "score": self.score,
            "t": self.last_time,
            "poss_us": round(self.poss["us"] / tot, 2) if tot else None,
            "losses": dict(self.losses),
            "press_x": round(self.press_sum / self.press_n, 2) if self.press_n else None,
            "gk_depth": round(self.gk_sum / self.gk_n, 2) if self.gk_n else None,
            "goals": len(self.goals),
            "conceded_counters": sum(1 for g in self.goals if g["who"] == "them" and g["counter"]),
        }

    def report(self) -> str:
        s = self.stats()
        lines = [f"Score us {s['score'][0]} - {s['score'][1]} them, game time {s['t']:.0f}s"]
        if s["poss_us"] is not None:
            lines.append(f"Possession us {s['poss_us'] * 100:.0f}%")
        l = s["losses"]
        lines.append(f"Our ball losses by third: def {l['def']}, mid {l['mid']}, att {l['att']}")
        if s["press_x"] is not None:
            h = "high" if s["press_x"] > 0.2 else ("low" if s["press_x"] < -0.2 else "mid")
            lines.append(f"Opponent press height: {h} (x={s['press_x']}, +1 = their goal)")
        if s["gk_depth"] is not None:
            lines.append(f"Opponent GK depth: {s['gk_depth']} from goal line"
                         + (" (often off the line)" if s["gk_depth"] > 0.2 else ""))
        lines.append(f"Goals conceded from counters: {s['conceded_counters']}")
        for g in self.goals[-5:]:
            how = "counter" if g["counter"] else "build-up"
            won = f", won in {g['won_third']} third" if g["won_third"] else ""
            lines.append(f"t={g['t']:.0f}s {g['who'].upper()} scored: {how}, {g['passes']} passes, "
                         f"shot {g['shot_zone']}{won}")
        return "\n".join(lines[:12])


MEMORY = Memory()


def update(game_state: dict, team_id: int) -> None:
    MEMORY.update(game_state, team_id)


def report() -> str:
    try:
        return MEMORY.report()
    except Exception:  # noqa: BLE001
        return "no match data"


def stats() -> dict:
    return MEMORY.stats()


def reset() -> None:
    MEMORY.reset()
