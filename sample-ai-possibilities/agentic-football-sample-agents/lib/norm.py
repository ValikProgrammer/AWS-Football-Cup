"""Field-scale estimation and normalized geometry.

Normalized frame: x and y are both divided by the estimated half-length sx,
then rotated 180 degrees for the away team, so +x is always the opponent
goal and +y is always the attacker's left. Opponent goal = (1, 0).
"""

import math

_ASPECT = 0.64          # max half-width / half-length: lower bound for sx from the lateral spread
_ASPECT_MIN = 0.56      # prior half-width / half-length before the lateral spread is seen
_MARGIN = 1.0           # players are clamped at the end line (match: |x| <= 6.40, line 6.45)
_GK_MARGIN = 1.08       # a GK on its line stands at ~0.92 x half-length (match 5.94 / 6.4)
_FLOOR = 0.5
_LOCK_Z = 0.5           # |z| above this with z nearer the players: z is the lateral axis
_CANON = "_axes"        # marker on a ball already made canonical by canonical_ball()

_CLAMP_TOL = 1e-3       # relative: |x| this close to max_x counts as "on the end line"

_state = {"max_x": 0.0, "max_y": 0.0, "gk_x": 0.0, "last_tick": None, "z_lateral": False,
          "at_max": {}, "line_found": False}


def reset() -> None:
    _state.update(max_x=0.0, max_y=0.0, gk_x=0.0, last_tick=None, z_lateral=False,
                  at_max={}, line_found=False)


def _is_gk(pl: dict) -> bool:
    aid = pl.get("agentId")
    if isinstance(aid, str):
        return aid.rsplit("_", 1)[-1] == "0"
    return pl.get("playerId") == 0


def _xy(pos) -> tuple[float, float]:
    """(x, lateral) from a position dict. Uses z as lateral when y is absent."""
    if not isinstance(pos, dict):
        return 0.0, 0.0
    x = pos.get("x", 0) or 0
    y = pos.get("y")
    if y is None:
        y = pos.get("z", 0) or 0
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _swap_yz(v):
    if not isinstance(v, dict) or "z" not in v:
        return v
    out = dict(v)
    out["y"], out["z"] = v.get("z"), v.get("y")
    return out


def canonical_ball(game_state: dict) -> dict:
    """Return game_state with ball position/velocity as {x, y=lateral, z=height}.

    Scrimmages send the ball as {x, y=lateral, z=0}; real matches send
    {x, y=height, z=lateral}. Players are always {x, y=lateral}. The lateral
    axis of the ball is the one (y or z) that puts the ball nearer to a
    player. Once z wins with |z| > 0.5 the choice locks to z (reset()).
    A ball without z, or with z == 0, is left as it is. Idempotent: a ball
    this function swapped carries a marker and is not swapped again.
    """
    if not isinstance(game_state, dict):
        return game_state
    ball = game_state.get("ball")
    if not isinstance(ball, dict) or ball.get(_CANON):
        return game_state
    pos = ball.get("position")
    if not isinstance(pos, dict) or "z" not in pos:
        return game_state
    bx, by, bz = _num(pos.get("x")), _num(pos.get("y")), _num(pos.get("z"))
    if bx is None or bz is None:
        return game_state
    use_z = _state["z_lateral"]
    if not use_z and abs(bz) > 1e-6:
        if by is None:
            use_z = True
        else:
            pts = [_xy(p.get("position")) for p in game_state.get("players") or [] if isinstance(p, dict)]
            if pts:
                d_y = min(math.hypot(px - bx, py - by) for px, py in pts)
                d_z = min(math.hypot(px - bx, py - bz) for px, py in pts)
                use_z = d_z < d_y
        if use_z and abs(bz) > _LOCK_Z:
            _state["z_lateral"] = True
    if not use_z:
        return game_state
    b2 = dict(ball)
    b2["position"] = _swap_yz(pos)
    b2["velocity"] = _swap_yz(ball.get("velocity"))
    b2[_CANON] = "yz"
    gs2 = dict(game_state)
    gs2["ball"] = b2
    return gs2


def update_scale(game_state: dict) -> tuple[float, float]:
    """Update the running max with this tick; return (sx, sy) in raw units.

    Half-length comes from player x only: the ball goes behind the goal line.
    sx = max(max player |x|, 1.08 x GK |x|). The GK term uses the smaller
    |x| of the two GKs on a tick: one GK can walk to the far end line.
    The server clamps players at the end line: once two different players
    stand at the same max x on the same end, that x is the line and the GK
    term is dropped (a GK on that line would read 8 % too long).
    Half-width comes from player y and the ball lateral (canonical ball y).
    """
    tick = game_state.get("tick")
    last = _state["last_tick"]
    if isinstance(tick, (int, float)) and isinstance(last, (int, float)) and tick < last:
        lock = _state["z_lateral"]
        reset()  # new game; the ball axis layout does not change
        _state["z_lateral"] = lock
    if isinstance(tick, (int, float)):
        _state["last_tick"] = tick
    gks = []
    for pl in game_state.get("players", []) or []:
        if not isinstance(pl, dict):
            continue
        x, y = _xy(pl.get("position"))
        tol = _CLAMP_TOL * max(_state["max_x"], 1e-6)
        if abs(x) > _state["max_x"] + tol:
            _state["max_x"] = abs(x)
            _state["at_max"] = {}
            _state["line_found"] = False
        if _state["max_x"] > 0 and abs(x) >= _state["max_x"] - tol:
            who = _state["at_max"].setdefault(x > 0, set())
            who.add((pl.get("teamCode", pl.get("teamId")), pl.get("agentId", pl.get("playerId"))))
            if len(who) >= 2:
                _state["line_found"] = True
        _state["max_y"] = max(_state["max_y"], abs(y))
        if _is_gk(pl):
            gks.append(abs(x))
    if gks:
        _state["gk_x"] = max(_state["gk_x"], min(gks))
    _, by = _xy((game_state.get("ball") or {}).get("position"))
    _state["max_y"] = max(_state["max_y"], abs(by))
    gk_term = 0.0 if _state["line_found"] else _state["gk_x"] * _GK_MARGIN
    sx = max(_state["max_x"] * _MARGIN, gk_term, _state["max_y"] / _ASPECT, 1e-3)
    if _state["max_x"] <= 0:
        sx = _FLOOR
    sy = max(_state["max_y"], sx * _ASPECT_MIN)
    return sx, sy


class Frame:
    """Converts between raw coordinates and the normalized attacking frame."""

    def __init__(self, sx: float, sy: float, team_id: int):
        self.sx = sx
        self.half_w = sy / sx          # normalized half-width
        self.d = 1.0 if team_id == 0 else -1.0

    def to_norm(self, pos) -> tuple[float, float]:
        x, y = _xy(pos)
        return x / self.sx * self.d, y / self.sx * self.d

    def to_raw(self, nx: float, ny: float) -> tuple[float, float]:
        nx = max(-1.0, min(1.0, nx))
        ny = max(-self.half_w, min(self.half_w, ny))
        return round(nx * self.sx * self.d, 3), round(ny * self.sx * self.d, 3)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(p, a, b) -> float:
    """Distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return dist(p, (ax + t * dx, ay + t * dy))


def corridor(a, b, others) -> float:
    """Min distance of any point in others to segment a-b (large = open)."""
    return min((seg_dist(o, a, b) for o in others), default=9.0)


def nearest(point, others) -> tuple[float, int]:
    """(distance, index) of the nearest point in others; (9.0, -1) if empty."""
    best, idx = 9.0, -1
    for i, o in enumerate(others):
        d = dist(point, o)
        if d < best:
            best, idx = d, i
    return best, idx
