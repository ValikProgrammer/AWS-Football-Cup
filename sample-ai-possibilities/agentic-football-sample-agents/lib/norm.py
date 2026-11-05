"""Field-scale estimation and normalized geometry.

Normalized frame: x and y are both divided by the estimated half-length sx,
then rotated 180 degrees for the away team, so +x is always the opponent
goal and +y is always the attacker's left. Opponent goal = (1, 0).
"""

import math

_ASPECT = 0.64          # half-width / half-length of a typical pitch
_MARGIN = 1.08          # GK stands slightly inside the goal line
_FLOOR = 0.5

_state = {"max_x": 0.0, "max_y": 0.0, "last_tick": None}


def reset() -> None:
    _state.update(max_x=0.0, max_y=0.0, last_tick=None)


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


def update_scale(game_state: dict) -> tuple[float, float]:
    """Update the running max with this tick; return (sx, sy) in raw units."""
    tick = game_state.get("tick")
    last = _state["last_tick"]
    if isinstance(tick, (int, float)) and isinstance(last, (int, float)) and tick < last:
        reset()  # new game
    if isinstance(tick, (int, float)):
        _state["last_tick"] = tick
    pts = [p.get("position") for p in game_state.get("players", []) or []]
    pts.append((game_state.get("ball") or {}).get("position"))
    for pos in pts:
        x, y = _xy(pos)
        _state["max_x"] = max(_state["max_x"], abs(x))
        _state["max_y"] = max(_state["max_y"], abs(y))
    sx = max(_state["max_x"] * _MARGIN, _state["max_y"] / _ASPECT, 1e-3)
    if _state["max_x"] <= 0:
        sx = _FLOOR
    sy = max(_state["max_y"], sx * _ASPECT)
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
