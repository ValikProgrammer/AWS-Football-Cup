"""Coach one-word orders from teamChat, with typo-tolerant matching."""

import re
from difflib import SequenceMatcher

VOCAB = ("shoot", "attack", "defend", "press", "mark", "pass")
THRESHOLD = 0.6

_TEXT_KEYS = ("text", "message", "content", "msg", "order")
_TEAM_KEYS = ("teamId", "team", "teamCode", "team_id", "sender", "from")


def match_order(text) -> str | None:
    """Best vocab word for any token in text, or None."""
    if not isinstance(text, str):
        return None
    best, best_s = None, 0.0
    for tok in re.findall(r"[a-z]+", text.lower()):
        if len(tok) < 3:
            continue
        for w in VOCAB:
            s = SequenceMatcher(None, tok, w).ratio()
            pre = 0
            for a, b in zip(tok, w):
                if a != b:
                    break
                pre += 1
            s += 0.1 * pre / len(w)
            if s > best_s:
                best, best_s = w, s
    return best if best_s >= THRESHOLD else None


def _team_ok(msg: dict, team_id: int) -> bool:
    for k in _TEAM_KEYS:
        if k not in msg:
            continue
        v = msg[k]
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v == team_id
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("0", "1"):
                return int(s) == team_id
            if "home" in s or "away" in s:
                return ("home" if team_id == 0 else "away") in s
    return True  # no team info: assume it is for us


def latest_text(team_chat, team_id: int):
    """(key, text) of the latest message for my team, or (None, None)."""
    if isinstance(team_chat, (str, dict)):
        team_chat = [team_chat]
    if not isinstance(team_chat, list):
        return None, None
    for msg in reversed(team_chat):
        if isinstance(msg, str):
            return repr(msg), msg
        if isinstance(msg, dict) and _team_ok(msg, team_id):
            for k in _TEXT_KEYS:
                if isinstance(msg.get(k), str):
                    return repr(sorted(msg.items(), key=lambda kv: str(kv[0]))), msg[k]
    return None, None


class OrderTracker:
    """Holds the active order for order_ticks ticks after a new message."""

    def __init__(self):
        self.last_key = None
        self.order = None
        self.expires = -1
        self.calls = 0

    def update(self, game_state: dict, team_id: int, order_ticks: int = 20) -> str | None:
        self.calls += 1
        tick = game_state.get("tick")
        now = tick if isinstance(tick, (int, float)) else self.calls
        key, text = latest_text(game_state.get("teamChat"), team_id)
        if key is not None and key != self.last_key:
            self.last_key = key
            o = match_order(text)
            if o:
                self.order = o
                self.expires = now + order_ticks
        if self.order and now > self.expires:
            self.order = None
        return self.order


TRACKER = OrderTracker()
