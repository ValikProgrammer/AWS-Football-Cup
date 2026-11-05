"""Coach one-word orders from teamChat, with typo-tolerant matching."""

import re
from difflib import SequenceMatcher

VOCAB = ("shoot", "attack", "defend", "press", "mark", "pass", "tackle")
# Tactic-change words -> strategy preset names (strategies.py).
STRATEGY_WORDS = {
    "balanced": "BALANCED", "possession": "POSSESSION", "counter": "COUNTER",
    "highpress": "HIGH_PRESS", "lowblock": "LOW_BLOCK", "allout": "ALL_OUT",
    "shootonsight": "SHOOT_ON_SIGHT",
}
ALIASES = {"park": "lowblock"}   # "park the bus"
AUTO = "auto"                     # coach releases the strategy back to the GK manager (S0)
_ALL_WORDS = VOCAB + tuple(STRATEGY_WORDS) + tuple(ALIASES) + (AUTO,)
THRESHOLD = 0.6

# Short codes: S<n> = strategy, C<n> = one-word command (case-insensitive, "S 2" ok).
CODES = {
    "s0": "auto", "s1": "balanced", "s2": "highpress", "s3": "counter", "s4": "lowblock", "s5": "possession", "s6": "allout", "s7": "shootonsight",
    "c1": "shoot", "c2": "attack", "c3": "defend", "c4": "press", "c5": "mark", "c6": "pass", "c7": "tackle",
}
_CODE_RE = re.compile(r"(?<![a-z0-9])([sc])\s*([0-9])(?![0-9])")

# Text typed on the Russian keyboard layout -> the Latin keys ("фырщще" -> "ashoot", "ы2" -> "s2").
_RU = "йцукенгшщзхъфывапролджэячсмитьбюё"
_EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
_LAYOUT = str.maketrans(_RU + _RU.upper(), _EN + _EN.upper())


def _delayout(text: str) -> str:
    return text.translate(_LAYOUT)


def strategy_for(order) -> str | None:
    """Strategy preset name for a tactic-change order word, else None."""
    return STRATEGY_WORDS.get(order) if isinstance(order, str) else None

_TEXT_KEYS = ("text", "message", "content", "msg", "order")
_TEAM_KEYS = ("senderTeamCode", "teamId", "team", "teamCode", "team_id", "sender", "from")


def match_order(text) -> str | None:
    """Best vocab word for any token in text, or None."""
    if not isinstance(text, str):
        return None
    text = _delayout(text).lower()
    m = _CODE_RE.search(text)
    if m:
        return CODES.get(m.group(1) + m.group(2))
    words = re.findall(r"[a-z]+", text)
    # adjacent pairs too, so "high press" / "low block" / "all out" match as one word
    toks = words + [a + b for a, b in zip(words, words[1:])]
    best, best_key = None, (0.0, 0)
    for tok in toks:
        if len(tok) < 3:
            continue
        for w in _ALL_WORDS:
            s = SequenceMatcher(None, tok, w).ratio()
            pre = 0
            for a, b in zip(tok, w):
                if a != b:
                    break
                pre += 1
            s += 0.1 * pre / len(w)
            key = (round(s, 6), len(w))     # tie: the longer word wins ("highpress" over "press")
            if key > best_key:
                best, best_key = w, key
    if best_key[0] < THRESHOLD:
        return None
    return ALIASES.get(best, best)


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


def my_texts(team_chat, team_id: int) -> list:
    """Texts of all chat messages for my team, oldest first."""
    if isinstance(team_chat, (str, dict)):
        team_chat = [team_chat]
    if not isinstance(team_chat, list):
        return []
    out = []
    for msg in team_chat:
        if isinstance(msg, str):
            out.append(msg)
        elif isinstance(msg, dict) and _team_ok(msg, team_id):
            for k in _TEXT_KEYS:
                if isinstance(msg.get(k), str):
                    out.append(msg[k])
                    break
    return out


class OrderTracker:
    """Coach orders from teamChat.

    - A new message = the count of my team's messages grew (timestamps are 0 in
      the live game, and the same text may be sent twice on purpose).
    - Strategy words (S1..S6, highpress, counter, ...) are sticky: they hold until
      the next strategy word, so the GK manager cannot overwrite them.
    - Action words (C1..C6, shoot, attack, ...) hold for ORDER_SECONDS of game time
      (about the next decision only).
      (Old bug: expiry was counted in game ticks, ~100 per decision, so an order
      lasted a single decision.)
    update() returns the active action word, else the sticky strategy word.
    """
    ORDER_SECONDS = 2.5   # about one decision (~2 s apart)
    HOLD_UNTIL_DONE = {"shoot": ("SHOOT",), "pass": ("PASS", "GK_DISTRIBUTE")}
    HOLD_SECONDS = 6.0    # shoot / pass wait for our next ball carrier, at most this long

    def __init__(self):
        self.seen = 0
        self.action = None
        self.action_until = -1.0
        self.strategy_word = None
        self.calls = 0
        self.last_t = None

    def _now(self, game_state):
        t = game_state.get("gameTime")
        return float(t) if isinstance(t, (int, float)) else self.calls * 2.0

    def update(self, game_state: dict, team_id: int, order_ticks: int = 20) -> str | None:
        self.calls += 1
        now = self._now(game_state)
        if self.last_t is not None and now < self.last_t - 30.0:   # clock restarted: new match
            self.__init__()
            self.calls = 1
        self.last_t = now
        texts = my_texts(game_state.get("teamChat"), team_id)
        if len(texts) < self.seen:          # chat was trimmed by the server
            self.seen = len(texts)
        for text in texts[self.seen:]:
            o = match_order(text)
            if o == AUTO:
                self.strategy_word = None
            elif o in STRATEGY_WORDS:
                self.strategy_word = o
            elif o:
                self.action = o
                self.action_until = now + (self.HOLD_SECONDS if o in self.HOLD_UNTIL_DONE else self.ORDER_SECONDS)
        self.seen = len(texts)
        if self.action and now > self.action_until:
            self.action = None
        return self.action or self.strategy_word

    def consume(self, command_type) -> None:
        """Clear a shoot/pass order once this agent has carried it out."""
        if command_type in self.HOLD_UNTIL_DONE.get(self.action, ()):
            self.action = None


TRACKER = OrderTracker()
