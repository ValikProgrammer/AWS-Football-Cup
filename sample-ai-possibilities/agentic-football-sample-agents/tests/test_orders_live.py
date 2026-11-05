"""Coach orders with the real live teamChat format and timing (match b0bb16cb)."""
import orders


def _msg(team, text):
    return {"senderTeamCode": team, "message": text, "timestamp": 0}


CHAT = [  # (gameTime, message appended at that time)
    (73.4, _msg("home", "фырщще")),
    (75.4, _msg("home", "shoot")),
    (93.4, _msg("home", "strategy all attack")),
    (93.4, _msg("away", "фырцщще")),
    (107.4, _msg("home", "HIGH_PRESS")),
]


def _replay(chat, until=140.0, step=2.0):
    tr, msgs, out, t = orders.OrderTracker(), [], [], 0.0
    while t <= until:
        msgs += [m for (mt, m) in chat if t - step < mt <= t]
        o = tr.update({"gameTime": t, "tick": int(t * 50), "teamChat": list(msgs)}, 0)
        out.append((t, o, tr.strategy_word))
        t += step
    return out


def at(rows, t):
    return min(rows, key=lambda r: abs(r[0] - t))


def test_shoot_waits_for_carrier_at_most_6s():
    rows = _replay(CHAT)
    assert at(rows, 74)[1] == "shoot"            # next decision after the message
    assert at(rows, 80)[1] == "shoot"            # still waiting for a ball carrier
    assert at(rows, 86)[1] is None               # gone after ~6 s


def test_other_action_words_are_short():
    rows = _replay([(10.0, _msg("home", "press"))], until=20)
    assert at(rows, 10)[1] == "press" and at(rows, 14)[1] is None


def test_consume_clears_shoot_and_pass():
    tr = orders.OrderTracker()
    tr.update({"gameTime": 10.0, "teamChat": [_msg("home", "C1")]}, 0)
    tr.consume("MOVE_TO")
    assert tr.action == "shoot"
    tr.consume("SHOOT")
    assert tr.action is None
    tr.update({"gameTime": 12.0, "teamChat": [_msg("home", "C1"), _msg("home", "C6")]}, 0)
    tr.consume("GK_DISTRIBUTE")
    assert tr.action is None


def test_new_message_replaces_action():
    rows = _replay(CHAT)
    assert at(rows, 94)[1] == "attack"


def test_strategy_is_sticky():
    rows = _replay(CHAT)
    assert at(rows, 108)[2] == "highpress"
    assert at(rows, 140)[2] == "highpress"       # still active at the end
    assert orders.strategy_for(at(rows, 140)[2]) == "HIGH_PRESS"


def test_away_message_ignored():
    texts = orders.my_texts([_msg("home", "a"), _msg("away", "b")], 0)
    assert texts == ["a"]
    assert orders.my_texts([_msg("home", "a"), _msg("away", "b")], 1) == ["b"]


def test_same_text_twice_retriggers():
    chat = [(10.0, _msg("home", "shoot")), (60.0, _msg("home", "shoot"))]
    rows = _replay(chat, until=80)
    assert at(rows, 50)[1] is None and at(rows, 60)[1] == "shoot"


def test_codes_and_layout():
    cases = {"S1": "balanced", "S2": "highpress", "s 3": "counter", "S4": "lowblock", "S5": "possession",
             "S6": "allout", "C1": "shoot", "c2": "attack", "C3": "defend", "C4": "press", "C5": "mark",
             "C6": "pass", "C7": "tackle", "S0": "auto", "ы0": "auto", "ы2": "highpress", "с1": "shoot", "фырщще": "shoot", "C8": None}
    for text, want in cases.items():
        assert orders.match_order(text) == want, text


def test_s0_releases_strategy_to_manager():
    chat = [(10.0, _msg("home", "S2")), (40.0, _msg("home", "S0"))]
    rows = _replay(chat, until=60)
    assert at(rows, 30)[2] == "highpress"
    assert at(rows, 50)[2] is None
