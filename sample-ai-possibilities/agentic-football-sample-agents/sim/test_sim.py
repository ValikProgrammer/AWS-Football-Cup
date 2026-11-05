"""pytest for the simulator. Run: python -m pytest sim/test_sim.py -q -s"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine  # noqa: E402
import baselines as B  # noqa: E402

F = B.FACTORIES


def test_determinism_by_seed():
    a = engine.run_match(F["stock"], F["aggressive"], 7, randomize_keys=True)
    b = engine.run_match(F["stock"], F["aggressive"], 7, randomize_keys=True)
    assert a == b
    c = engine.run_match(F["random_legal"], F["random_legal"], 7)
    d = engine.run_match(F["random_legal"], F["random_legal"], 7)
    assert c == d


def test_stock_vs_stock_sane():
    r = engine.run_match(F["stock"], F["stock"], 3)
    assert 0 <= r["home"] <= 10 and 0 <= r["away"] <= 10
    st = r["stats"]
    assert st["errors"] == [0, 0]
    assert st["passes"][0] > 0 and st["passes"][1] > 0
    assert sum(st["possession_s"]) > 20


def test_match_cpu_time():
    t = time.process_time()
    for s in range(5):
        engine.run_match(F["stock"], F["stock"], s)
    per = (time.process_time() - t) / 5
    print(f"\nCPU per match: {per * 1000:.0f} ms")
    assert per < 0.2


def test_goals_aggressive_vs_random():
    goals = sum(engine.run_match(F["aggressive"], F["random_legal"], s)["home"] +
                engine.run_match(F["aggressive"], F["random_legal"], s)["away"] for s in range(10))
    assert goals > 0


def test_payload_format():
    seen = {}

    def factory(team_id, seed):
        def fn(payload, tid, pid):
            seen["p"] = payload
            return []
        return fn
    engine.run_match(factory, F["stock"], 0, config_overrides={"match_time": 2.0})
    p = seen["p"]
    gs = p["gameState"]
    assert set(p) == {"gameState", "teamId", "myPlayers"}
    for k in ("tick", "gameTime", "playMode", "score", "ball", "players", "teamChat"):
        assert k in gs
    assert len(gs["players"]) == 10
    assert {"agentId", "teamCode", "position", "velocity", "stamina", "speed", "isSprinting"} <= set(gs["players"][0])


def test_policy_runs_if_present():
    if not B.policy_available():
        import pytest
        pytest.skip("lib/policy.py not available")
    r = engine.run_match(B.policy(), F["stock"], 1)
    assert r["stats"]["errors"][0] == 0


def _table(home, away, seeds=10):
    w = d = l = gf = ga = 0
    for s in range(seeds):
        for side in (0, 1):
            if side == 0:
                r = engine.run_match(home, away, s)
                a, b = r["home"], r["away"]
            else:
                r = engine.run_match(away, home, s)
                a, b = r["away"], r["home"]
            gf += a
            ga += b
            w += a > b
            d += a == b
            l += a < b
    return w, d, l, gf, ga


def test_print_baseline_table():
    rows = [("stock", "stock"), ("stock", "aggressive"), ("stock", "defensive")]
    if B.policy_available():
        rows += [("policy", "stock"), ("policy", "aggressive"), ("policy", "defensive")]
    fac = dict(F, policy=B.policy())
    print("\nteam       vs opp        W  D  L   GF-GA   (10 seeds x 2 sides)")
    for h, a in rows:
        w, d, l, gf, ga = _table(fac[h], fac[a])
        print(f"{h:10s} {a:12s} {w:2d} {d:2d} {l:2d}   {gf:2d}-{ga:2d}")


SMALL_FIELD = {"field_scale": 7.0 / 55.0, "half_width": 3.5 * 55.0 / 7.0}  # L=7, W=3.5 after scaling


def test_policy_small_field():
    """Scale test: the policy must not depend on metres (L=7, W=3.5)."""
    if not B.policy_available():
        import pytest
        pytest.skip("lib/policy.py not available")
    cfg = engine.load_config(SMALL_FIELD)
    assert abs(cfg["half_length"] - 7.0) < 1e-9 and abs(cfg["half_width"] - 3.5) < 1e-9
    big = small = 0
    for s in range(6):
        rb = engine.run_match(B.policy(), F["aggressive"], s)
        big += rb["home"] - rb["away"]
        r = engine.run_match(B.policy(), F["aggressive"], s, config_overrides=SMALL_FIELD)
        assert r["stats"]["errors"][0] == 0
        small += r["home"] - r["away"]
    print(f"\npolicy vs aggressive goal diff: big field {big}, small field {small}")
    assert small > 0
