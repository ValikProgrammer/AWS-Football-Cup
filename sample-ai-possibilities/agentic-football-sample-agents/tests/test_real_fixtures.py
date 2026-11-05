"""Regression tests on real payloads logged from the scrimmage (fitness check)."""
import glob
import json
import os

import norm
import policy

FX = os.path.join(os.path.dirname(__file__), "fixtures")
ROLES = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}
PASSIVE = {"CLEAR_OVERRIDE", "SET_STANCE", "RESET"}


def _payloads():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(FX, "real_*.json")))]


def test_fixtures_exist():
    assert len(_payloads()) >= 5


def test_every_role_active_on_real_ticks():
    P = policy.load_params()
    for pid, role in ROLES.items():
        active = 0
        pls = _payloads()
        norm.reset()
        for pl in pls:
            cmd = policy.decide(pl["gameState"], pl["teamId"], pid, role, P)[0].command
            active += cmd["commandType"] not in PASSIVE
        assert active / len(pls) > 0.5, (role, active, len(pls))


def test_mid_with_ball_passes_on_real_tick1():
    P = policy.load_params()
    pl = json.load(open(os.path.join(FX, "real_mid_tick1_3.json")))
    norm.reset()
    cmd = policy.decide(pl["gameState"], pl["teamId"], 2, "MID", P)[0].command
    assert cmd["commandType"] == "PASS", cmd
    assert isinstance(cmd["parameters"].get("target_player_id"), int)
