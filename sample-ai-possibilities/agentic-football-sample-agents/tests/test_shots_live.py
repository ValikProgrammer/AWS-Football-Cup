"""Real shots from match de47ae46 (we were team 1): wide-post shot blocked, far shot aims CENTER."""
import json
import os

import norm
import policy

FX = os.path.join(os.path.dirname(__file__), "fixtures")
P = policy.load_params()


def _top(name, pid, role):
    pl = json.load(open(os.path.join(FX, name)))
    norm.reset()
    return policy.decide(pl["gameState"], pl["teamId"], pid, role, P)


def test_wide_post_shot_is_blocked():
    # FWD2 at (-5.37, 1.58): outside the goal width, ~1 unit from the line; BL rolled across the mouth
    cands = _top("match_wide_post_fwd2.json", 4, "FWD2")
    assert cands[0].command["commandType"] != "SHOOT", cands[0].reason


def test_far_shot_aims_center_not_corner():
    # FWD1 at (-4.11, -1.61) aimed BR and the ball ended at y=+2.21 beside the far post
    for c in _top("match_far_corner_fwd1.json", 3, "FWD1"):
        if c.command["commandType"] == "SHOOT":
            assert c.command["parameters"]["aim_location"] == "CENTER"
            assert c.command["parameters"]["power"] <= 0.95
