"""Team-wide strategy presets: partial overrides of params.json values.

apply(params, name) returns a new dict. An unknown name means BALANCED.
Values are absolute (not deltas), so the result does not depend on order.
"""

DEFAULT = "BALANCED"

PRESETS: dict[str, dict] = {
    "BALANCED": {},
    "POSSESSION": {
        "pass_bonus": 0.12,          # prefer passes over dribbles
        "risk_weight": 0.8,          # avoid risky passes near own goal
        "shoot_range": 0.35,         # fewer long shots
        "shot_corridor": 0.08,
        "forward_pass_weight": 0.25,
    },
    "COUNTER": {
        "line_height": -0.2,         # deep line
        "def_max_x": -0.25,
        "forward_pass_weight": 0.7,  # fast forward balls after winning it
        "aerial_dist": 0.6,          # long balls go AERIAL earlier
        "through_ahead": 0.05,       # THROUGH as soon as the receiver is ahead
        "press_max_x": 0.3,          # do not chase high up the pitch
    },
    "HIGH_PRESS": {
        "press_bodies": 2,
        "second_press_dist": 0.35,
        "press_intensity": 1.0,      # sprint press
        "line_height": 0.2,
        "def_max_x": 0.1,
        "stamina_sprint_min": 0.2,
        "early_time_s": 0,
    },
    "LOW_BLOCK": {
        "line_height": -0.3,
        "def_max_x": -0.35,
        "press_bodies": 1,           # only one presser
        "press_max_x": 0.0,          # press only in our half
        "clear_zone": -0.2,          # clear the ball under pressure
        "pressure_radius": 0.18,
        "risk_weight": 0.9,
    },
    "ALL_OUT": {
        "line_height": 0.3,
        "def_max_x": 0.2,
        "shoot_range": 0.65,         # shoot from further out
        "shot_corridor": 0.03,
        "shot_bonus": 0.3,
        "forward_pass_weight": 0.55,
        "risk_weight": 0.3,
        "press_bodies": 2,
    },
}

DESCRIPTIONS: dict[str, str] = {
    "BALANCED": "default mix of attack and defence",
    "POSSESSION": "keep the ball, safe passes, fewer long shots",
    "COUNTER": "deep line, fast forward passes after winning the ball",
    "HIGH_PRESS": "two pressers, high line, press hard near their goal",
    "LOW_BLOCK": "low compact line, one presser, clear under pressure",
    "ALL_OUT": "high line, shoot often, all forward (use when losing late)",
}

NAMES = tuple(PRESETS)


def normalize(name) -> str | None:
    """Canonical preset name or None. Accepts any case, spaces, dashes."""
    if not isinstance(name, str):
        return None
    key = name.strip().upper().replace("-", "_").replace(" ", "_")
    if key in PRESETS:
        return key
    flat = key.replace("_", "")
    for n in NAMES:
        if n.replace("_", "") == flat:
            return n
    return None


def apply(params: dict, name) -> dict:
    """Copy of params with the preset overrides. Unknown name -> BALANCED."""
    out = dict(params)
    out.update(PRESETS.get(normalize(name) or DEFAULT, {}))
    return out
