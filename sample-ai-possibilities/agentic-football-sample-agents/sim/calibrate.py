"""Estimate simulator constants from real tick logs (stub).

Input: lines with JSON tick records ({"kind": "tick", ..., "payload": {"gameState": {...}}}),
for example a CloudWatch export. A line can also be a CloudWatch wrapper with a
"message" field, or have a text prefix before the JSON.

Output: sim/config.calibrated.json with suggested values. sim/config.json is not changed.

Usage:
    python sim/calibrate.py logs1.jsonl [logs2.jsonl ...] [--out sim/config.calibrated.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    for cand in (line, line[line.find("{"):] if "{" in line else ""):
        try:
            rec = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict) and isinstance(rec.get("message"), str):
            return parse_line(rec["message"])
        if isinstance(rec, dict) and rec.get("kind") == "tick":
            return rec
    return None


def _gs(rec):
    p = rec.get("payload") or {}
    if not isinstance(p, dict):
        return None
    gs = p.get("gameState", p)
    return gs if isinstance(gs, dict) and "players" in gs else None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def calibrate(paths):
    states = {}          # (match key, tick) -> gameState, dedup across players
    intervals = []
    last_time = {}
    for path in paths:
        with open(path, errors="replace") as f:
            for line in f:
                rec = parse_line(line)
                if not rec:
                    continue
                gs = _gs(rec)
                if gs is None:
                    continue
                mk = rec.get("label") or rec.get("proc") or path
                tick, gt = gs.get("tick"), gs.get("gameTime")
                states[(path, tick, gt)] = gs
                key = (path, mk, rec.get("team_id"), rec.get("pid"))
                if isinstance(gt, (int, float)):
                    if key in last_time and gt > last_time[key]:
                        intervals.append(gt - last_time[key])
                    last_time[key] = gt

    xs, ys, run, sprint, ball_xs, ball_ys = [], [], [], [], [], []
    for gs in states.values():
        for p in gs.get("players") or []:
            pos = p.get("position") or {}
            xs.append(abs(pos.get("x", 0) or 0))
            ys.append(abs(pos.get("y", 0) or 0))
            v = p.get("speed")
            if not isinstance(v, (int, float)):
                vel = p.get("velocity") or {}
                v = math.hypot(vel.get("x", 0) or 0, vel.get("y", 0) or 0)
            (sprint if p.get("isSprinting") else run).append(v)
        bp = (gs.get("ball") or {}).get("position") or {}
        ball_xs.append(abs(bp.get("x", 0) or 0))
        ball_ys.append(abs(bp.get("y", 0) or 0))

    # ball speed decay between consecutive snapshots with a free, moving ball
    decay = []
    by_file = {}
    for (path, tick, gt), gs in states.items():
        if isinstance(gt, (int, float)):
            by_file.setdefault(path, []).append((gt, gs))
    for seq in by_file.values():
        seq.sort(key=lambda t: t[0])
        for (t1, g1), (t2, g2) in zip(seq, seq[1:]):
            b1, b2 = g1.get("ball") or {}, g2.get("ball") or {}
            if not (b1.get("isFree") and b2.get("isFree")) or not 0 < t2 - t1 < 3:
                continue
            v1 = math.hypot(*(b1.get("velocity", {}).get(k, 0) or 0 for k in ("x", "y")))
            v2 = math.hypot(*(b2.get("velocity", {}).get(k, 0) or 0 for k in ("x", "y")))
            if v1 > 1.0 and 0.05 < v2 < v1:
                decay.append(-math.log(v2 / v1) / (t2 - t1))

    out = {"_samples": {"snapshots": len(states), "decay_pairs": len(decay), "intervals": len(intervals)}}
    if states:
        L = max(max(xs), max(ball_xs))
        W = max(max(ys), max(ball_ys))
        out["half_length"] = round(L, 2)
        out["half_width"] = round(W, 2)
        out["field_scale"] = round(L / 55.0, 4)
        if run:
            out["player_speed"] = round(pct(run, 0.98), 3)
        if sprint:
            out["sprint_speed"] = round(pct(sprint, 0.98), 3)
    if decay:
        out["ball_friction"] = round(statistics.median(decay), 4)
    if intervals:
        out["decision_interval"] = round(statistics.median(intervals), 3)
    out["_note"] = ("Raw estimates. half_length/half_width are max observed |x|/|y| (lower bounds). "
                    "Speeds are 98th percentiles. ball_friction is the median exponential decay rate (1/s). "
                    "If you copy speeds into config.json with field_scale != 1, divide them by field_scale.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "config.calibrated.json"))
    a = ap.parse_args(argv)
    res = calibrate(a.logs)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
