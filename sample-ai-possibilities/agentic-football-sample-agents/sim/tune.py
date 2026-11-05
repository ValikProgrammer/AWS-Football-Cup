"""Self-play parameter tuner for lib/policy.py.

CMA-ES (falls back to a (1+lambda)-ES without the cma package) over the
numeric values in lib/params.json. Each value is normalized by its start
value: value = start + z * scale, scale = |start| (or 0.1 when start == 0).

Fitness of a candidate = mean over matches vs a pool (stock, aggressive,
defensive, current best), both sides, with domain randomization:
    3*win + 1*draw + 0.3*goal_diff  -  l2 * mean(z^2)

A new best is accepted only if it beats the current best on a fixed
held-out seed set. Output: sim/params.best.json and sim/report.md.
lib/params.json is never written.

Usage:
    python sim/tune.py --budget-sec 600 --matches 24 --workers 14
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engine  # noqa: E402
import baselines as B  # noqa: E402

FIXED_OPPS = ["stock", "aggressive", "defensive"]
# Keys that are times / tick counts and have little meaning in a 120 s sim.
DEFAULT_EXCLUDE = "early_time_s,half_time_s,order_ticks"
L2 = 0.3


# ---------------------------------------------------------------- encoding
class Space:
    def __init__(self, start: dict, exclude=()):
        self.start = dict(start)
        self.keys = [k for k, v in start.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in exclude]
        self.scale = [abs(start[k]) if start[k] != 0 else 0.1 for k in self.keys]
        self.is_int = [isinstance(start[k], int) for k in self.keys]

    def decode(self, z):
        p = dict(self.start)
        for k, s, zi, isint in zip(self.keys, self.scale, z, self.is_int):
            v = self.start[k] + zi * s
            if isint:
                v = max(0, int(round(v)))
            p[k] = v
        return p


# ---------------------------------------------------------------- matches
def _factory(spec):
    kind, params = spec
    if kind == "policy":
        return B.policy(params)
    return B.FACTORIES[kind]


def play(task):
    """task = (cand_spec, opp_spec, seed, side). Returns (points, goal_diff) for the candidate."""
    cand, opp, seed, side = task
    fc, fo = _factory(cand), _factory(opp)
    if side == 0:
        r = engine.run_match(fc, fo, seed, randomize_keys=True)
        gf, ga = r["home"], r["away"]
    else:
        r = engine.run_match(fo, fc, seed, randomize_keys=True)
        gf, ga = r["away"], r["home"]
    pts = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
    return pts, gf - ga


def score_of(results):
    return sum(p + 0.3 * gd for p, gd in results) / max(1, len(results))


def build_tasks(cand_spec, opp_specs, seeds):
    tasks = []
    for i, s in enumerate(seeds):
        opp = opp_specs[i % len(opp_specs)]
        tasks.append((cand_spec, opp, s, i % 2))
    return tasks


def evaluate_many(pool, cand_specs, opp_specs, seeds):
    """Common random numbers: every candidate plays the same seed/opponent/side list."""
    tasks, idx = [], []
    for ci, cs in enumerate(cand_specs):
        t = build_tasks(cs, opp_specs, seeds)
        tasks += t
        idx += [ci] * len(t)
    res = pool.map(play, tasks, chunksize=max(1, len(tasks) // (4 * pool._processes)))
    out = [[] for _ in cand_specs]
    for ci, r in zip(idx, res):
        out[ci].append(r)
    return [score_of(o) for o in out]


def per_opponent(pool, cand_spec, opp_names, best_spec, seeds):
    rows = {}
    for name in opp_names:
        spec = best_spec if name == "best" else (name, None)
        tasks = [(cand_spec, spec, s, side) for s in seeds for side in (0, 1)]
        res = pool.map(play, tasks)
        w = sum(1 for p, _ in res if p == 3.0)
        d = sum(1 for p, _ in res if p == 1.0)
        l = len(res) - w - d
        gd = sum(g for _, g in res)
        rows[name] = dict(points=score_of(res), w=w, d=d, l=l, gd=gd, n=len(res))
    return rows


# ---------------------------------------------------------------- optimizer
class OnePlusLambda:
    """Minimal (1+lambda)-ES with 1/5th-rule step size. Same ask/tell API as cma."""

    def __init__(self, x0, sigma, popsize, seed):
        self.x = list(x0)
        self.sigma = sigma
        self.lam = popsize
        self.rng = random.Random(seed)
        self.fx = None

    def ask(self):
        return [[xi + self.rng.gauss(0, self.sigma) for xi in self.x] for _ in range(self.lam)]

    def tell(self, xs, fs):  # fs: lower is better
        i = min(range(len(fs)), key=lambda j: fs[j])
        if self.fx is None or fs[i] < self.fx:
            self.x, self.fx = list(xs[i]), fs[i]
            self.sigma *= 1.5
        else:
            self.sigma *= 0.82

    @property
    def mean(self):
        return self.x


def make_es(n, sigma, popsize, seed):
    try:
        import cma
        es = cma.CMAEvolutionStrategy([0.0] * n, sigma,
                                      {"popsize": popsize, "seed": seed + 1, "verbose": -9,
                                       "bounds": [-0.9, 2.0]})
        return es, "CMA-ES"
    except Exception as e:
        print(f"cma unavailable ({e!r}); using (1+lambda)-ES")
        return OnePlusLambda([0.0] * n, sigma, popsize, seed), "(1+lambda)-ES"


def _mean(es):
    m = getattr(es, "mean", None)
    if m is None:
        m = es.result.xfavorite
    return list(m)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-sec", type=float, default=600)
    ap.add_argument("--matches", type=int, default=24, help="matches per candidate per generation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--holdout", type=int, default=60, help="held-out matches for acceptance")
    ap.add_argument("--report-seeds", type=int, default=15, help="seeds per opponent (x2 sides) in report")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="comma-separated keys not to tune")
    ap.add_argument("--params", default=os.path.join(B.LIB_DIR, "params.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not B.policy_available():
        sys.exit("lib/policy.py is not importable; nothing to tune")

    t_start = time.time()
    start = B.load_params(args.params)
    space = Space(start, exclude=set(filter(None, args.exclude.split(","))))
    n = len(space.keys)
    print(f"tuning {n} keys: {', '.join(space.keys)}")
    es, algo = make_es(n, args.sigma, args.popsize, args.seed)
    rng = random.Random(args.seed)

    holdout_seeds = list(range(900_000, 900_000 + args.holdout))
    report_seeds = list(range(800_000, 800_000 + args.report_seeds))
    fixed_specs = [(o, None) for o in FIXED_OPPS]

    best_z = [0.0] * n
    best_params = space.decode(best_z)
    history = []

    with Pool(args.workers) as pool:
        def holdout_score(params):
            return evaluate_many(pool, [("policy", params)], fixed_specs, holdout_seeds)[0]

        best_hold = holdout_score(best_params)
        start_hold = best_hold
        print(f"[{time.time() - t_start:6.1f}s] start held-out score {best_hold:.3f} ({algo}, {args.workers} workers)")
        # leave time for the final report (approx. 3 held-out evals)
        t_report = 3 * (time.time() - t_start) + 5
        gen = 0
        while time.time() - t_start < args.budget_sec - t_report:
            gen += 1
            zs = es.ask()
            seeds = [rng.randrange(1, 10**8) for _ in range(args.matches)]
            opp_specs = fixed_specs + [("policy", best_params)]
            specs = [("policy", space.decode(z)) for z in zs]
            fits = evaluate_many(pool, specs, opp_specs, seeds)
            fits = [f - L2 * sum(zi * zi for zi in z) / n for f, z in zip(fits, zs)]
            es.tell(zs, [-f for f in fits])
            i = max(range(len(fits)), key=lambda j: fits[j])
            # challengers: best of generation and the distribution mean
            accepted = False
            for z in (list(zs[i]), _mean(es)):
                cp = space.decode(z)
                h = holdout_score(cp) - L2 * sum(zi * zi for zi in z) / n
                if h > best_hold + 0.02:
                    best_z, best_params, best_hold, accepted = list(z), cp, h, True
                    break
            history.append(dict(gen=gen, t=round(time.time() - t_start, 1), gen_best=round(fits[i], 3),
                                best_hold=round(best_hold, 3), accepted=accepted))
            print(f"[{time.time() - t_start:6.1f}s] gen {gen:3d} gen_best {fits[i]:.3f} "
                  f"held-out best {best_hold:.3f}{'  ACCEPT' if accepted else ''}")

        out_path = os.path.join(HERE, "params.best.json")
        with open(out_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"wrote {out_path}")

        before_spec, after_spec = ("policy", start), ("policy", best_params)
        names = FIXED_OPPS
        before = per_opponent(pool, before_spec, names, None, report_seeds)
        after = per_opponent(pool, after_spec, names, None, report_seeds)
        h2h = per_opponent(pool, after_spec, ["best"], before_spec, report_seeds)["best"]

    write_report(args, algo, space, start, best_params, start_hold, best_hold, before, after, h2h, history,
                 time.time() - t_start)


def write_report(args, algo, space, start, best, start_hold, best_hold, before, after, h2h, history, elapsed):
    L = []
    L.append("# Tuner report\n")
    L.append(f"- Algorithm: {algo}, budget {args.budget_sec:.0f} s (used {elapsed:.0f} s), "
             f"{args.matches} matches/candidate/gen, popsize {args.popsize}, workers {args.workers}")
    L.append(f"- Generations: {len(history)}, accepted: {sum(h['accepted'] for h in history)}")
    L.append(f"- Held-out score (vs stock/aggressive/defensive, {args.holdout} matches, randomized): "
             f"start {start_hold:.3f} -> best {best_hold:.3f}")
    L.append("- Score = mean(3*win + draw + 0.3*goal_diff). Simulator is NOT the AWS engine: "
             "treat gains as hints, check on the real server.\n")
    L.append(f"## Points per match ({args.report_seeds} seeds x 2 sides, domain randomization)\n")
    L.append("| opponent | before | after | before W/D/L GD | after W/D/L GD |")
    L.append("|---|---|---|---|---|")
    for k in before:
        b, a = before[k], after[k]
        L.append(f"| {k} | {b['points']:.2f} | {a['points']:.2f} | {b['w']}/{b['d']}/{b['l']} {b['gd']:+d} | "
                 f"{a['w']}/{a['d']}/{a['l']} {a['gd']:+d} |")
    L.append(f"| tuned vs start (head-to-head) | | {h2h['points']:.2f} | | {h2h['w']}/{h2h['d']}/{h2h['l']} {h2h['gd']:+d} |")
    L.append("\n## Changed params\n")
    L.append("| key | start | best |")
    L.append("|---|---|---|")
    for k in space.keys:
        if best[k] != start[k]:
            bv = best[k] if isinstance(best[k], int) else round(best[k], 4)
            L.append(f"| {k} | {start[k]} | {bv} |")
    L.append("\nBest params: `sim/params.best.json` (lib/params.json is unchanged).\n")
    with open(os.path.join(HERE, "report.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
