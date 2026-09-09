#!/usr/bin/env python3
"""Where does the mug actually stop, and why is it not on its mat?

``mug_placed`` fires on 1 of 10 seeds while the controller's own comment says
the mug is *gripped* on 8 of 10.  Those two numbers can only both be true if
the grasp works and the transport does not, so this measures the transport:
the mug's distance to ``target_mug`` sampled through the whole rollout, split
at the boundaries of the two shunt blocks, plus the three things the scorer
asks of it at the end -- planar distance, resting height, uprightness.

It changes nothing.  It runs the shipped script unmodified and reads the
simulator, so a number here is the number the scorer would have seen.

Run:  PYTHONPATH=. python3 scripts/measure_mug_shunt.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from envs.randomize import make_env                            # noqa: E402
from envs.task import TaskMonitor, TABLE_TOP_Z, PLACEMENTS, UPRIGHT_COS  # noqa: E402
from envs.controller import run_dinner_table                   # noqa: E402
from envs import scene_source                                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def track(model, data, body, site):
    """Planar distance, height and uprightness of ``body`` right now."""
    b, s = _bid(model, body), _sid(model, site)
    d = float(np.linalg.norm(data.xpos[b][:2] - data.site_xpos[s][:2]))
    return {
        "dist_mm": round(d * 1000, 1),
        "z": round(float(data.xpos[b][2]), 4),
        "rest_err_mm": round(abs(float(data.xpos[b][2]) - TABLE_TOP_Z) * 1000, 1),
        "upright_cos": round(float(data.xmat[b].reshape(3, 3)[2, 2]), 3),
    }


def episode(seed: int, bodies) -> dict:
    model, data, log = make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    start = {b: track(model, data, b, PLACEMENTS[f"{b}_placed"][1]) for b in bodies}
    series = {b: [] for b in bodies}

    def sample(d):
        for b in bodies:
            series[b].append((round(float(d.time), 3),
                              track(model, d, b, PLACEMENTS[f"{b}_placed"][1])))

    roll = run_dinner_table(model, data, monitor=mon, on_step=sample)
    rep = mon.report(data)
    trace = roll["trace"]

    out = {"seed": seed, "subgoals": rep["subgoals"],
           "subgoals_met": rep["subgoals_met"], "bodies": {}}

    for b in bodies:
        tol_m = PLACEMENTS[f"{b}_placed"][2]
        ser = series[b]
        dists = [(t, v["dist_mm"]) for t, v in ser]
        best_t, best = min(dists, key=lambda p: p[1])
        final = track(model, data, b, PLACEMENTS[f"{b}_placed"][1])
        # what the scorer's three clauses say about the final pose, separately
        near = bool(final["dist_mm"] <= tol_m * 1000)
        resting = bool(final["rest_err_mm"] < 90.0)
        upright = bool(final["upright_cos"] >= UPRIGHT_COS
                       if b in ("plate", "mug") else True)
        # distance at the end of each labelled block that names this body
        marks = {}
        for i, tr in enumerate(trace):
            lab = tr.get("label") or ""
            if b not in lab:
                continue
            t0 = tr["t"]
            nxt = next((x["t"] for x in trace[i + 1:] if x["t"] > t0), None)
            cut = nxt if nxt is not None else ser[-1][0]
            at = min(ser, key=lambda p: abs(p[0] - cut))
            marks[lab] = {"t": t0, "dist_mm_after": at[1]["dist_mm"],
                          "upright_cos_after": at[1]["upright_cos"]}
        # the first instant the body stops being upright, and what was
        # commanded then -- a tip and an inversion are different failures and
        # only the timeline tells them apart
        lost = next(((t, v["upright_cos"]) for t, v in ser
                     if v["upright_cos"] < UPRIGHT_COS), None)
        during = None
        if lost is not None:
            during = max((tr for tr in trace if tr["t"] <= lost[0]),
                         key=lambda tr: tr["t"], default=None)
        out["bodies"][b] = {
            "upright_lost_at_t": None if lost is None else lost[0],
            "upright_cos_when_lost": None if lost is None else lost[1],
            "upright_lost_during_move": None if during is None else {
                "label": during["label"], "arm": during["arm"],
                "t": during["t"]},
            "tol_mm": round(tol_m * 1000, 1),
            "start": start[b],
            "closest_mm": best, "closest_t": best_t,
            "final": final,
            "scorer_clauses": {"near": near, "resting": resting,
                               "upright": upright,
                               "dropped": bool(b in rep["objects_dropped"])},
            "held_by_arms": sorted(mon.touched[b]),
            "was_gripped": bool(mon.touched[b]),
            "moved_mm": round(abs(start[b]["dist_mm"] - final["dist_mm"]), 1),
            "block_marks": marks,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only", default="",
                    help="comma-separated seeds to run instead of range(--seeds)")
    ap.add_argument("--bodies", default="mug,plate")
    ap.add_argument("--out", default="evidence/mug_shunt.json")
    a = ap.parse_args()

    bodies = [b for b in a.bodies.split(",") if b]
    eps = []
    which = ([int(x) for x in a.only.split(",") if x.strip()]
             if a.only else list(range(a.seeds)))
    for s in which:
        e = episode(s, bodies)
        eps.append(e)
        line = " ".join(
            f"{b}: start {e['bodies'][b]['start']['dist_mm']:>6.1f}mm "
            f"-> best {e['bodies'][b]['closest_mm']:>6.1f} "
            f"-> final {e['bodies'][b]['final']['dist_mm']:>6.1f} "
            f"({'HIT' if e['bodies'][b]['scorer_clauses']['near'] else 'short'}"
            f"{'' if e['bodies'][b]['scorer_clauses']['upright'] else ',TIPPED'}"
            f"{'' if e['bodies'][b]['scorer_clauses']['resting'] else ',AIRBORNE'}"
            f"{'' if e['bodies'][b]['was_gripped'] else ',NEVER-GRIPPED'})"
            for b in bodies)
        print(f"seed {s}  {line}", flush=True)

    summary = {}
    for b in bodies:
        summary[b] = {
            "gripped_seeds": sum(1 for e in eps if e["bodies"][b]["was_gripped"]),
            "near_at_end": sum(1 for e in eps
                               if e["bodies"][b]["scorer_clauses"]["near"]),
            "near_at_some_point": sum(
                1 for e in eps
                if e["bodies"][b]["closest_mm"] <= e["bodies"][b]["tol_mm"]),
            "upright_at_end": sum(1 for e in eps
                                  if e["bodies"][b]["scorer_clauses"]["upright"]),
            "median_final_mm": round(float(np.median(
                [e["bodies"][b]["final"]["dist_mm"] for e in eps])), 1),
            "median_start_mm": round(float(np.median(
                [e["bodies"][b]["start"]["dist_mm"] for e in eps])), 1),
            "median_closest_mm": round(float(np.median(
                [e["bodies"][b]["closest_mm"] for e in eps])), 1),
        }

    doc = {"probe": "measure_mug_shunt",
           "what": "distance of each body to its target through the shipped "
                   "rollout, and the scorer's three clauses at the end",
           "policy": "scripted, unmodified", "scene": "privileged",
           "seeds": which, "summary": summary, "episodes": eps}
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"-> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
