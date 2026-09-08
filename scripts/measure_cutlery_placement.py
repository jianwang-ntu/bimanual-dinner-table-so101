#!/usr/bin/env python3
"""What does placement-randomizing the cutlery cost, and what does it buy?

Criterion T4 is worded "maintains performance under randomized object
placement, weights, friction, shapes, lighting, and background conditions",
and ``scripts/test_randomization_coverage.py`` measured on 2026-09-05 that the
FIRST of those axes covered three of the five graspables: the plate, the mug
and the bottle were jittered, and the fork and the spoon started at the same
x, the same y and the same yaw on all ten seeds -- spread 0.000 mm against
58 / 85 / 119 mm.  Their length and mass varied; their pose did not.

``envs/randomize.py::randomize_cutlery`` closes that.  This probe measures the
change instead of asserting it, because a randomization that is added to a
scene where nothing is grasped can move the score in either direction and the
honest number is whichever one it is.

Swept: ``CUTLERY_PLACE_SPREAD`` in {0.0, 0.5, 1.0} on the same ten seeds and
the same untouched scorer (``envs/task.py`` is not imported by this file for
anything but reading its report).  0.0 is not a quoted baseline, it is a run:
at that setting the two draws collapse to zero and the cutlery sits where the
scene builder authored it, which is the configuration every figure published
before 2026-09-08 was measured at.

Four controls are built into the sweep rather than left to the suite, because
each one can fail in a way that would make the headline meaningless:

  spread_0_is_fixed      at 0.0 the realized cutlery spread must be 0.000 mm.
                         If it is not, the flag does not mean what it says.
  spread_1_moves         at 1.0 it must not be.  A randomizer that silently
                         rejects every draw would otherwise read as "no
                         change" and be reported as a null result.
  upstream_untouched     the plate, the mug, the bottle and the drawer of a
                         given seed must be IDENTICAL in all three arms.  The
                         cutlery is drawn last precisely so this holds; if it
                         does not, the random stream was reshuffled and every
                         difference in the score is unattributable.
  baseline_reproduces    the 0.0 arm must reproduce the published per-sub-goal
                         tally (drawer 10/10, plate 4/10, mug 1/10, cutlery
                         0/10, 15/50).  This is the one control that can catch
                         the change leaking into behaviour it does not name.

Writes evidence/cutlery_placement.json.
Run:  python3 scripts/measure_cutlery_placement.py --seeds 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

BODIES = ("fork", "spoon")
SHIPPED_BEFORE = 0.0        # the setting every pre-2026-09-08 figure used
# The published tally the 0.0 arm has to reproduce.  Source: manifest T1
# `measured.per_goal` and evidence/eval_seeds_scripted.json, both of which
# were recorded before envs/randomize.py::randomize_cutlery existed.
PUBLISHED = {"drawer_open": 10, "fork_placed": 0, "spoon_placed": 0,
             "plate_placed": 4, "mug_placed": 1}


def run(seed: int, spread: float) -> dict:
    from envs import randomize as R
    from envs.task import TaskMonitor, PLACEMENTS
    from envs import controller as C
    from envs import scene_source

    R.CUTLERY_PLACE_SPREAD = float(spread)
    model, data, log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    start = {o: data.xpos[bid(o)].copy() for o in BODIES}
    peak = {o: 0.0 for o in BODIES}

    roll = C.Rollout(model, data)

    def on_step(d):
        for o in BODIES:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    objects = {}
    for b in BODIES:
        end = data.xpos[bid(b)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{b}_placed"][1])]
        objects[b] = {
            "start_xy": [round(float(v), 4) for v in start[b][:2]],
            "lifted_mm": round(peak[b] * 1000, 1),
            "planar_move_mm": round(
                float(np.linalg.norm(end[:2] - start[b][:2])) * 1000, 1),
            "final_to_target_mm": round(
                float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
            "placed": bool(rep["subgoals"][f"{b}_placed"]),
            "draw": log["state"]["cutlery"][b],
        }

    # The upstream draws, verbatim, so `upstream_untouched` compares values and
    # not a hash of a dict whose key order could differ.
    upstream = {k: v for k, v in log["state"].items() if k != "cutlery"}
    upstream["dims"] = log["dims"]

    return {
        "seed": seed,
        "spread": float(spread),
        "shipped_before": abs(float(spread) - SHIPPED_BEFORE) < 1e-12,
        "subgoals": rep["subgoals"],
        "subgoals_met": int(sum(rep["subgoals"].values())),
        "task_success": bool(rep["task_success"]),
        "objects_dropped": rep["objects_dropped"],
        "upstream": upstream,
        "objects": objects,
    }


def _job(a):
    seed, spread = a
    try:
        return run(seed, spread)
    except Exception as exc:                       # keep the sweep going
        return {"seed": seed, "spread": spread,
                "error": f"{type(exc).__name__}: {exc}"}


def _spread_mm(rows, body, axis) -> float:
    v = [r["objects"][body]["start_xy"][axis] for r in rows if "objects" in r]
    return round((max(v) - min(v)) * 1000, 2) if v else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--spreads", default="0.0,0.5,1.0")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default=str(EVID / "cutlery_placement.json"))
    a = ap.parse_args()

    spreads = [float(x) for x in a.spreads.split(",")]
    if not any(abs(s - SHIPPED_BEFORE) < 1e-12 for s in spreads):
        raise SystemExit("spread 0.0 must be in the sweep -- it is the "
                         "configuration every published figure was measured "
                         "at, and it is this probe's only baseline")

    jobs = [(s, sp) for sp in spreads for s in range(a.seeds)]
    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            tag = r.get("error") or f"{r['subgoals_met']}/5"
            print(f"  spread={r['spread']:<4} seed={r['seed']:<2} {tag}")

    variants = []
    by_spread: dict[float, list[dict]] = {}
    for sp in spreads:
        got = [r for r in rows if abs(r["spread"] - sp) < 1e-12 and "objects" in r]
        by_spread[sp] = got
        per_goal = {g: sum(1 for r in got if r["subgoals"][g]) for g in PUBLISHED}
        variants.append({
            "spread": sp,
            "shipped_before": abs(sp - SHIPPED_BEFORE) < 1e-12,
            "n": len(got),
            "subgoals_met_total": sum(r["subgoals_met"] for r in got),
            "subgoals_possible": 5 * len(got),
            "per_goal": per_goal,
            "task_success": sum(1 for r in got if r["task_success"]),
            "fork_start_spread_mm": {"x": _spread_mm(got, "fork", 0),
                                     "y": _spread_mm(got, "fork", 1)},
            "spoon_start_spread_mm": {"x": _spread_mm(got, "spoon", 0),
                                      "y": _spread_mm(got, "spoon", 1)},
            "draws_rejected_to_authored": sum(
                1 for r in got for b in BODIES
                if not r["objects"][b]["draw"]["accepted"]),
            "fork_lifted_mm_median": round(float(np.median(
                [r["objects"]["fork"]["lifted_mm"] for r in got])), 1) if got else None,
            "spoon_lifted_mm_median": round(float(np.median(
                [r["objects"]["spoon"]["lifted_mm"] for r in got])), 1) if got else None,
        })

    base = next(v for v in variants if v["shipped_before"])
    moved = [v for v in variants if not v["shipped_before"]]

    def _both_zero(v):
        return (v["fork_start_spread_mm"] == {"x": 0.0, "y": 0.0}
                and v["spoon_start_spread_mm"] == {"x": 0.0, "y": 0.0})

    # upstream_untouched: compare each seed's non-cutlery draws across arms
    mismatch = []
    ref = {r["seed"]: r["upstream"] for r in by_spread[SHIPPED_BEFORE]}
    for sp, got in by_spread.items():
        for r in got:
            if r["upstream"] != ref.get(r["seed"]):
                mismatch.append({"spread": sp, "seed": r["seed"]})

    controls = {
        "spread_0_is_fixed": _both_zero(base),
        "spread_1_moves": all(not _both_zero(v) for v in moved) if moved else None,
        "upstream_untouched": not mismatch,
        "upstream_mismatches": mismatch,
        "baseline_reproduces": base["per_goal"] == PUBLISHED,
        "baseline_per_goal": base["per_goal"],
        "published_per_goal": PUBLISHED,
        "all_runs_ok": all("objects" in r for r in rows),
    }

    out = {
        "probe": "measure_cutlery_placement.py",
        "question": ("Does placement-randomizing the fork and the spoon inside "
                     "the drawer -- T4's first named axis, previously covering "
                     "three of five graspables -- change what the entry scores?"),
        "finding_id": "F-T4-CUTLERY-PLACEMENT-001",
        "seeds": a.seeds,
        "spreads": spreads,
        "ranges": {"cutlery_xy_m": [-0.016, 0.016],
                   "cutlery_yaw_rad": [-0.20, 0.20]},
        "controls": controls,
        "variants": variants,
        "runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n  spread  subgoals   drawer fork spoon plate mug   fork_spread_mm")
    for v in variants:
        g = v["per_goal"]
        fs = v["fork_start_spread_mm"]
        print(f"  {v['spread']:<7} {v['subgoals_met_total']:>2}/{v['subgoals_possible']:<6} "
              f"{g['drawer_open']:>4} {g['fork_placed']:>5} {g['spoon_placed']:>5} "
              f"{g['plate_placed']:>5} {g['mug_placed']:>4}   x={fs['x']} y={fs['y']}")
    print("\n  controls:")
    for k in ("spread_0_is_fixed", "spread_1_moves", "upstream_untouched",
              "baseline_reproduces", "all_runs_ok"):
        print(f"    {k}: {controls[k]}")
    print(f"\n  wrote {a.out}")
    bad = [k for k in ("spread_0_is_fixed", "spread_1_moves",
                       "upstream_untouched", "baseline_reproduces",
                       "all_runs_ok") if controls[k] is False]
    if bad:
        print(f"  CONTROL FAILED: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
