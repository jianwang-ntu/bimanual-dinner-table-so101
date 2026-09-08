#!/usr/bin/env python3
"""The one cell of the cutlery grid nobody has run: descend height CROSSED
with the two knobs that each halved the arrival gap on their own.

Eight explanations for ``fork_placed`` / ``spoon_placed`` 0 / 10 have been
tested and refuted (TECHNICAL_SUMMARY sections 8, 8a, 8b).  Every one of them
was swept ONE AT A TIME, with the other knobs left at their shipped values:

  * ``scripts/measure_fork_descent.py``  swept ``CUTLERY_DESCEND_Z`` (0.003 and
    0.006 m) x steps x square -- at the shipped opening and the shipped
    ``plan_at``.
  * ``scripts/measure_gripper_envelope.py`` swept ``CUTLERY_DESCEND_OPENING`` x
    ``CUTLERY_PLAN_AT_OPEN`` -- at the shipped ``CUTLERY_DESCEND_Z`` of 0.003 m.

So the two knobs that MOVED the arrival gap most have never been crossed with
each other, and never with the descent height at all:

  | knob                       | best arrival it reached | measured at        |
  |----------------------------|-------------------------|--------------------|
  | opening 0.00 rad           | 11.0 mm (from 24.5)     | descend_z = 0.003  |
  | plan_at_open = True        | 14.75 mm (from 28.2)    | descend_z = 0.003  |
  | descend_z = 0.006          | swept                   | shipped opening    |

``scripts/measure_grasp_feasibility.py`` reports that the FORK has a pose 6.0 mm
above the asked height that clears the woodwork on 3/3 seeds and grips.  The
shipped controller asks 3.0 mm.  If the fork's descent is an execution failure
against a known-good pose -- which is what this repository's own record says --
then the cell that asks for the known-good height, with the executed hand no
longer displaced 21.83 mm sideways from the solved one, is the cell that has
the best claim on it.  It has never been run.

This probe runs it, and 23 others around it.  Nothing here changes shipped
behaviour: the constants are module globals read at ``dinner_table_script()``
call time and are set per variant inside a worker process.

The shipped cell (0.003 m, plan_at False, GRIPPER_NARROW) is carried IN the
grid rather than quoted from an earlier file, so the comparison is inside one
process pool and one code state, and so the probe can fail: if the shipped cell
does not reproduce fork 0/10, spoon 0/10, the run is not trustworthy and the
controls say so.

Writes evidence/cutlery_zcross.json.
Run:  python3 scripts/measure_cutlery_zcross.py --seeds 10 --workers 48
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
DESCEND = {"fork": "fork_descend", "spoon": "spoon_descend"}

SHIPPED_Z = 0.003
SHIPPED_PLAN_AT = False
SHIPPED_OPENING = None            # None == GRIPPER_NARROW, whatever it is

ZS = (0.003, 0.006, 0.010, 0.015)
PLAN_MODES = (False, True)
OPENINGS = (None, 0.30, 0.15)


def _geom_name(model, g: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{g}"


def _arm_geoms(model, arm: str) -> set[int]:
    out = set()
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                              int(model.geom_bodyid[g])) or ""
        if b.startswith(f"{arm}_"):
            out.add(g)
    return out


def run(seed: int, z: float, plan_at_open: bool, opening) -> dict:
    from envs.randomize import make_env
    from envs.task import TaskMonitor, PLACEMENTS
    from envs import controller as C
    from envs import scene_source

    model, data, _ = make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    start = {o: data.xpos[bid(o)].copy() for o in BODIES}

    C.CUTLERY_DESCEND_Z = float(z)
    C.CUTLERY_PLAN_AT_OPEN = bool(plan_at_open)
    C.CUTLERY_DESCEND_OPENING = (C.GRIPPER_NARROW if opening is None
                                 else float(opening))
    script = C.dinner_table_script()
    roll = C.Rollout(model, data)

    mine = {a: _arm_geoms(model, a) for a in ("left", "right")}
    live = {"label": "", "arm": "", "body": None}
    asked, lowest = {}, {}
    peak = {o: 0.0 for o in BODIES}
    tally: dict[str, dict[str, int]] = {}
    orig = roll._plan

    def plan(mv):
        lb = getattr(mv, "label", "") or ""
        live["label"], live["arm"] = lb, mv.arm
        live["body"] = next((b for b in BODIES if lb == DESCEND[b]), None)
        b = live["body"]
        if b is not None and mv.where is not None:
            asked[b] = np.asarray(mv.where(model, data), float)
        return orig(mv)

    roll._plan = plan

    def on_step(d):
        for o in BODIES:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))
        b = live["body"]
        if b is None:
            return
        arm = live["arm"]
        g = roll.grips[arm]
        lowest[b] = min(lowest.get(b, 9.9), float(g.tip_mid(model, d)[2]))
        t = tally.setdefault(b, {})
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if (g1 in mine[arm]) == (g2 in mine[arm]):
                continue
            hand, other = (g1, g2) if g1 in mine[arm] else (g2, g1)
            pair = f"{_geom_name(model, hand)} | {_geom_name(model, other)}"
            t[pair] = t.get(pair, 0) + 1

    roll.run(script, monitor=mon, on_step=on_step)
    rep = mon.report(data)

    out = {}
    for b in BODIES:
        end = data.xpos[bid(b)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{b}_placed"][1])]
        rec = {"placed": bool(rep["subgoals"][f"{b}_placed"]),
               "lifted_mm": round(peak[b] * 1000, 1),
               "planar_move_mm": round(
                   float(np.linalg.norm(end[:2] - start[b][:2])) * 1000, 1),
               "final_to_target_mm": round(
                   float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1)}
        if b in asked and b in lowest:
            rec["asked_tip_z_m"] = round(float(asked[b][2]), 4)
            rec["lowest_tip_z_m"] = round(lowest[b], 4)
            rec["stalled_above_target_mm"] = round(
                (lowest[b] - float(asked[b][2])) * 1000, 1)
        rec["contacts"] = [{"pair": p, "steps": s} for p, s in
                           sorted(tally.get(b, {}).items(),
                                  key=lambda kv: -kv[1])[:3]]
        out[b] = rec

    return {"seed": seed, "descend_z_m": float(z),
            "plan_at_open": bool(plan_at_open),
            "opening": None if opening is None else float(opening),
            "opening_rad_effective": float(C.CUTLERY_DESCEND_OPENING),
            "shipped": (abs(float(z) - SHIPPED_Z) < 1e-12
                        and bool(plan_at_open) == SHIPPED_PLAN_AT
                        and opening is SHIPPED_OPENING),
            "subgoals_met": int(sum(rep["subgoals"].values())),
            "subgoals": rep["subgoals"],
            "bodies": out}


def _job(a):
    s, z, p, o = a
    try:
        return run(s, z, p, o)
    except Exception as exc:                                   # pragma: no cover
        return {"seed": s, "descend_z_m": z, "plan_at_open": p, "opening": o,
                "error": f"{type(exc).__name__}: {exc}"}


def _med(v):
    return round(float(np.median(v)), 2) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=str(EVID / "cutlery_zcross.json"))
    a = ap.parse_args()

    grid = [(s, z, p, o) for z in ZS for p in PLAN_MODES for o in OPENINGS
            for s in range(a.seeds)]
    runs = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, grid):
            runs.append(r)
            if "error" in r:
                print(f"  ERROR {r['seed']} z={r['descend_z_m']}: {r['error']}")

    good = [r for r in runs if "error" not in r]
    cells = []
    for z in ZS:
        for p in PLAN_MODES:
            for o in OPENINGS:
                got = [r for r in good
                       if r["descend_z_m"] == z and r["plan_at_open"] == p
                       and r["opening"] == (None if o is None else float(o))]
                if not got:
                    continue
                stalls = [r["bodies"]["fork"]["stalled_above_target_mm"]
                          for r in got
                          if "stalled_above_target_mm" in r["bodies"]["fork"]]
                pairs: dict[str, int] = {}
                for r in got:
                    for c in r["bodies"]["fork"]["contacts"]:
                        pairs[c["pair"]] = pairs.get(c["pair"], 0) + c["steps"]
                cells.append({
                    "descend_z_mm": round(z * 1000, 1),
                    "plan_at_open": p,
                    "opening": o,
                    "opening_rad_effective": got[0]["opening_rad_effective"],
                    "shipped": bool(got[0]["shipped"]),
                    "n": len(got),
                    "fork_placed": sum(r["bodies"]["fork"]["placed"] for r in got),
                    "spoon_placed": sum(r["bodies"]["spoon"]["placed"] for r in got),
                    "subgoals_met_total": sum(r["subgoals_met"] for r in got),
                    "fork_stall_mm_median": _med(stalls),
                    "fork_lift_mm_median": _med(
                        [r["bodies"]["fork"]["lifted_mm"] for r in got]),
                    "spoon_lift_mm_median": _med(
                        [r["bodies"]["spoon"]["lifted_mm"] for r in got]),
                    "fork_top_contact": max(pairs, key=pairs.get) if pairs else None,
                })

    shipped = next((c for c in cells if c["shipped"]), None)
    placed = [c for c in cells if c["fork_placed"] + c["spoon_placed"] > 0]
    best = min((c for c in cells if c["fork_stall_mm_median"] is not None),
               key=lambda c: c["fork_stall_mm_median"], default=None)
    best_score = max(cells, key=lambda c: c["subgoals_met_total"]) if cells else None

    doc = {
        "probe": "measure_cutlery_zcross.py",
        "question": "Do the descent height and the two knobs that each halved "
                    "the arrival gap place the cutlery when crossed, having "
                    "only ever been swept one at a time?",
        "finding_id": "H-CUTLERY-ZCROSS-001",
        "seeds": a.seeds,
        "grid": {"descend_z_m": list(ZS), "plan_at_open": list(PLAN_MODES),
                 "openings": [o for o in OPENINGS],
                 "cells": len(cells), "rollouts": len(good)},
        "controls": {
            "every_cell_ran_all_seeds": all(c["n"] == a.seeds for c in cells),
            "all_rollouts_ok": len(good) == len(grid),
            "shipped_cell_in_grid": shipped is not None,
            "shipped_cell_reproduces_fork_0_of_10":
                shipped is not None and shipped["fork_placed"] == 0,
            "shipped_cell_reproduces_spoon_0_of_10":
                shipped is not None and shipped["spoon_placed"] == 0,
        },
        "shipped_cell": shipped,
        "best_arrival_cell": best,
        "best_subgoal_cell": best_score,
        "cells_that_placed_any_cutlery": placed,
        "verdict": ("PLACED" if placed else "REFUTED -- no cell in the cross "
                    "places the fork or the spoon on any seed"),
        "cells": cells,
        "runs": runs,
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")

    print(f"\n{'z mm':>5} {'planAt':>7} {'open':>6} {'fork':>5} {'spoon':>6} "
          f"{'subg':>5} {'stall mm':>9} {'lift mm':>8}  top contact")
    for c in sorted(cells, key=lambda c: (c["descend_z_mm"], c["plan_at_open"],
                                          -1 if c["opening"] is None else c["opening"])):
        op = "NARROW" if c["opening"] is None else f"{c['opening']:.2f}"
        print(f"{c['descend_z_mm']:>5.1f} {str(c['plan_at_open']):>7} {op:>6} "
              f"{c['fork_placed']:>5} {c['spoon_placed']:>6} "
              f"{c['subgoals_met_total']:>5} {c['fork_stall_mm_median']:>9} "
              f"{c['fork_lift_mm_median']:>8}  {c['fork_top_contact']}")
    print(f"\nverdict: {doc['verdict']}")
    print(f"controls: {doc['controls']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
