#!/usr/bin/env python3
"""Is the cutlery ungraspable, or ungraspable WHERE THE SCENE SEATS IT?

``F-CUTLERY-LIP-001`` has six refuted routes.  Every one of them moved the ARM
-- the standoff, the jaw squaring, the descent interpolation, the approach
vector, the reach ceiling, the spoon's grasp depth -- and every one of them
left the fork at the single seat ``envs/dinner_table.py`` authors it at:
``("spoon", -0.032), ("fork", 0.018)`` about the cabinet, i.e. world
(0.000, 0.128) and (0.000, 0.178) on every seed that has ever been run.
``TECHNICAL_SUMMARY.md`` section 8 ends by naming what is left: "The remaining
candidates are changes to the SCENE or to the GRIPPER rather than to the
trajectory, and neither has been made."  This is the scene one.

Two mechanisms already measured in this tree both point at the seat, and they
point the SAME way, which is why this is worth a sweep rather than a guess:

  reach     the servos saturate near 0.30 m of horizontal reach -- three
            joints at the +/-2.94 N.m limit with no contact anywhere on the
            arm (TECHNICAL_SUMMARY section 5).  The fork sits 0.387 m from the
            right arm's base and the spoon 0.347 m from the left's.  Both arm
            bases are at y = -0.14, so moving the seat toward the drawer's
            open front (-y) shortens BOTH reaches.
  clearance the fork is the object at the BACK of the drawer, 26-37 mm from
            drawer_back's inner face depending on where the randomized
            drawer_slide has carried the walls, and H-CUTLERY-DESCENT-
            REFUTED-001 measured drawer_back as the geom the arm then wears.
            -y moves the fork out of that pocket and into the open front.

Swept:

  seat_dy    a rigid y offset on BOTH bodies.  Negative is toward the open
             front.  +0.015 is the control direction: it seats the fork
             DEEPER into the wall the contact tally names and further from
             its arm, and must not help.
  splay_dx   each body displaced along x toward the arm that picks it (fork
             right, spoon left).  A second, independent way to shorten the
             reach, orthogonal to seat_dy.

``CUTLERY_PLACE_SPREAD`` is pinned to 0.0 for the whole sweep, so the seat is
the only thing that varies and a cell's result is not a draw of the jitter
this tick also added.

Controls, all four able to fail:

  baseline_reproduces  the (0, 0) cell must reproduce the published per-goal
                       tally -- drawer 10/10, plate 4/10, mug 1/10, cutlery
                       0/10, 15/50.  It is the shipped scene; if it does not
                       reproduce, this file is measuring something else.
  seat_is_applied      every non-zero cell's recorded start position must
                       differ from the baseline's by the offset asked for.  A
                       constant that silently does nothing would otherwise
                       read as "the seat does not matter".
  feasible_seats       a cell whose cutlery starts INSIDE a drawer wall is not
                       a scene, and cannot be a win.  Penetration is read from
                       MuJoCo's own contact pass and reported per cell.
  control_direction    +y (deeper into the back wall, further from the arm)
                       must not out-score the baseline on the cutlery.

Writes evidence/cutlery_seat.json.
Run:  python3 scripts/measure_cutlery_seat.py --seeds 10
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
ARM_FOR = {"fork": "right", "spoon": "left"}
ARM_BASE = {"right": np.array([0.22, -0.14]), "left": np.array([-0.22, -0.14])}
# The shipped scene: the seat dinner_table.build authors, no splay, no jitter.
SHIPPED = (0.0, 0.0)
PUBLISHED = {"drawer_open": 10, "fork_placed": 0, "spoon_placed": 0,
             "plate_placed": 4, "mug_placed": 1}


def run(seed: int, seat_dy: float, splay: float, square: bool = False) -> dict:
    from envs import randomize as R
    from envs.task import TaskMonitor, PLACEMENTS
    from envs import controller as C
    from envs import scene_source

    R.CUTLERY_PLACE_SPREAD = 0.0            # the seat is the only variable
    R.CUTLERY_SEAT_XY = (0.0, float(seat_dy))
    R.CUTLERY_SPLAY_X = float(splay)
    C.CUTLERY_DESCEND_SQUARE = bool(square)

    model, data, log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    start = {o: data.xpos[bid(o)].copy() for o in BODIES}
    peak = {o: 0.0 for o in BODIES}

    roll = C.Rollout(model, data)
    live = {"body": None, "arm": None}
    asked_z: dict[str, float] = {}
    reached_z: dict[str, float] = {}
    contacts: dict[str, dict[str, int]] = {}
    orig_plan = roll._plan

    def plan(mv):
        lb = getattr(mv, "label", "") or ""
        live["arm"] = mv.arm
        live["body"] = next((b for b in BODIES if lb.startswith(f"{b}_descend")),
                            None)
        if live["body"] and lb == f"{live['body']}_descend" and mv.where:
            asked_z[live["body"]] = float(
                np.asarray(mv.where(model, data), float)[2])
        return orig_plan(mv)

    roll._plan = plan

    def on_step(d):
        for o in BODIES:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))
        b = live["body"]
        if b is None:
            return
        tip = roll.grips[live["arm"]].tip_mid(model, d)
        reached_z[b] = min(reached_z.get(b, 9.9), float(tip[2]))
        mine = {g for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[g])) or ""
                    ).startswith(f"{live['arm']}_")}
        tally = contacts.setdefault(b, {})
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if (g1 in mine) != (g2 in mine):
                other = g2 if g1 in mine else g1
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       other) or f"geom{other}"
                tally[nm] = tally.get(nm, 0) + 1

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    objects = {}
    for b in BODIES:
        end = data.xpos[bid(b)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{b}_placed"][1])]
        rec = {
            "start_xy": [round(float(v), 4) for v in start[b][:2]],
            "reach_mm": round(float(np.linalg.norm(
                start[b][:2] - ARM_BASE[ARM_FOR[b]])) * 1000, 1),
            "penetration_mm": log["state"]["cutlery"][b]["penetration_mm"],
            "lifted_mm": round(peak[b] * 1000, 1),
            "planar_move_mm": round(
                float(np.linalg.norm(end[:2] - start[b][:2])) * 1000, 1),
            "final_to_target_mm": round(
                float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
            "placed": bool(rep["subgoals"][f"{b}_placed"]),
        }
        if b in asked_z and b in reached_z:
            rec["stalled_above_target_mm"] = round(
                (reached_z[b] - asked_z[b]) * 1000, 1)
        top = sorted(contacts.get(b, {}).items(), key=lambda kv: -kv[1])[:3]
        rec["blocking_contacts"] = [{"geom": n, "steps": s} for n, s in top]
        objects[b] = rec

    return {"seed": seed, "seat_dy": float(seat_dy), "splay": float(splay),
            "square": bool(square),
            "shipped": (abs(seat_dy - SHIPPED[0]) < 1e-12
                        and abs(splay - SHIPPED[1]) < 1e-12
                        and not square),
            "subgoals": rep["subgoals"],
            "subgoals_met": int(sum(rep["subgoals"].values())),
            "task_success": bool(rep["task_success"]),
            "objects": objects}


def _job(a):
    try:
        return run(*a)
    except Exception as exc:
        return {"seed": a[0], "seat_dy": a[1], "splay": a[2], "square": a[3],
                "error": f"{type(exc).__name__}: {exc}"}


def _med(rows, body, key):
    v = [r["objects"][body].get(key) for r in rows if "objects" in r]
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 1) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seat-dys", default="0.015,0.0,-0.010,-0.020,-0.030,-0.038")
    ap.add_argument("--splays", default="0.0,0.015,0.030")
    ap.add_argument("--squares", default="0,1")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=str(EVID / "cutlery_seat.json"))
    a = ap.parse_args()

    dys = [float(x) for x in a.seat_dys.split(",")]
    sps = [float(x) for x in a.splays.split(",")]
    sqs = [bool(int(x)) for x in a.squares.split(",")]
    if False not in sqs:
        raise SystemExit("the shipped descent (unsquared) must be in the "
                         "sweep -- CUTLERY_DESCEND_SQUARE defaults to False")
    if not (any(abs(d) < 1e-12 for d in dys) and any(abs(s) < 1e-12 for s in sps)):
        raise SystemExit("the shipped seat (dy=0, splay=0) must be in the "
                         "sweep -- it is this probe's only baseline and the "
                         "configuration every published figure was measured at")
    if not any(d > 0 for d in dys):
        raise SystemExit("a +y seat must be in the sweep -- it is the control "
                         "direction that must not help")

    jobs = [(s, d, p, q) for d in dys for p in sps for q in sqs
            for s in range(a.seeds)]
    print(f"  {len(jobs)} rollouts over "
          f"{len(dys)}x{len(sps)}x{len(sqs)} cells")
    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            if "error" in r:
                print(f"  ERROR seat_dy={r['seat_dy']} splay={r['splay']} "
                      f"sq={r['square']} seed={r['seed']}: {r['error']}")

    cells = []
    for d in dys:
      for p in sps:
        for q in sqs:
            got = [r for r in rows
                   if abs(r["seat_dy"] - d) < 1e-12 and abs(r["splay"] - p) < 1e-12
                   and r["square"] is q and "objects" in r]
            if not got:
                continue
            per_goal = {g: sum(1 for r in got if r["subgoals"][g]) for g in PUBLISHED}
            cells.append({
                "seat_dy": d, "splay": p, "square": q,
                "shipped": abs(d) < 1e-12 and abs(p) < 1e-12 and not q,
                "n": len(got),
                "subgoals_met_total": sum(r["subgoals_met"] for r in got),
                "per_goal": per_goal,
                "task_success": sum(1 for r in got if r["task_success"]),
                "fork_start_xy": got[0]["objects"]["fork"]["start_xy"],
                "spoon_start_xy": got[0]["objects"]["spoon"]["start_xy"],
                "fork_reach_mm": got[0]["objects"]["fork"]["reach_mm"],
                "spoon_reach_mm": got[0]["objects"]["spoon"]["reach_mm"],
                "worst_penetration_mm": min(
                    r["objects"][b]["penetration_mm"] for r in got for b in BODIES),
                "fork_lifted_mm_median": _med(got, "fork", "lifted_mm"),
                "spoon_lifted_mm_median": _med(got, "spoon", "lifted_mm"),
                "fork_stall_mm_median": _med(got, "fork", "stalled_above_target_mm"),
                "spoon_stall_mm_median": _med(got, "spoon", "stalled_above_target_mm"),
                "fork_blocking_contacts": sorted(
                    ({k: sum(c["steps"] for r in got
                             for c in r["objects"]["fork"]["blocking_contacts"]
                             if c["geom"] == k)
                      for k in {c["geom"] for r in got
                                for c in r["objects"]["fork"]["blocking_contacts"]}}
                     ).items(), key=lambda kv: -kv[1])[:3],
            })

    base = next(c for c in cells if c["shipped"])
    feasible = [c for c in cells if c["worst_penetration_mm"] >= -2.0]
    infeasible = [c for c in cells if c["worst_penetration_mm"] < -2.0]
    plus_y = [c for c in cells if c["seat_dy"] > 0]

    seat_applied = all(
        (abs(c["fork_start_xy"][1] - (base["fork_start_xy"][1] + c["seat_dy"])) < 1e-3
         and abs(c["fork_start_xy"][0] - (base["fork_start_xy"][0] + c["splay"])) < 1e-3
         and abs(c["spoon_start_xy"][0] - (base["spoon_start_xy"][0] - c["splay"])) < 1e-3)
        for c in cells)

    best = max(feasible, key=lambda c: (c["per_goal"]["fork_placed"]
                                        + c["per_goal"]["spoon_placed"],
                                        c["subgoals_met_total"]))
    controls = {
        "baseline_reproduces": base["per_goal"] == PUBLISHED,
        "baseline_per_goal": base["per_goal"],
        "published_per_goal": PUBLISHED,
        "seat_is_applied": seat_applied,
        "feasible_cells": len(feasible),
        "infeasible_cells": [{"seat_dy": c["seat_dy"], "splay": c["splay"],
                              "worst_penetration_mm": c["worst_penetration_mm"]}
                             for c in infeasible],
        "control_direction_did_not_help": all(
            c["per_goal"]["fork_placed"] + c["per_goal"]["spoon_placed"]
            <= base["per_goal"]["fork_placed"] + base["per_goal"]["spoon_placed"]
            for c in plus_y),
        "all_runs_ok": all("objects" in r for r in rows),
    }

    out = {
        "probe": "measure_cutlery_seat.py",
        "question": ("F-CUTLERY-LIP-001 route (c): is the cutlery ungraspable, "
                     "or ungraspable at the ONE seat the scene has always put "
                     "it at? Six refuted routes all moved the arm."),
        "hypothesis_id": "H-CUTLERY-SEAT-001",
        "seeds": a.seeds,
        "jitter": "CUTLERY_PLACE_SPREAD pinned to 0.0 for every cell",
        "controls": controls,
        "best_feasible_cell": {k: best[k] for k in
                               ("seat_dy", "splay", "per_goal",
                                "subgoals_met_total", "fork_reach_mm",
                                "spoon_reach_mm", "fork_lifted_mm_median",
                                "fork_stall_mm_median")},
        "cells": cells,
        "runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n sq seat_dy splay  subg  drawer fork spoon plate mug  "
          "fork_reach fork_stall fork_lift  pen_mm")
    for c in sorted(cells, key=lambda c: (c["square"], c["seat_dy"], c["splay"])):
        g = c["per_goal"]
        mark = " *" if c["shipped"] else ("  " if c["worst_penetration_mm"] >= -2 else " X")
        print(f" {int(c['square'])}{mark}{c['seat_dy']:+.3f} {c['splay']:+.3f} "
              f"{c['subgoals_met_total']:>3}/{5*c['n']:<3} "
              f"{g['drawer_open']:>4} {g['fork_placed']:>5} {g['spoon_placed']:>5} "
              f"{g['plate_placed']:>5} {g['mug_placed']:>4} "
              f"{c['fork_reach_mm']:>9} {str(c['fork_stall_mm_median']):>10} "
              f"{str(c['fork_lifted_mm_median']):>9}  {c['worst_penetration_mm']:>6}")
    print("\n  controls:")
    for k in ("baseline_reproduces", "seat_is_applied",
              "control_direction_did_not_help", "all_runs_ok"):
        print(f"    {k}: {controls[k]}")
    print(f"    infeasible cells: {len(infeasible)}")
    print(f"\n  wrote {a.out}")
    bad = [k for k in ("baseline_reproduces", "seat_is_applied", "all_runs_ok")
           if controls[k] is False]
    if bad:
        print(f"  CONTROL FAILED: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
