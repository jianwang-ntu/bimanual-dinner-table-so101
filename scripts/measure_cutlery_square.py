#!/usr/bin/env python3
"""The axis nine probes never crossed: solve the cutlery pick for its jaw AXIS.

``TECHNICAL_SUMMARY`` section 8 records nine tested explanations of
``fork_placed`` / ``spoon_placed`` 0 of 10.  Read together, this repository's own
evidence says the stall is a jaw-ORIENTATION failure, and no probe has ever
crossed the knob that fixes orientation with the knobs that fix arrival:

  * ``evidence/hand_floor.json`` measures the hand hanging 26.5 mm below the
    point it pinches with at ``fork_descend`` -- and its OWN control reports
    ``drop_is_a_property_of_the_hand_not_the_waypoint: false``.  The drop is a
    property of the POSE.  At the fork it is 26.5 mm with the jaw axis
    [-0.46 0.45 0.77] -- 50 degrees out of horizontal, one jaw a fifth of a
    metre-tenth below the other -- and at the spoon 11.7 mm with the jaw axis
    14 degrees out.  The lower jaw is what reaches the drawer floor first.
  * ``evidence/grasp_feasibility.json`` reports a fork pose 6.0 mm above the
    asked height that clears the woodwork on 3 of 3 seeds AND grips
    (``grips_at_min_clear: true``, jaw 3.9 mm below the handle top).  A
    graspable pose exists.  The shipped controller does not arrive at it.
  * ``evidence/cutlery_zcross.json``'s best-arrival cell reaches 3.70 mm of the
    asked height and still lifts nothing, and its ``fork_top_contact`` is
    ``geom115 | fork_handle`` -- the low jaw resting ON the handle rather than
    beside it, which is what a tilted jaw axis does.

``plan_pose_squared`` is the solver that puts the jaw axis where it was asked;
its own docstring records taking the mug from 0 of 10 to 8 of 10.  The cutlery
has never used it: ``CUTLERY_DESCEND_SQUARE`` is False, both cutlery ``_pick``
calls take the default ``square=False``, and all 24 cells of the zcross grid ran
with it off.  ``measure_fork_descent`` swept ``square`` for the descend move
ALONE, at the shipped opening and the shipped ``plan_at`` -- the two knobs that
between them displace the descending hand 21.83 mm from the solved one.

So this probe crosses three-way: square mode x descent height x the arrival
pair.  36 cells, 10 seeds, 360 rollouts.

THE CONTROL is the shipped cell (square none, 0.003 m, plan_at False,
GRIPPER_NARROW), carried IN the grid rather than quoted from an earlier file, so
the comparison is inside one process pool and one code state and so the probe
can FAIL: if the shipped cell does not reproduce fork 0/10, spoon 0/10 and 15
subgoals, the run is not trustworthy and the controls say so.

SECOND CONTROL: the plate and the mug.  ``CUTLERY_SQUARE`` reaches only the two
cutlery picks.  If a cell moves ``plate_placed`` or ``mug_placed`` the knob is
not doing what this docstring says it does, and that is reported too.

Nothing here changes shipped behaviour: the constants are module globals read at
``dinner_table_script()`` call time and set per variant inside a worker process.

Writes evidence/cutlery_square.json.
Run:  python3 scripts/measure_cutlery_square.py --seeds 10 --workers 48
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

# (CUTLERY_SQUARE, CUTLERY_DESCEND_SQUARE) per mode.  "descend" is what
# measure_fork_descent swept; "all" squares the approach and the lift with it,
# because a squared descent followed by an unsquared lift re-rolls the wrist
# while the jaws are closed on the handle.
SQUARE_MODES = {"none": (False, False, False),
                "descend": (False, True, False),
                "all": (True, True, False),
                "all+handoff": (True, True, True),
                "handoff": (False, False, True)}

SHIPPED_SQUARE = "none"
SHIPPED_Z = 0.003
SHIPPED_PLAN_AT = False
SHIPPED_OPENING = None            # None == GRIPPER_NARROW, whatever it is

MODES = ("none", "descend", "all", "all+handoff", "handoff")
ZS = (0.003, 0.006, 0.010)
PLAN_MODES = (False, True)
OPENINGS = (None, 0.15)


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


def run(seed: int, mode: str, z: float, plan_at_open: bool, opening,
        steps: int = 1, aim: bool = False) -> dict:
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

    sq_all, sq_desc, sq_hand = SQUARE_MODES[mode]
    C.CUTLERY_SQUARE = bool(sq_all)
    C.CUTLERY_DESCEND_SQUARE = bool(sq_desc)
    C.CUTLERY_HANDOFF_SQUARE = bool(sq_hand)
    C.CUTLERY_PLACE_AIM_BODY = bool(aim)
    C.CUTLERY_DESCEND_Z = float(z)
    C.CUTLERY_DESCEND_STEPS = int(steps)
    C.CUTLERY_PLAN_AT_OPEN = bool(plan_at_open)
    C.CUTLERY_DESCEND_OPENING = (C.GRIPPER_NARROW if opening is None
                                 else float(opening))
    script = C.dinner_table_script()
    roll = C.Rollout(model, data)

    mine = {a: _arm_geoms(model, a) for a in ("left", "right")}
    live = {"label": "", "arm": "", "body": None}
    asked, lowest, jawz = {}, {}, {}
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
        q = orig(mv)
        if b is not None:
            # the jaw axis the solver actually achieved, at the pose it returned
            g = roll.grips[mv.arm]
            sc = mujoco.MjData(model)
            sc.qpos[:] = data.qpos
            sc.qpos[g.qadr] = np.asarray(q, float)
            mujoco.mj_kinematics(model, sc)
            ax = g.jaw_axis(model, sc)
            jawz[b] = round(float(abs(ax[2])), 4)
        return q

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
                   float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
               "jaw_axis_abs_z": jawz.get(b)}
        if b in asked and b in lowest:
            rec["asked_tip_z_m"] = round(float(asked[b][2]), 4)
            rec["lowest_tip_z_m"] = round(lowest[b], 4)
            rec["stalled_above_target_mm"] = round(
                (lowest[b] - float(asked[b][2])) * 1000, 1)
        rec["contacts"] = [{"pair": p, "steps": s} for p, s in
                           sorted(tally.get(b, {}).items(),
                                  key=lambda kv: -kv[1])[:3]]
        out[b] = rec

    return {"seed": seed, "square_mode": mode, "descend_z_m": float(z),
            "steps": int(steps), "aim_body": bool(aim),
            "plan_at_open": bool(plan_at_open),
            "opening": None if opening is None else float(opening),
            "opening_rad_effective": float(C.CUTLERY_DESCEND_OPENING),
            # every axis this probe can move must appear here, or a cell that
            # is NOT the shipped script gets flagged as one.  ``aim_body`` and
            # ``steps`` were missing and two cells carried the flag; caught by
            # reading the run rows back rather than the summary.
            "shipped": (mode == SHIPPED_SQUARE
                        and abs(float(z) - SHIPPED_Z) < 1e-12
                        and bool(plan_at_open) == SHIPPED_PLAN_AT
                        and opening is SHIPPED_OPENING
                        and int(steps) == 1
                        and not bool(aim)),
            "subgoals_met": int(sum(rep["subgoals"].values())),
            "subgoals": rep["subgoals"],
            "task_success": bool(rep["task_success"]),
            "bodies": out}


def _job(a):
    s, m, z, p, o, k, v = a
    try:
        return run(s, m, z, p, o, k, v)
    except Exception as exc:                                   # pragma: no cover
        return {"seed": s, "square_mode": m, "descend_z_m": z, "steps": k,
                "aim_body": v, "plan_at_open": p, "opening": o,
                "error": f"{type(exc).__name__}: {exc}"}


def _med(v):
    return round(float(np.median(v)), 2) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--zs", default=",".join(str(z) for z in ZS))
    ap.add_argument("--openings", default="none,0.15",
                    help="comma list; 'none' means GRIPPER_NARROW")
    ap.add_argument("--steps", default="1", help="comma list of descend_steps")
    ap.add_argument("--aim", default="0", help="comma list of 0/1: aim the body")
    ap.add_argument("--out", default=str(EVID / "cutlery_square.json"))
    a = ap.parse_args()

    modes = [m for m in a.modes.split(",") if m]
    zs = [float(z) for z in a.zs.split(",") if z]
    opens = [None if t.strip().lower() == "none" else float(t)
             for t in a.openings.split(",") if t.strip()]
    steps = [int(t) for t in a.steps.split(",") if t.strip()]
    aims = [bool(int(t)) for t in a.aim.split(",") if t.strip()]
    jobs = [(s, m, z, p, o, k, v)
            for m in modes for z in zs for k in steps for v in aims
            for p in PLAN_MODES for o in opens
            for s in range(a.seeds)]

    rows = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            if "error" in r:
                print(f"  ERROR {r['square_mode']} z={r['descend_z_m']} "
                      f"seed {r['seed']}: {r['error']}")
    good = [r for r in rows if "bodies" in r]

    cells = []
    for m in modes:
      for z in zs:
       for k in steps:
        for v in aims:
            for p in PLAN_MODES:
                for o in opens:
                    got = [r for r in good
                           if r["square_mode"] == m
                           and abs(r["descend_z_m"] - z) < 1e-12
                           and r.get("steps") == k
                           and bool(r.get("aim_body")) == v
                           and r["plan_at_open"] == p
                           and ((o is None and r["opening"] is None)
                                or (o is not None and r["opening"] == o))]
                    if not got:
                        continue
                    def col(body, key):
                        return [g["bodies"][body][key] for g in got
                                if g["bodies"][body].get(key) is not None]
                    top = {}
                    for g in got:
                        for c in g["bodies"]["fork"]["contacts"][:1]:
                            top[c["pair"]] = top.get(c["pair"], 0) + 1
                    cells.append({
                        "square_mode": m,
                        "descend_z_mm": round(z * 1000, 1),
                        "steps": k, "aim_body": v,
                        "plan_at_open": p,
                        "opening": o,
                        "opening_rad_effective": got[0]["opening_rad_effective"],
                        "shipped": bool(got[0]["shipped"]),
                        "n": len(got),
                        "fork_placed": sum(1 for g in got
                                           if g["bodies"]["fork"]["placed"]),
                        "spoon_placed": sum(1 for g in got
                                            if g["bodies"]["spoon"]["placed"]),
                        "plate_placed": sum(1 for g in got
                                            if g["subgoals"].get("plate_placed")),
                        "mug_placed": sum(1 for g in got
                                          if g["subgoals"].get("mug_placed")),
                        "task_success": sum(1 for g in got if g["task_success"]),
                        "subgoals_met_total": sum(g["subgoals_met"] for g in got),
                        "fork_jaw_axis_abs_z_median": _med(col("fork",
                                                               "jaw_axis_abs_z")),
                        "spoon_jaw_axis_abs_z_median": _med(col("spoon",
                                                                "jaw_axis_abs_z")),
                        "fork_stall_mm_median": _med(
                            col("fork", "stalled_above_target_mm")),
                        "spoon_stall_mm_median": _med(
                            col("spoon", "stalled_above_target_mm")),
                        "fork_lift_mm_median": _med(col("fork", "lifted_mm")),
                        "spoon_lift_mm_median": _med(col("spoon", "lifted_mm")),
                        "fork_top_contact": (max(top.items(),
                                                 key=lambda kv: kv[1])[0]
                                             if top else None),
                    })

    ship = next((c for c in cells if c["shipped"]), None)
    placed = [c for c in cells if c["fork_placed"] or c["spoon_placed"]]
    best = max(cells, key=lambda c: (c["fork_placed"] + c["spoon_placed"],
                                     c["subgoals_met_total"])) if cells else None
    ctl = {
        "every_cell_ran_all_seeds": all(c["n"] == a.seeds for c in cells),
        "all_rollouts_ok": len(good) == len(rows),
        "shipped_cell_in_grid": ship is not None,
        "shipped_cell_reproduces_fork_0_of_10":
            bool(ship and ship["fork_placed"] == 0),
        "shipped_cell_reproduces_spoon_0_of_10":
            bool(ship and ship["spoon_placed"] == 0),
        "shipped_cell_reproduces_15_subgoals":
            bool(ship and ship["subgoals_met_total"] == 15),
        "square_moved_the_jaw_axis": None,
        "knob_left_plate_and_mug_alone": None,
    }
    if ship is not None:
        sq = [c for c in cells if c["square_mode"] == "all"
              and c["steps"] == ship["steps"]
              and c["aim_body"] == ship["aim_body"]
              and c["descend_z_mm"] == ship["descend_z_mm"]
              and c["plan_at_open"] == ship["plan_at_open"]
              and c["opening"] == ship["opening"]]
        if sq and sq[0]["fork_jaw_axis_abs_z_median"] is not None:
            ctl["square_moved_the_jaw_axis"] = bool(
                sq[0]["fork_jaw_axis_abs_z_median"]
                < ship["fork_jaw_axis_abs_z_median"] - 0.05)
        ctl["knob_left_plate_and_mug_alone"] = bool(
            sq and sq[0]["plate_placed"] == ship["plate_placed"]
            and sq[0]["mug_placed"] == ship["mug_placed"])

    doc = {
        "probe": "measure_cutlery_square.py",
        "finding_id": "H-CUTLERY-SQUARE-001",
        "question": ("Does solving the cutlery pick for its jaw CLOSING AXIS -- "
                     "the one axis the nine prior probes never crossed -- place "
                     "the fork or the spoon when crossed with the descent height "
                     "and the arrival pair?"),
        "seeds": a.seeds,
        "grid": {"square_modes": modes, "descend_z_m": zs,
                 "descend_steps": steps, "aim_body": aims,
                 "plan_at_open": list(PLAN_MODES),
                 "openings": opens,
                 "cells": len(cells), "rollouts": len(rows)},
        "controls": ctl,
        "shipped_cell": ship,
        "best_cell": best,
        "cells_that_placed_any_cutlery": placed,
        "verdict": ("CONFIRMED -- at least one cell places cutlery the shipped "
                    "controller does not" if placed else
                    "REFUTED -- no cell in the cross places the fork or the "
                    "spoon on any seed"),
        "cells": cells,
        "runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in
                      ("controls", "verdict")}, indent=1))
    for c in cells:
        print(f"  {c['square_mode']:8s} dz={c['descend_z_mm']:5.1f} "
              f"k={c['steps']} aim={int(c['aim_body'])} "
              f"pao={str(c['plan_at_open']):5s} open={str(c['opening']):5s} "
              f"jawz={c['fork_jaw_axis_abs_z_median']} "
              f"sub={c['subgoals_met_total']:3d} fork={c['fork_placed']} "
              f"spoon={c['spoon_placed']} plate={c['plate_placed']} "
              f"mug={c['mug_placed']} stall={c['fork_stall_mm_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
