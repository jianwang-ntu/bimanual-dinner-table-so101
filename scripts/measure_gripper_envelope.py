#!/usr/bin/env python3
"""The eighth candidate: the jaws that descend are not the jaws that were solved.

``TECHNICAL_SUMMARY.md`` section 8 named two candidates never tried.  The scene
is closed (``H-CUTLERY-SEAT-001``, 360 rollouts).  This is the other one, the
GRIPPER, and it is not about the gripper's shape -- it is about a mismatch
between two numbers in ``_pick``:

    Move(..., opening=open_to,        # GRIPPER_NARROW, ~51 mm: what DESCENDS
             plan_at=close_to)        # pinch(fork handle), ~7 mm: what is SOLVED

``Rollout._plan`` hands ``plan_at`` to the solver as the gripper opening and
then overwrites the gripper command with ``opening``.  ``plan_pose`` places the
jaw MEETING POINT on the target, and that point slides along the hand as the
jaws open -- ``Move``'s own docstring says "the jaws meet ~41 mm nearer the
wrist closed than open".  So the pose is placed for a hand that is not the hand
that goes down.

Two things are measured, and they are independent of each other:

A. THE OFFSET, kinematically.  Take the joint vector ``plan_pose`` actually
   returned for ``fork_descend``, put it in a scratch ``MjData`` twice -- once
   with the gripper at the PLANNING width, once at the EXECUTED width -- and
   read ``tip_mid`` at each.  No dynamics, no contact, no controller: this is
   the displacement the mismatch introduces before the arm has moved at all.
   ``fork_above`` and ``spoon_above`` are the built-in NEGATIVE CONTROL: those
   moves carry no ``plan_at``, so planning and execution widths are equal there
   and the offset must come out at zero.  If it does not, this probe is
   measuring its own arithmetic.

B. THE OUTCOME, over a sweep.  ``CUTLERY_DESCEND_OPENING`` x
   ``CUTLERY_PLAN_AT_OPEN``, ten seeds each, scored by the same predicates as
   every other run.  The contact tally names BOTH geoms -- which part of the
   hand, against which part of the world -- and the first contact of the
   descend hold is kept separately, because "which lands first" is a question
   about time and a running total cannot answer it.

Nothing here changes a shipped default.  The shipped cell is in the grid and
must reproduce the published fork/spoon figures, or the sweep is not measuring
the shipped controller.  Writes evidence/gripper_envelope.json.

Run:  python3 scripts/measure_gripper_envelope.py --seeds 10
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
ARM_OF = {"fork": "right", "spoon": "left"}
DESCEND = {b: f"{b}_descend" for b in BODIES}
ABOVE = {b: f"{b}_above" for b in BODIES}

# The grid.  ``None`` for the opening means "the shipped constant", so the
# shipped cell is named by the same symbol the module ships rather than by a
# literal copied into this file that could drift away from it.
SHIPPED_PLAN_AT_OPEN = False
OPENINGS = (None, 0.30, 0.15, 0.00)
PLAN_MODES = (False, True)


# ------------------------------------------------------------------ helpers
def _geom_name(model, g: int) -> str:
    """``body/geom``.  Several jaw geoms in this model are unnamed, and a
    contact reported as ``geom115`` does not answer "which part of the hand
    landed first"; the body is always named, so it is carried too."""
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             int(model.geom_bodyid[int(g)])) or "?"
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
    return f"{body}/{name}" if name else f"{body}/geom{g}"


def _arm_geoms(model, arm: str) -> set[int]:
    return {g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[g])) or ""
                ).startswith(f"{arm}_")}


# ------------------------------------------------- A. the kinematic offset
def offsets(seed: int) -> dict:
    """Where the executed jaws sit relative to the solved ones, per waypoint.

    Runs the shipped script.  For every waypoint of interest it captures the
    solver's own joint answer and the two gripper widths involved, then reads
    ``tip_mid`` at each width from a scratch ``MjData`` -- so the number is the
    hand's geometry alone, with the arm frozen at the pose the solver chose.
    """
    from envs.randomize import make_env
    from envs.task import TaskMonitor
    from envs import controller as C
    from envs import scene_source

    model, data, _ = make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))
    roll = C.Rollout(model, data)
    scratch = mujoco.MjData(model)
    orig = roll._plan
    want = set(DESCEND.values()) | set(ABOVE.values())
    rows: dict[str, dict] = {}

    def plan(mv):
        q = orig(mv)
        lb = getattr(mv, "label", "") or ""
        if lb not in want or mv.where is None:
            return q
        g = roll.grips[mv.arm]
        # The two widths this move involves, resolved exactly as _plan does.
        opening = mv.opening(model, data, g) if callable(mv.opening) else mv.opening
        plan_at = mv.plan_at(model, data, g) if callable(mv.plan_at) else mv.plan_at
        solved_at = plan_at if plan_at is not None else opening
        executed_at = opening if opening is not None else solved_at
        target = np.asarray(mv.where(model, data), float)

        def tip_at(width):
            scratch.qpos[:] = data.qpos
            scratch.qvel[:] = 0.0
            scratch.qpos[g.qadr[:-1]] = q[:-1]      # the arm as solved
            scratch.qpos[g.grip_q] = float(width)
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            return (g.tip_mid(model, scratch).copy(),
                    g.jaw_axis(model, scratch).copy(),
                    g.approach_axis(scratch).copy())

        t_solved, jaw_s, app_s = tip_at(solved_at)
        t_exec, _, _ = tip_at(executed_at)
        d = t_exec - t_solved
        rows[lb] = {
            "waypoint": lb,
            "arm": mv.arm,
            "has_plan_at": plan_at is not None,
            "solved_at_rad": round(float(solved_at), 4),
            "executed_at_rad": round(float(executed_at), 4),
            "solved_sep_mm": round(g.sep_for_q(float(solved_at)) * 1000, 2),
            "executed_sep_mm": round(g.sep_for_q(float(executed_at)) * 1000, 2),
            "offset_mm": round(float(np.linalg.norm(d)) * 1000, 2),
            "offset_xyz_mm": [round(float(v) * 1000, 2) for v in d],
            "along_approach_mm": round(float(d @ app_s) * 1000, 2),
            "along_jaw_mm": round(float(d @ jaw_s) * 1000, 2),
            "solved_tip_to_target_mm": round(
                float(np.linalg.norm(t_solved - target)) * 1000, 2),
            "executed_tip_to_target_mm": round(
                float(np.linalg.norm(t_exec - target)) * 1000, 2),
        }
        return q

    roll._plan = plan
    roll.run(C.dinner_table_script(), monitor=mon)
    return {"seed": seed, "waypoints": rows}


# ------------------------------------------------------- B. the outcome sweep
def run(seed: int, opening, plan_at_open: bool) -> dict:
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

    shipped_opening = C.GRIPPER_NARROW
    C.CUTLERY_DESCEND_OPENING = shipped_opening if opening is None else float(opening)
    C.CUTLERY_PLAN_AT_OPEN = bool(plan_at_open)
    script = C.dinner_table_script()
    roll = C.Rollout(model, data)

    mine = {a: _arm_geoms(model, a) for a in ("left", "right")}
    live = {"label": "", "arm": "", "body": None}
    asked = {}
    lowest = {}
    peak = {o: 0.0 for o in BODIES}
    tally: dict[str, dict[str, int]] = {}
    first: dict[str, dict] = {}
    steps_in_hold: dict[str, int] = {}
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
        steps_in_hold[b] = steps_in_hold.get(b, 0) + 1
        t = tally.setdefault(b, {})
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if (g1 in mine[arm]) == (g2 in mine[arm]):
                continue
            hand, other = (g1, g2) if g1 in mine[arm] else (g2, g1)
            pair = f"{_geom_name(model, hand)} | {_geom_name(model, other)}"
            t[pair] = t.get(pair, 0) + 1
            if b not in first:
                first[b] = {"pair": pair, "t": round(float(d.time), 3),
                            "step_into_hold": steps_in_hold[b],
                            "tip_z_m": round(float(g.tip_mid(model, d)[2]), 4)}

    roll.run(script, monitor=mon, on_step=on_step)
    rep = mon.report(data)

    out = {}
    for b in BODIES:
        end = data.xpos[bid(b)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{b}_placed"][1])]
        rec = {
            "placed": bool(rep["subgoals"][f"{b}_placed"]),
            "lifted_mm": round(peak[b] * 1000, 1),
            "planar_move_mm": round(
                float(np.linalg.norm(end[:2] - start[b][:2])) * 1000, 1),
            "final_to_target_mm": round(
                float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
        }
        if b in asked and b in lowest:
            rec["asked_tip_z_m"] = round(float(asked[b][2]), 4)
            rec["lowest_tip_z_m"] = round(lowest[b], 4)
            rec["stalled_above_target_mm"] = round(
                (lowest[b] - float(asked[b][2])) * 1000, 1)
        rec["contacts"] = [{"pair": p, "steps": s} for p, s in
                           sorted(tally.get(b, {}).items(), key=lambda kv: -kv[1])[:4]]
        rec["first_contact"] = first.get(b)
        out[b] = rec

    return {"seed": seed,
            "opening": None if opening is None else float(opening),
            "opening_rad_effective": float(C.CUTLERY_DESCEND_OPENING),
            "plan_at_open": bool(plan_at_open),
            "shipped": opening is None and bool(plan_at_open) == SHIPPED_PLAN_AT_OPEN,
            "subgoals_met": int(sum(rep["subgoals"].values())),
            "subgoals": rep["subgoals"],
            "bodies": out}


def _job_off(s):
    try:
        return offsets(s)
    except Exception as exc:                                   # pragma: no cover
        return {"seed": s, "error": f"{type(exc).__name__}: {exc}"}


def _job_run(a):
    s, o, p = a
    try:
        return run(s, o, p)
    except Exception as exc:                                   # pragma: no cover
        return {"seed": s, "opening": o, "plan_at_open": p,
                "error": f"{type(exc).__name__}: {exc}"}


def _med(vals):
    return round(float(np.median(vals)), 2) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(EVID / "gripper_envelope.json"))
    a = ap.parse_args()

    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        off_rows = list(ex.map(_job_off, range(a.seeds)))
    off_good = [r for r in off_rows if "waypoints" in r]

    per_wp = {}
    for wp in list(DESCEND.values()) + list(ABOVE.values()):
        got = [r["waypoints"][wp] for r in off_good if wp in r["waypoints"]]
        if not got:
            continue
        per_wp[wp] = {
            "n": len(got),
            "has_plan_at": got[0]["has_plan_at"],
            "solved_sep_mm_median": _med([g["solved_sep_mm"] for g in got]),
            "executed_sep_mm_median": _med([g["executed_sep_mm"] for g in got]),
            "offset_mm_median": _med([g["offset_mm"] for g in got]),
            "offset_mm_max": round(max(g["offset_mm"] for g in got), 2),
            "along_approach_mm_median": _med([g["along_approach_mm"] for g in got]),
            "along_jaw_mm_median": _med([g["along_jaw_mm"] for g in got]),
            "solved_tip_to_target_mm_median": _med(
                [g["solved_tip_to_target_mm"] for g in got]),
            "executed_tip_to_target_mm_median": _med(
                [g["executed_tip_to_target_mm"] for g in got]),
        }

    grid = [(s, o, p) for o in OPENINGS for p in PLAN_MODES
            for s in range(a.seeds)]
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job_run, grid))
    for r in rows:
        if "error" in r:
            print(f"  ERROR seed={r['seed']} opening={r['opening']} "
                  f"plan_at_open={r['plan_at_open']}: {r['error']}")
    good = [r for r in rows if "bodies" in r]

    cells = []
    for o in OPENINGS:
        for p in PLAN_MODES:
            sel = [r for r in good
                   if r["opening"] == (None if o is None else float(o))
                   and r["plan_at_open"] == bool(p)]
            if not sel:
                continue
            pairs: dict[str, int] = {}
            firsts: dict[str, int] = {}
            for r in sel:
                for b in BODIES:
                    for c in r["bodies"][b]["contacts"]:
                        pairs[c["pair"]] = pairs.get(c["pair"], 0) + c["steps"]
                    fc = r["bodies"][b].get("first_contact")
                    if fc:
                        firsts[fc["pair"]] = firsts.get(fc["pair"], 0) + 1
            cells.append({
                "opening": None if o is None else float(o),
                "opening_rad_effective": sel[0]["opening_rad_effective"],
                "executed_sep_mm": None,
                "plan_at_open": bool(p),
                "shipped": all(r["shipped"] for r in sel),
                "n": len(sel),
                "fork_placed": sum(r["bodies"]["fork"]["placed"] for r in sel),
                "spoon_placed": sum(r["bodies"]["spoon"]["placed"] for r in sel),
                "subgoals_met_total": sum(r["subgoals_met"] for r in sel),
                "fork_lifted_mm_median": _med(
                    [r["bodies"]["fork"]["lifted_mm"] for r in sel]),
                "spoon_lifted_mm_median": _med(
                    [r["bodies"]["spoon"]["lifted_mm"] for r in sel]),
                "fork_stall_mm_median": _med(
                    [r["bodies"]["fork"]["stalled_above_target_mm"] for r in sel
                     if "stalled_above_target_mm" in r["bodies"]["fork"]]),
                "spoon_stall_mm_median": _med(
                    [r["bodies"]["spoon"]["stalled_above_target_mm"] for r in sel
                     if "stalled_above_target_mm" in r["bodies"]["spoon"]]),
                "top_contact_pairs": [
                    {"pair": k, "steps": v} for k, v in
                    sorted(pairs.items(), key=lambda kv: -kv[1])[:4]],
                "first_contact_pairs": [
                    {"pair": k, "episodes": v} for k, v in
                    sorted(firsts.items(), key=lambda kv: -kv[1])[:4]],
            })

    shipped = next((c for c in cells if c["shipped"]), None)
    # The verdict is DERIVED from the cell table, never written beside it: the
    # hypothesis is that the width mismatch is what stops the cutlery, so it
    # stands only if closing the mismatch places cutlery that the shipped
    # controller does not place.  A sub-goal total moving on the plate or the
    # mug is not this hypothesis being right.
    placed_any = any(c["fork_placed"] + c["spoon_placed"] > 0 for c in cells)
    verdict = "SUPPORTED" if placed_any else "REFUTED"
    best = max(cells, key=lambda c: (c["fork_placed"] + c["spoon_placed"],
                                     c["subgoals_met_total"])) if cells else None
    desc = [per_wp[w] for w in DESCEND.values() if w in per_wp]
    above = [per_wp[w] for w in ABOVE.values() if w in per_wp]

    controls = {
        "all_offset_seeds_ok": len(off_good) == a.seeds,
        "all_sweep_cells_ran": len(good) == len(grid),
        # A: the descend waypoints carry a plan_at and the approach ones do not.
        # If this ever stops being true the two populations below are not the
        # comparison this probe claims to be making.
        "descend_waypoints_carry_a_plan_at": all(w["has_plan_at"] for w in desc),
        "approach_waypoints_do_not": all(not w["has_plan_at"] for w in above),
        # A: the offset exists where the widths differ ...
        "solved_and_executed_widths_differ_at_the_descent": all(
            abs(w["solved_sep_mm_median"] - w["executed_sep_mm_median"]) > 1.0
            for w in desc),
        "offset_is_larger_than_the_solver_residual": all(
            w["offset_mm_median"] > 5.0 for w in desc),
        # ... and NOT where they are equal.  This is the negative control: the
        # same arithmetic, the same seeds, the same hand, one number changed.
        "no_offset_where_the_widths_are_equal": all(
            w["offset_mm_max"] < 0.01 for w in above),
        # B: the grid must contain the controller that is actually shipped, and
        # it must reproduce the published cutlery result (0/10 and 0/10).
        "shipped_cell_present": shipped is not None,
        "shipped_cell_reproduces_zero_placements": bool(
            shipped and shipped["fork_placed"] == 0 and shipped["spoon_placed"] == 0),
        "every_cell_ran_all_seeds": all(c["n"] == a.seeds for c in cells),
    }

    out = {
        "probe": "measure_gripper_envelope.py",
        "question": ("Is the cutlery stall the open-jaw envelope -- the descent "
                     "being executed at a gripper width the pose was not solved "
                     "at?"),
        "hypothesis_id": "H-GRIPPER-ENVELOPE-001",
        "seeds": a.seeds,
        "grid": {"openings": list(OPENINGS), "plan_at_open": list(PLAN_MODES),
                 "cells": len(cells), "rollouts": len(good)},
        "verdict": verdict,
        "verdict_basis": ("SUPPORTED iff some cell places a fork or a spoon "
                          "that the shipped cell does not; sub-goal totals on "
                          "the plate, mug or drawer are not this hypothesis."),
        "controls": controls,
        "offset_by_waypoint": per_wp,
        "cells": cells,
        "shipped_cell": shipped,
        "best_cell": best,
        "offset_runs": off_rows,
        "sweep_runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n  A. offset between the solved hand and the executed hand")
    print("  waypoint         n  plan_at  solved_sep  exec_sep   offset  "
          "along_approach  solved->tgt  exec->tgt")
    for wp in list(DESCEND.values()) + list(ABOVE.values()):
        w = per_wp.get(wp)
        if not w:
            continue
        print(f"  {wp:15} {w['n']:>2}  {str(w['has_plan_at']):>7}  "
              f"{w['solved_sep_mm_median']:>10}  {w['executed_sep_mm_median']:>8}  "
              f"{w['offset_mm_median']:>7}  {w['along_approach_mm_median']:>14}  "
              f"{w['solved_tip_to_target_mm_median']:>11}  "
              f"{w['executed_tip_to_target_mm_median']:>9}")

    print("\n  B. outcome sweep")
    print("  opening  plan_at_open  shipped   fork  spoon  subgoals  "
          "fork_stall  spoon_stall  first contact")
    for c in cells:
        op = "shipped" if c["opening"] is None else f"{c['opening']:.2f}"
        fc = c["first_contact_pairs"][0]["pair"] if c["first_contact_pairs"] else "-"
        print(f"  {op:>7}  {str(c['plan_at_open']):>12}  {str(c['shipped']):>7}  "
              f"{c['fork_placed']:>2}/{c['n']}  {c['spoon_placed']:>2}/{c['n']}  "
              f"{c['subgoals_met_total']:>8}  {str(c['fork_stall_mm_median']):>10}  "
              f"{str(c['spoon_stall_mm_median']):>11}  {fc}")

    print(f"\n  verdict: {verdict}")
    print("\n  controls:")
    for k, v in controls.items():
        print(f"    {k}: {v}")
    print(f"\n  wrote {a.out}")
    return 0 if all(controls.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
