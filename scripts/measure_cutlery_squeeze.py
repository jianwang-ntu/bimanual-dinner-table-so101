#!/usr/bin/env python3
"""Is the cutlery pinch squeezing the fork OUT of the jaws that hold it?

F-FORK-HANDOFF-001 found that the placing arm never holds the fork.  This
probe's predecessor, ``measure_fork_knockout.py``, instrumented every geom of
both arms -- not just the 23 per arm that carry "gripper" or "jaw" in their
body name -- and found what happens instead:

  * on 7 of 10 seeds the GIVER lifts the fork and carries it to the hand-off;
  * the giver's jaw-on-fork contact force over that carry peaks at 84-92 N;
  * the force then collapses to 0 in 6-14 ms while the jaw separation stays
    at 6.8-8.2 mm -- the jaws do not move and the fork is simply gone;
  * on 2 of the 7 a taker body is in contact inside +/-50 ms of the loss
    (``left_gripper``, ``left_camera_mount``), but on 2 others the taker never
    touches the fork ANYWHERE in the episode and the fork is lost anyway.

So a taker knock is not the mechanism.  The remaining candidate is the pinch
itself.  ``pinch``'s ``squeeze`` is an ABSOLUTE 5 mm, and it was set on the mug,
where 5 mm of a 60 mm wall is 8 percent.  ``fork_handle`` is a box 12 mm across
the jaw axis, so the same constant asks the jaws to close 5 mm INSIDE a 12 mm
object -- 42 percent -- and 60-90 N is what the contact solver produces to
resolve that.  A flat box held at 42 percent interference leaves as soon as the
pressure across its faces goes asymmetric.

This sweeps that one number and nothing else, and it is honest about which way
it could come out: a squeeze too small does not grip at all, so the sweep runs
cells on BOTH sides of the shipped value.

Reported per cell: fork_placed, spoon_placed, subgoals, task_success over the
same 10 seeds and the same unchanged scorer, plus the peak carry force.

Controls, each able to fail:

  shipped_cell_reproduces  the squeeze=0.005 cell must reproduce the ten
                           final_to_target_mm numbers the shipped controller
                           publishes in evidence/fork_release.json, pinned as a
                           literal.  This is what makes the other cells
                           comparable to the shipped entry rather than to a
                           neighbouring configuration -- and it is also the
                           proof that naming the constant CUTLERY_SQUEEZE
                           changed no behaviour at its default.
  knob_is_live             the COMMANDED jaw separation must actually differ
                           between cells, read back from the controller's own
                           pinch closure against the live model.  A sweep whose
                           cells all command the same number swept nothing.
  sweep_is_sensitive       NEGATIVE CONTROL.  The deliberately-too-tight cell
                           (squeeze at the handle's full width, which the
                           ``floor`` clamps to 4 mm of separation on a 12 mm
                           bar) must not score BETTER than the best cell.  If
                           an obviously destructive setting scores as well as
                           the best one, this measurement cannot see the
                           quantity it claims to be measuring and no cell may
                           be adopted on it.
  every_cell_ran           every cell x seed must produce a scored episode.

OUTCOME, recorded here because a probe that quietly outlives the hypothesis it
was written for is how a dead idea gets re-proposed: the hypothesis is REFUTED.
At squeeze=0.0005 the jaws command 11.5 mm across a 12.0 mm handle -- a
twentieth of the shipped interference -- and the peak force is still 74.1 N and
the score still 18/50, indistinguishable from the shipped cell.  Across the
whole sweep fork_placed is 1-3 of 10 and sub-goals 15-18 of 50, which at n=10 is
noise.  Interference is not the cause and no cell is adoptable.

Writes evidence/cutlery_squeeze.json.  Adopts nothing: it reports.
Run:  python3 scripts/measure_cutlery_squeeze.py --seeds 10
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

SHIPPED = 0.005
# Both sides of the shipped value.  0.012 is the handle's own full width: it
# asks for zero separation and is clamped by ``pinch``'s floor to 4 mm, i.e.
# maximum interference.  It is the negative control, not a candidate.
CELLS = (0.0005, 0.001, 0.002, 0.003, SHIPPED, 0.012)
DESTRUCTIVE = 0.012

# evidence/fork_release.json -> summary.final_to_target_mm, shipped controller.
PUBLISHED_FINAL_MM = [158.9, 149.4, 14.0, 54.6, 70.2, 83.0, 159.8, 35.9,
                      143.6, 20.3]
REPRO_TOL_MM = 1.0


def _mm(v) -> float:
    return round(float(v) * 1000.0, 1)


def run(args) -> dict:
    squeeze, seed = args
    from envs.task import TaskMonitor, gripper_geoms
    from envs import randomize as R
    from envs import controller as C
    from envs import scene_source

    C.CUTLERY_SQUEEZE = squeeze          # read at script-build time

    model, data, _log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_fork")
    fork_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
    giver_jaws = gripper_geoms(model, "right_")

    # What the knob actually commands, read back from the controller's own
    # closure against this model -- not recomputed here from the constant.
    grip_r = C._grip(model, "right")
    commanded_sep_mm = _mm(grip_r.sep_for_q(
        C.pinch(C.geom_width("fork_handle", 0), squeeze=squeeze)(
            model, data, grip_r)))

    _f6 = np.zeros(6)
    peak = {"force": 0.0}

    def on_step(d):
        tot = 0.0
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if ((g1 in fork_geoms and g2 in giver_jaws)
                    or (g2 in fork_geoms and g1 in giver_jaws)):
                mujoco.mj_contactForce(model, d, c, _f6)
                tot += abs(float(_f6[0]))
        if tot > peak["force"]:
            peak["force"] = tot

    C.run_dinner_table(model, data, monitor=mon, on_step=on_step)
    rep = mon.report(data)
    fork, tgt = data.xpos[bid], data.site_xpos[sid]
    return {
        "squeeze": squeeze,
        "seed": seed,
        "commanded_sep_mm": commanded_sep_mm,
        "peak_giver_force_N": round(peak["force"], 2),
        "final_to_target_mm": _mm(np.linalg.norm(fork[:2] - tgt[:2])),
        "subgoals_met": int(rep["subgoals_met"]),
        "task_success": bool(rep["task_success"]),
        "subgoals": {k: bool(v) for k, v in rep["subgoals"].items()},
        "handoffs": len(rep["handoffs"]),
        "objects_dropped": rep["objects_dropped"],
    }


def _job(args) -> dict:
    try:
        return run(args)
    except Exception as exc:
        import traceback
        return {"squeeze": args[0], "seed": args[1],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1200:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(EVID / "cutlery_squeeze.json"))
    a = ap.parse_args()

    jobs = [(s, n) for s in CELLS for n in range(a.seeds)]
    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            if "error" in r:
                print(f"  sq={r['squeeze']} seed={r['seed']} ERROR {r['error']}")

    cells = {}
    for s in CELLS:
        rs = sorted([r for r in rows if r["squeeze"] == s and "error" not in r],
                    key=lambda r: r["seed"])
        if not rs:
            cells[str(s)] = {"n": 0}
            continue
        per = lambda k: [r["subgoals"][k] for r in rs]
        cells[str(s)] = {
            "n": len(rs),
            "commanded_sep_mm": rs[0]["commanded_sep_mm"],
            "subgoals_total": sum(r["subgoals_met"] for r in rs),
            "task_success": sum(r["task_success"] for r in rs),
            "fork_placed": sum(per("fork_placed")),
            "spoon_placed": sum(per("spoon_placed")),
            "plate_placed": sum(per("plate_placed")),
            "mug_placed": sum(per("mug_placed")),
            "drawer_open": sum(per("drawer_open")),
            "peak_force_N": [r["peak_giver_force_N"] for r in rs],
            "max_peak_force_N": max(r["peak_giver_force_N"] for r in rs),
            "final_to_target_mm": [r["final_to_target_mm"] for r in rs],
            "handoffs": sum(r["handoffs"] for r in rs),
        }
        print(f"  squeeze={s:<7} sep={cells[str(s)]['commanded_sep_mm']:>5} mm  "
              f"subgoals={cells[str(s)]['subgoals_total']:>3}/{5*len(rs)}  "
              f"fork={cells[str(s)]['fork_placed']}  "
              f"spoon={cells[str(s)]['spoon_placed']}  "
              f"success={cells[str(s)]['task_success']}  "
              f"peakF={cells[str(s)]['max_peak_force_N']}")

    ship = cells[str(SHIPPED)]
    got = ship.get("final_to_target_mm", [])
    deltas = [round(abs(g - p), 2)
              for g, p in zip(got, PUBLISHED_FINAL_MM)]
    repro = {
        "pass": len(got) == len(PUBLISHED_FINAL_MM)
                and all(d <= REPRO_TOL_MM for d in deltas),
        "pinned_literal": PUBLISHED_FINAL_MM,
        "measured": got,
        "abs_delta_mm": deltas,
        "tol_mm": REPRO_TOL_MM,
        "also_proves": "naming CUTLERY_SQUEEZE and threading it through the two "
                       "cutlery pinches changed nothing at its default value",
    }
    seps = {k: v.get("commanded_sep_mm") for k, v in cells.items()}
    live = {
        "pass": len({v for v in seps.values() if v is not None}) >= len(CELLS) - 1,
        "commanded_sep_mm_per_cell": seps,
        "why_it_matters": "cells that command the same jaw separation are the "
                          "same experiment run twice",
    }
    best = max((v for k, v in cells.items()
                if v.get("n") and float(k) != DESTRUCTIVE),
               key=lambda v: v["subgoals_total"], default=None)
    dest = cells.get(str(DESTRUCTIVE), {})
    sens = {
        "pass": bool(best) and bool(dest.get("n"))
                and dest["subgoals_total"] <= best["subgoals_total"],
        "destructive_cell": DESTRUCTIVE,
        "destructive_subgoals": dest.get("subgoals_total"),
        "best_candidate_subgoals": best["subgoals_total"] if best else None,
        "why_it_matters": "if maximum interference scores as well as the best "
                          "cell, this sweep cannot see the quantity it sweeps",
    }
    ran = {
        "pass": all("error" not in r for r in rows),
        "expected": len(jobs),
        "ok": sum(1 for r in rows if "error" not in r),
        "errors": [{k: r[k] for k in ("squeeze", "seed", "error")}
                   for r in rows if "error" in r][:8],
    }

    controls = {"shipped_cell_reproduces": repro, "knob_is_live": live,
                "sweep_is_sensitive": sens, "every_cell_ran": ran}
    out = {
        "probe": "measure_cutlery_squeeze.py",
        "question": "does the cutlery pinch's absolute 5 mm squeeze eject the "
                    "fork from the jaws carrying it?",
        "knob": "envs.controller.CUTLERY_SQUEEZE",
        "shipped_value": SHIPPED,
        "cells": CELLS,
        "seeds": a.seeds,
        "controls": controls,
        "per_cell": cells,
        "per_run": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print("\ncontrols:")
    for k, v in controls.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
