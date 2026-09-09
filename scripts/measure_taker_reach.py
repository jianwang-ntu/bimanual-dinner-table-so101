#!/usr/bin/env python3
"""Does the TAKER arm ever arrive at the fork -- and if not, whose fault is it?

``evidence/fork_knockout.json`` answered "what took the fork out of the giver's
jaws" and returned SLIPPED on 5 seeds, KNOCKED_OUT_BY_TAKER on 2, NEVER_LIFTED
on 3.  Read for a DIFFERENT column, the same file says something the verdicts
do not:

    min_giver_tip_mm   3.7  4.0  3.8  3.9  4.0  4.3  4.3  4.3  5.9  77.1
    min_taker_tip_mm  18.6 29.6 41.7 43.3 44.5 54.8 56.1 63.2 72.0 118.8

The giver's jaw midpoint gets within 4 mm of the fork on 9 of 10 seeds.  The
taker's NEVER gets closer than 18.6 mm, and on 7 of 10 seeds never closer than
41 mm.  Both arms are commanded to the same site -- ``fork_grasp`` plus a few
millimetres of z -- by the same solver.  So the hand-off does not fail at the
release: the taker is not there to receive.

This probe separates the two things that can produce that, because they have
opposite fixes:

  PLANNER error   the IK solve for ``fork_take`` did not find the pose.  The
                  number is ``ik_err_mm``, already on every trace entry and
                  never once read out per-label.  Fix = the target or the reach.
  TRACKING error  the solve succeeded and the arm did not get there -- blocked
                  by the giver, by the drawer, or the ramp ran out of time.
                  Fix = the motion, not the target.

Reported per seed, at the LAST simulator step of each phase, so the ramp has
finished:

  ik_err_mm[label]      the solver's own residual, from the trace
  track_err_mm[label]   |taker jaw midpoint - the live commanded target|
  giver holds fork      contact + normal force at that instant

Controls, each able to fail:

  reproduces_published   the per-seed final_to_target_mm column must reproduce
                         the ten numbers evidence/fork_release.json publishes
                         for the shipped controller, pinned as a literal.
  positive_control_giver the SAME tracking measurement applied to the giver's
                         own ``fork_descend`` must come out SMALL on the seeds
                         where fork_knockout recorded a real grasp.  An
                         instrument that shows both arms missing is measuring
                         itself, not the arms.
  planted_target_moves   the taker tracking error recomputed against a target
                         displaced 1000 mm in +x must exceed 900 mm.  A number
                         that does not follow its own target is not a distance
                         to that target.
  labels_all_present     every label this probe reports must be found on every
                         seed.  A phase that is silently absent reports nothing
                         and would read as a pass.

Writes evidence/taker_reach.json.
Run:  python3 scripts/measure_taker_reach.py --seeds 10
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

BODY = "fork"
TARGET = "target_fork"
GIVER = "right"
TAKER = "left"

# evidence/fork_release.json -> summary.final_to_target_mm, shipped controller.
PUBLISHED_FINAL_MM = [158.9, 149.4, 14.0, 54.6, 70.2, 83.0, 159.8, 35.9,
                      143.6, 20.3]
REPRO_TOL_MM = 1.0

# fork_knockout.json min_giver_tip_mm: the giver really grasps on these seeds.
GIVER_GRASPED_SEEDS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
GIVER_TRACK_TOL_MM = 25.0          # generous; the giver measures 3.7-5.9 mm

PLANT_MM = 1000.0
PLANT_MIN_MM = 900.0

TAKER_LABELS = ("fork_meet", "fork_take_above", "fork_take",
                "fork_taker_close", "fork_taker_lift")
GIVER_LABELS = ("fork_descend", "fork_close", "fork_lift", "fork_present",
                "fork_giver_release")


def _mm(v) -> float:
    return round(float(v) * 1000.0, 1)


def run(seed: int) -> dict:
    from envs.task import TaskMonitor, gripper_geoms
    from envs import randomize as R
    from envs import controller as C
    from envs import scene_source

    model, data, _log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET)
    fork_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
    jaw_only = {a: gripper_geoms(model, f"{a}_") for a in (GIVER, TAKER)}

    roll = C.Rollout(model, data)
    grips = {a: C._grip(model, a) for a in (GIVER, TAKER)}

    # The two live targets, rebuilt exactly as dinner_table_script emits them.
    #   taker  _handoff: Move(taker, site_xyz(grasp, dz=0.004), ...)
    #   giver  _pick:    final descend at approach*(0) + [0,0,CUTLERY_DESCEND_Z]
    taker_tgt = C.site_xyz(f"{BODY}_grasp", dz=0.004)
    giver_tgt = C.site_xyz(f"{BODY}_grasp", dz=float(C.CUTLERY_DESCEND_Z))

    _f6 = np.zeros(6)

    def grip_force(d, want: set[int]) -> float:
        tot = 0.0
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if not ((g1 in fork_geoms and g2 in want)
                    or (g2 in fork_geoms and g1 in want)):
                continue
            mujoco.mj_contactForce(model, d, c, _f6)
            tot += abs(float(_f6[0]))
        return tot

    steps: list[dict] = []

    def on_step(d):
        tt = np.asarray(taker_tgt(model, d), float)
        gt = np.asarray(giver_tgt(model, d), float)
        tmid = grips[TAKER].tip_mid(model, d)
        gmid = grips[GIVER].tip_mid(model, d)
        steps.append({
            "i": len(steps),
            "trace_n": len(roll.trace),
            "t": round(float(d.time), 3),
            "taker_track_mm": _mm(np.linalg.norm(tmid - tt)),
            "giver_track_mm": _mm(np.linalg.norm(gmid - gt)),
            "taker_plant_mm": _mm(np.linalg.norm(
                tmid - (tt + np.array([PLANT_MM / 1000.0, 0.0, 0.0])))),
            "taker_to_fork_mm": _mm(np.linalg.norm(tmid - d.xpos[bid])),
            "giver_to_fork_mm": _mm(np.linalg.norm(gmid - d.xpos[bid])),
            "giver_force_N": round(grip_force(d, jaw_only[GIVER]), 2),
            "taker_force_N": round(grip_force(d, jaw_only[TAKER]), 2),
            "fork_z_mm": _mm(d.xpos[bid][2]),
        })

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    # Trace entries are appended at the START of each script entry, one per arm.
    # Steps executed for that entry all carry trace_n == (length after append).
    # Group trace entries by their shared t to recover the script entries.
    entries: list[dict] = []
    for k, tr in enumerate(roll.trace):
        if entries and entries[-1]["t"] == tr["t"]:
            entries[-1]["labels"][f"{tr['label']}"] = tr["ik_err_mm"]
            entries[-1]["n"] = k + 1
        else:
            entries.append({"t": tr["t"], "n": k + 1,
                            "labels": {tr["label"]: tr["ik_err_mm"]}})

    by_n: dict[int, list[dict]] = {}
    for s in steps:
        by_n.setdefault(s["trace_n"], []).append(s)

    ik_err, track_err, at_end = {}, {}, {}
    for e in entries:
        blk = by_n.get(e["n"])
        for label, err in e["labels"].items():
            if label not in TAKER_LABELS and label not in GIVER_LABELS:
                continue
            if label in ik_err:              # only the fork phase; take first
                continue
            ik_err[label] = err
            if not blk:
                continue
            last = blk[-1]
            track_err[label] = (last["taker_track_mm"] if label in TAKER_LABELS
                                else last["giver_track_mm"])
            at_end[label] = {
                "t": last["t"],
                "taker_track_mm": last["taker_track_mm"],
                "giver_track_mm": last["giver_track_mm"],
                "taker_plant_mm": last["taker_plant_mm"],
                "taker_to_fork_mm": last["taker_to_fork_mm"],
                "giver_to_fork_mm": last["giver_to_fork_mm"],
                "taker_force_N": last["taker_force_N"],
                "giver_force_N": last["giver_force_N"],
                "fork_z_mm": last["fork_z_mm"],
            }

    fork = data.xpos[bid]
    tgt = data.site_xpos[sid]
    return {
        "seed": seed,
        "final_to_target_mm": _mm(np.linalg.norm(fork[:2] - tgt[:2])),
        "subgoals_met": int(rep["subgoals_met"]),
        "ik_err_mm": ik_err,
        "track_err_mm": track_err,
        "at_phase_end": at_end,
        "min_taker_track_mm": min(s["taker_track_mm"] for s in steps),
        "min_taker_to_fork_mm": min(s["taker_to_fork_mm"] for s in steps),
        "min_giver_to_fork_mm": min(s["giver_to_fork_mm"] for s in steps),
        "max_taker_force_N": max(s["taker_force_N"] for s in steps),
        "labels_missing": sorted(set(TAKER_LABELS + GIVER_LABELS) - set(ik_err)),
        "steps": len(steps),
    }


def _job(seed: int) -> dict:
    try:
        return run(seed)
    except Exception as exc:
        import traceback
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(EVID / "taker_reach.json"))
    a = ap.parse_args()

    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, range(a.seeds)):
            rows.append(r)
            if "error" in r:
                print(f"  seed={r['seed']:<2} ERROR {r['error']}")
            else:
                print(f"  seed={r['seed']:<2} "
                      f"ik(fork_take)={r['ik_err_mm'].get('fork_take')} "
                      f"track(fork_take)={r['track_err_mm'].get('fork_take')} "
                      f"min_taker_to_fork={r['min_taker_to_fork_mm']}")
    rows.sort(key=lambda r: r["seed"])
    ok = [r for r in rows if "error" not in r]

    got = [r["final_to_target_mm"] for r in ok]
    deltas = [round(abs(g - p), 2)
              for g, p in zip(got, PUBLISHED_FINAL_MM[:len(got)])]
    repro = {
        "pass": (len(ok) == len(PUBLISHED_FINAL_MM)
                 and all(d <= REPRO_TOL_MM for d in deltas)),
        "pinned_literal": PUBLISHED_FINAL_MM,
        "measured": got,
        "abs_delta_mm": deltas,
        "tol_mm": REPRO_TOL_MM,
        "why_it_matters": "a probe that has drifted off the shipped "
                          "configuration is measuring a different robot",
    }

    gv = {r["seed"]: r["track_err_mm"].get("fork_descend")
          for r in ok if r["seed"] in GIVER_GRASPED_SEEDS}
    gv_ok = [s for s, v in gv.items() if v is not None and v <= GIVER_TRACK_TOL_MM]
    pos = {
        "pass": len(gv_ok) >= len(GIVER_GRASPED_SEEDS) - 1 and len(gv) > 0,
        "giver_fork_descend_track_mm": gv,
        "tol_mm": GIVER_TRACK_TOL_MM,
        "seeds_within_tol": sorted(gv_ok),
        "why_it_matters": "the same instrument must show the arm that DOES "
                          "grasp arriving; if both arms miss, the number is "
                          "about the instrument",
    }

    pl = {r["seed"]: (r["at_phase_end"].get("fork_take") or {}).get("taker_plant_mm")
          for r in ok}
    pl_vals = [v for v in pl.values() if v is not None]
    plant = {
        "pass": bool(pl_vals) and all(v >= PLANT_MIN_MM for v in pl_vals),
        "displacement_mm": PLANT_MM,
        "min_required_mm": PLANT_MIN_MM,
        "measured": pl,
        "why_it_matters": "NEGATIVE CONTROL: a distance that does not follow "
                          "its own target is not a distance to that target",
    }

    miss = {r["seed"]: r["labels_missing"] for r in ok if r["labels_missing"]}
    labels = {
        "pass": not miss,
        "expected": sorted(set(TAKER_LABELS + GIVER_LABELS)),
        "missing_by_seed": miss,
        "why_it_matters": "a phase that is silently absent reports nothing, "
                          "which would read as a pass",
    }

    def col(key, label):
        return {r["seed"]: r[key].get(label) for r in ok}

    summary = {
        "ik_err_mm_fork_take": col("ik_err_mm", "fork_take"),
        "track_err_mm_fork_take": col("track_err_mm", "fork_take"),
        "ik_err_mm_fork_take_above": col("ik_err_mm", "fork_take_above"),
        "track_err_mm_fork_take_above": col("track_err_mm", "fork_take_above"),
        "ik_err_mm_fork_descend": col("ik_err_mm", "fork_descend"),
        "track_err_mm_fork_descend": col("track_err_mm", "fork_descend"),
        "min_taker_to_fork_mm": {r["seed"]: r["min_taker_to_fork_mm"] for r in ok},
        "min_giver_to_fork_mm": {r["seed"]: r["min_giver_to_fork_mm"] for r in ok},
        "max_taker_force_N": {r["seed"]: r["max_taker_force_N"] for r in ok},
        "seeds_taker_ever_loaded_fork": sorted(r["seed"] for r in ok
                                               if r["max_taker_force_N"] > 0.0),
        "subgoals_total": sum(r["subgoals_met"] for r in ok),
    }

    out = {
        "probe": "measure_taker_reach.py",
        "question": "does the taker arm ever arrive at the fork, and is the "
                    "miss a PLANNER (ik_err) or a TRACKING error?",
        "seeds": len(ok),
        "controls": {
            "reproduces_published": repro,
            "positive_control_giver": pos,
            "planted_target_moves": plant,
            "labels_all_present": labels,
        },
        "summary": summary,
        "per_seed": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\ncontrols: " + "  ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}"
        for k, v in out["controls"].items()))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
