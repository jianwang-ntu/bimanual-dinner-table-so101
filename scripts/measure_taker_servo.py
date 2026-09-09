#!/usr/bin/env python3
"""The taker's IK solves and the arm still does not arrive.  Which link breaks?

``evidence/taker_reach.json`` measured the hand-off's real failure, and it is
not the one any previous probe looked for.  At ``fork_take`` the solver's own
residual is 1.6-7.7 mm on all ten seeds -- the pose exists and is found -- while
the taker's jaw midpoint never comes closer than 18.6 mm to the fork and on 7 of
10 seeds never closer than 41 mm.  The giver, running the same solver against
the same site, reaches 3.7-5.9 mm.  The taker is not mis-planned.  It is planned
and does not go.

Between a solved ``q`` and a jaw that arrives there are four links.  A single
"tracking error" number cannot tell them apart, and they have different fixes:

  TARGET_DRIFT  the target is a callable evaluated at PLAN time; the fork then
                moves during the phase.  A distance measured against the target
                at phase END charges the arm for the fork's own motion.
  CLIPPED       ``Rollout.run`` does ``np.clip(self._plan(mv), g.lo, g.hi)`` and
                the IK residual is computed on the UNCLIPPED q.
  PLAN_ERR      the commanded joint vector itself does not put the jaws on the
                target -- e.g. ``_plan`` overwrites the solved gripper value
                with ``opening`` AFTER the solve, so a pose solved at the pinch
                width executes at the open width.
  EXEC_ERR      the command is right and the arm does not reach it: blocked, or
                the ramp ran out of time.

So the total is decomposed, not summarised:

  plan_err_mm   |jaws under the COMMANDED joint vector - target at plan time|
  exec_err_mm   |jaws at phase end - jaws under the commanded joint vector|
  drift_mm      |target at phase end - target at plan time|

Controls, each able to fail, each reported as it comes out:

  reproduces_published    the per-seed final_to_target_mm column must reproduce
                          the ten numbers evidence/fork_release.json publishes.
  fk_matches_sim          the forward kinematics used for plan_err must, when
                          fed the ACHIEVED qpos, reproduce the simulator's own
                          jaw midpoint to under 1 mm.  Otherwise plan_err is a
                          number about this script, not about the robot.
  positive_control_giver  the same decomposition on the giver's own
                          ``fork_descend`` -- the grasp fork_knockout measured
                          at 3.7-5.9 mm -- must show BOTH plan_err and exec_err
                          small, on the seeds where that grasp happened.  This
                          replaces a first version that thresholded a raw servo
                          error at 0.05 rad and FAILED: 7 of 10 seeds sat at
                          0.023-0.025 rad, two grasping seeds at 0.088/0.090,
                          and seed 6 -- where the giver never grasped at all --
                          at 0.588.  The threshold was guessed a priori and was
                          wrong; the replacement is measured in millimetres at
                          the jaws, which is the quantity the claim is about.
  clip_detector_fires     the clip metric, applied to a vector 5 rad outside
                          the range, must report that displacement.
  contact_detector_fires  the arm-contact watch must report the giver holding
                          the fork during its own carry.
  labels_all_present      every phase reported must be found on every seed.

Writes evidence/taker_servo.json.
Run:  python3 scripts/measure_taker_servo.py --seeds 10
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

PUBLISHED_FINAL_MM = [158.9, 149.4, 14.0, 54.6, 70.2, 83.0, 159.8, 35.9,
                      143.6, 20.3]
REPRO_TOL_MM = 1.0

# fork_knockout.json min_giver_tip_mm: on these seeds the giver's jaw midpoint
# came within 6 mm of the fork, i.e. it really grasped.  Seed 6 (77.1 mm) did
# not and is deliberately excluded from the positive control.
GIVER_GRASPED_SEEDS = (0, 1, 2, 3, 4, 5, 7, 8, 9)
GIVER_TOL_MM = 30.0
FK_TOL_MM = 1.0
PLANT_RAD = 5.0

WATCH = ("fork_meet", "fork_take_above", "fork_take", "fork_taker_close",
         "fork_descend", "fork_close", "fork_lift")
TAKER_PHASES = ("fork_meet", "fork_take_above", "fork_take", "fork_taker_close")


def _r(v, n=4):
    return [round(float(x), n) for x in np.asarray(v).ravel()]


def _mm(v) -> float:
    return round(float(v) * 1000.0, 1)


def run(seed: int) -> dict:
    from envs.task import TaskMonitor
    from envs import randomize as R
    from envs import controller as C
    from envs import scene_source

    model, data, _log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET)
    geom_body = {g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(model.geom_bodyid[g])) or f"body{g}")
                 for g in range(model.ngeom)}
    arm_geoms = {a: {g for g, nm in geom_body.items() if nm.startswith(f"{a}_")}
                 for a in (GIVER, TAKER)}
    grips = {a: C._grip(model, a) for a in (GIVER, TAKER)}
    fk = mujoco.MjData(model)

    def tip_mid_for(arm, cmd, ref_qpos):
        """Jaw midpoint the model gives for this arm at this joint command."""
        fk.qpos[:] = ref_qpos
        fk.qvel[:] = 0.0
        fk.qpos[grips[arm].qadr] = np.asarray(cmd, float)
        mujoco.mj_kinematics(model, fk)
        return grips[arm].tip_mid(model, fk).copy()

    cap: dict[str, dict] = {}
    base_plan = C.Rollout._plan

    class Probe(C.Rollout):
        def _plan(self, mv):
            q = base_plan(self, mv)
            if mv.label in WATCH and mv.label not in cap:
                tgt = None
                if callable(getattr(mv, "where", None)):
                    try:
                        tgt = np.asarray(mv.where(self.model, self.data), float).copy()
                    except Exception:
                        tgt = None
                cap[mv.label] = {"cmd_raw": np.asarray(q, float).copy(),
                                 "target_at_plan": tgt,
                                 "qpos_at_plan": self.data.qpos.copy()}
            return q

    roll = Probe(model, data)
    steps: list[dict] = []

    def touching(d, arm):
        out = set()
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if g1 in arm_geoms[arm]:
                out.add(geom_body[g2])
            elif g2 in arm_geoms[arm]:
                out.add(geom_body[g1])
        return {n for n in out if not n.startswith(f"{arm}_")}

    def on_step(d):
        steps.append({
            "i": len(steps), "trace_n": len(roll.trace),
            "t": round(float(d.time), 3),
            "qpos": d.qpos.copy(),
            "taker_ctrl": d.ctrl[grips[TAKER].act].copy(),
            "giver_ctrl": d.ctrl[grips[GIVER].act].copy(),
            "taker_qpos": d.qpos[grips[TAKER].qadr].copy(),
            "giver_qpos": d.qpos[grips[GIVER].qadr].copy(),
            "taker_tip": grips[TAKER].tip_mid(model, d).copy(),
            "giver_tip": grips[GIVER].tip_mid(model, d).copy(),
            "taker_touching": sorted(touching(d, TAKER)),
            "giver_touching_fork": BODY in touching(d, GIVER),
            "fork_xpos": d.xpos[bid].copy(),
        })

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    entries: list[dict] = []
    for k, tr in enumerate(roll.trace):
        if entries and entries[-1]["t"] == tr["t"]:
            entries[-1]["labels"][tr["label"]] = tr["ik_err_mm"]
            entries[-1]["n"] = k + 1
        else:
            entries.append({"t": tr["t"], "n": k + 1,
                            "labels": {tr["label"]: tr["ik_err_mm"]}})
    by_n: dict[int, list[dict]] = {}
    for s in steps:
        by_n.setdefault(s["trace_n"], []).append(s)

    # The live target function, rebuilt exactly as dinner_table_script emits it.
    tgt_fn = {"fork_take": C.site_xyz(f"{BODY}_grasp", dz=0.004),
              "fork_take_above": C.site_xyz(f"{BODY}_grasp", dz=0.040),
              "fork_descend": C.site_xyz(f"{BODY}_grasp",
                                         dz=float(C.CUTLERY_DESCEND_Z))}

    phases: dict[str, dict] = {}
    fk_checks: list[float] = []
    for e in entries:
        blk = by_n.get(e["n"]) or []
        for label in e["labels"]:
            if label not in WATCH or label in phases or not blk:
                continue
            arm = TAKER if label in TAKER_PHASES else GIVER
            g, c = grips[arm], cap.get(label)
            last = blk[-1]
            key = "taker" if arm == TAKER else "giver"
            ctrl_end = last[f"{key}_ctrl"]
            qpos_end = last[f"{key}_qpos"]
            tip_end = last[f"{key}_tip"]

            # CONTROL: FK fed the ACHIEVED qpos must reproduce the sim's jaws.
            fk_tip = tip_mid_for(arm, qpos_end, last["qpos"])
            fk_checks.append(_mm(np.linalg.norm(fk_tip - tip_end)))

            row = {"arm": arm, "ik_err_mm": e["labels"][label],
                   "steps": len(blk),
                   "max_servo_err_rad": round(float(np.max(np.abs(
                       np.asarray(qpos_end) - np.asarray(ctrl_end)))), 4),
                   "arm_touching_at_end": last["taker_touching"] if arm == TAKER
                                          else sorted(touching(data, GIVER))}
            if c is not None:
                raw = c["cmd_raw"]
                clipped = np.clip(raw, g.lo, g.hi)
                row["max_clip_rad"] = round(float(np.max(np.abs(raw - clipped))), 4)
                plan_tip = tip_mid_for(arm, clipped, c["qpos_at_plan"])
                row["exec_err_mm"] = _mm(np.linalg.norm(tip_end - plan_tip))
                if c["target_at_plan"] is not None:
                    row["plan_err_mm"] = _mm(np.linalg.norm(
                        plan_tip - c["target_at_plan"]))
                    fn = tgt_fn.get(label)
                    if fn is not None:
                        row["drift_mm"] = _mm(np.linalg.norm(
                            np.asarray(fn(model, data), float)
                            - c["target_at_plan"]))
            phases[label] = row

    fork, tgt = data.xpos[bid], data.site_xpos[sid]
    return {
        "seed": seed,
        "final_to_target_mm": _mm(np.linalg.norm(fork[:2] - tgt[:2])),
        "subgoals_met": int(rep["subgoals_met"]),
        "phases": phases,
        "fk_max_err_mm": max(fk_checks) if fk_checks else None,
        "labels_missing": sorted(set(WATCH) - set(phases)),
        "giver_ever_touched_fork": any(s["giver_touching_fork"] for s in steps),
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
    ap.add_argument("--out", default=str(EVID / "taker_servo.json"))
    a = ap.parse_args()

    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, range(a.seeds)):
            rows.append(r)
            if "error" in r:
                print(f"  seed={r['seed']:<2} ERROR {r['error']}")
            else:
                p = r["phases"].get("fork_take", {})
                print(f"  seed={r['seed']:<2} plan_err={p.get('plan_err_mm')} "
                      f"exec_err={p.get('exec_err_mm')} drift={p.get('drift_mm')} "
                      f"clip={p.get('max_clip_rad')} "
                      f"touch={p.get('arm_touching_at_end')}")
    rows.sort(key=lambda r: r["seed"])
    ok = [r for r in rows if "error" not in r]

    got = [r["final_to_target_mm"] for r in ok]
    deltas = [round(abs(g - p), 2) for g, p in zip(got, PUBLISHED_FINAL_MM[:len(got)])]
    repro = {"pass": len(ok) == len(PUBLISHED_FINAL_MM)
                     and all(d <= REPRO_TOL_MM for d in deltas),
             "pinned_literal": PUBLISHED_FINAL_MM, "measured": got,
             "abs_delta_mm": deltas, "tol_mm": REPRO_TOL_MM}

    fkv = {r["seed"]: r["fk_max_err_mm"] for r in ok}
    fkctl = {"pass": all(v is not None and v <= FK_TOL_MM for v in fkv.values()),
             "max_err_mm_by_seed": fkv, "tol_mm": FK_TOL_MM,
             "why_it_matters": "if the forward kinematics does not reproduce "
                               "the simulator's own jaws, plan_err is a number "
                               "about this script, not about the robot"}

    gp = {r["seed"]: (r["phases"].get("fork_descend") or {}).get("plan_err_mm")
          for r in ok if r["seed"] in GIVER_GRASPED_SEEDS}
    ge = {r["seed"]: (r["phases"].get("fork_descend") or {}).get("exec_err_mm")
          for r in ok if r["seed"] in GIVER_GRASPED_SEEDS}
    bad = sorted(s for s in gp
                 if gp[s] is None or ge[s] is None
                 or gp[s] > GIVER_TOL_MM or ge[s] > GIVER_TOL_MM)
    pos = {"pass": not bad, "seeds_tested": sorted(gp),
           "giver_fork_descend_plan_err_mm": gp,
           "giver_fork_descend_exec_err_mm": ge,
           "tol_mm": GIVER_TOL_MM, "seeds_over_tol": bad,
           "excluded": {"seed_6": "fork_knockout min_giver_tip_mm 77.1 -- the "
                                  "giver never grasped on this seed, so it is "
                                  "not a positive control"},
           "why_it_matters": "the same decomposition on the grasp that DOES "
                             "work must come out small, or the numbers below "
                             "are about the instrument"}

    plo, phi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    planted = phi + PLANT_RAD
    got_clip = float(np.max(np.abs(planted - np.clip(planted, plo, phi))))
    clipctl = {"pass": abs(got_clip - PLANT_RAD) < 1e-9,
               "displacement_rad": PLANT_RAD, "measured_rad": got_clip,
               "why_it_matters": "a clip detector that cannot say yes proves "
                                 "nothing by saying no"}

    fired = [r["seed"] for r in ok if r["giver_ever_touched_fork"]]
    conctl = {"pass": len(fired) >= len(ok) - 1, "seeds_where_fired": fired,
              "why_it_matters": "a contact test that never fires proves "
                                "nothing by not firing"}

    miss = {r["seed"]: r["labels_missing"] for r in ok if r["labels_missing"]}
    labels = {"pass": not miss, "expected": sorted(WATCH), "missing_by_seed": miss}

    def col(label, key):
        return {r["seed"]: (r["phases"].get(label) or {}).get(key) for r in ok}

    verdicts = {}
    for r in ok:
        p = r["phases"].get("fork_take") or {}
        clip = p.get("max_clip_rad") or 0.0
        pe, xe = p.get("plan_err_mm") or 0.0, p.get("exec_err_mm") or 0.0
        touch = [b for b in (p.get("arm_touching_at_end") or [])
                 if b.startswith(f"{GIVER}_")]
        if clip > 0.01:
            v = "CLIPPED"
        elif xe > pe and touch:
            v = "BLOCKED_BY_GIVER"
        elif xe > pe:
            v = "EXEC_DOMINATES"
        else:
            v = "PLAN_DOMINATES"
        verdicts[r["seed"]] = v

    out = {
        "probe": "measure_taker_servo.py",
        "question": "the taker's IK solves and the arm does not arrive -- is "
                    "the command CLIPPED, is the COMMAND ITSELF wrong "
                    "(plan_err), or does the arm fail to execute it (exec_err), "
                    "and is the giver's own arm what stops it?",
        "seeds": len(ok),
        "controls": {"reproduces_published": repro,
                     "fk_matches_sim": fkctl,
                     "positive_control_giver": pos,
                     "clip_detector_fires": clipctl,
                     "contact_detector_fires": conctl,
                     "labels_all_present": labels},
        "summary": {
            "verdict_fork_take": verdicts,
            "verdict_counts": {v: sum(1 for x in verdicts.values() if x == v)
                               for v in sorted(set(verdicts.values()))},
            "plan_err_mm_fork_take": col("fork_take", "plan_err_mm"),
            "exec_err_mm_fork_take": col("fork_take", "exec_err_mm"),
            "drift_mm_fork_take": col("fork_take", "drift_mm"),
            "max_clip_rad_fork_take": col("fork_take", "max_clip_rad"),
            "arm_touching_fork_take": col("fork_take", "arm_touching_at_end"),
            "plan_err_mm_fork_take_above": col("fork_take_above", "plan_err_mm"),
            "exec_err_mm_fork_take_above": col("fork_take_above", "exec_err_mm"),
            "plan_err_mm_fork_descend": col("fork_descend", "plan_err_mm"),
            "exec_err_mm_fork_descend": col("fork_descend", "exec_err_mm"),
        },
        "per_seed": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\ncontrols: " + "  ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                                     for k, v in out["controls"].items()))
    print("verdicts:", out["summary"]["verdict_counts"])
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
