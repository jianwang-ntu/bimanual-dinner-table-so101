#!/usr/bin/env python3
"""Where do the 24 mm go: the solver, the reach, or one joint on a wall?

The repository has carried one explanation of the cutlery stall since
2026-09-05, in TECHNICAL_SUMMARY sections 5 and 8, in README.md and in the
manifest's T1 entry:

    "Servo saturation near 0.30 m of horizontal reach: three joints at the
     +/-2.94 N.m limit with no contact anywhere on the arm, while random joint
     sampling reaches 0.46 m.  IK returns poses the arm never reaches."

That sentence was measured at a FREE-SPACE waypoint 0.297 m out.  It was then
used to explain a different thing -- why ``fork_descend`` arrives roughly 7 mm
above a 5 mm handle -- and that step was never measured.  Two results make it
worth measuring:

  * ``measure_cutlery_seat.py`` moved the fork's reach over 39.1 mm of legal
    seats toward that 0.30 m line and the stall did not follow: the closest
    legal cell stalls 27.3 mm and the lowest stall, 22.6 mm, is 16.7 mm
    further out.  A reach-driven ceiling would not behave like that.
  * a single-seed read on 2026-09-08 found ONE joint saturated, not three, and
    its gravity term nowhere near the limit.

This probe decomposes the residual on all ten seeds, at the last simulation
step of the ``*_descend`` hold -- after the ramp and the settle, so a joint
still moving is distinguishable from one that has stopped:

  ik_err_mm       what ``plan_pose`` reported.  If the solver never found the
                  pose, nothing downstream is a tracking question.
  dz_mm           asked tip z minus achieved tip z.
  per joint       command, achieved, error, actuator force against its own
                  forcerange, |qvel|, and ``qfrc_bias`` -- the gravity and
                  Coriolis term the actuator would have to hold on its own.

``fork_above`` and ``spoon_above`` are the built-in negative control.  They are
the approach via-points: the same arm, the same seeds, a comparable reach, and
NO object under the jaws.  If the saturating joint saturates there too, this
probe is measuring the arm's posture and not the wall, and says so.

Nothing here changes any behaviour.  Writes evidence/descent_tracking.json.
Run:  python3 scripts/measure_descent_tracking.py --seeds 10
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

WAYPOINTS = ("fork_descend", "spoon_descend", "fork_above", "spoon_above")
GRASPS = ("fork_descend", "spoon_descend")
FREE = ("fork_above", "spoon_above")
SAT = 99.0          # percent of forcerange at which a joint is called saturated
STOPPED_MRAD_S = 5.0


def run(seed: int) -> dict:
    from envs import randomize as R
    from envs.task import TaskMonitor
    from envs import controller as C
    from envs import scene_source
    from envs.ik import ARM_JOINTS

    model, data, _ = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))
    roll = C.Rollout(model, data)
    orig = roll._plan
    live = {"label": "", "arm": "", "asked": None, "ikerr": 0.0}
    snap: dict[str, dict] = {}
    joints = ARM_JOINTS + ("gripper",)

    def plan(mv):
        q = orig(mv)
        live["label"] = getattr(mv, "label", "") or ""
        live["arm"] = mv.arm
        live["ikerr"] = float(roll.last_err)
        w = getattr(mv, "where", None)
        live["asked"] = (np.asarray(w(model, data), float)
                         if w is not None else None)
        return q

    roll._plan = plan

    def on_step(d):
        lb = live["label"]
        if lb not in WAYPOINTS or live["asked"] is None:
            return
        g = roll.grips[live["arm"]]
        tip = g.tip_mid(model, d)
        rec = {"waypoint": lb, "arm": live["arm"],
               "t": round(float(d.time), 3),
               "ik_err_mm": round(live["ikerr"] * 1000, 2),
               "tip_err_mm": round(float(np.linalg.norm(
                   tip - live["asked"])) * 1000, 2),
               "dz_mm": round(float(tip[2] - live["asked"][2]) * 1000, 2),
               "joints": []}
        contacts = set()
        mine = {gg for gg in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[gg])) or ""
                    ).startswith(f"{live['arm']}_")}
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if (g1 in mine) != (g2 in mine):
                other = g2 if g1 in mine else g1
                contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                               other) or f"geom{other}")
        rec["arm_contacts"] = sorted(contacts)
        for i, j in enumerate(joints):
            aid = int(g.act[i])
            qa = int(g.qadr[i])
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{live['arm']}_{j}")
            dof = int(model.jnt_dofadr[jid])
            fr = float(model.actuator_forcerange[aid][1])
            f = float(d.actuator_force[aid])
            rec["joints"].append({
                "joint": j,
                "err_mrad": round((float(d.ctrl[aid]) - float(d.qpos[qa])) * 1000, 2),
                "force_nm": round(f, 3),
                "forcerange_nm": round(fr, 3),
                "sat_pct": round(100.0 * abs(f) / fr, 1),
                "qvel_mrad_s": round(abs(float(d.qvel[dof])) * 1000, 2),
                "gravity_nm": round(abs(float(d.qfrc_bias[dof])), 3),
            })
        snap[lb] = rec           # last write wins: the end of the hold

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    return {"seed": seed, "waypoints": snap}


def _job(s):
    try:
        return run(s)
    except Exception as exc:
        return {"seed": s, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(EVID / "descent_tracking.json"))
    a = ap.parse_args()

    rows = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, range(a.seeds)):
            rows.append(r)
            if "error" in r:
                print(f"  ERROR seed {r['seed']}: {r['error']}")

    good = [r for r in rows if "waypoints" in r]
    per_wp = {}
    for wp in WAYPOINTS:
        got = [r["waypoints"][wp] for r in good if wp in r["waypoints"]]
        if not got:
            continue
        sat_counts: dict[str, int] = {}
        for rec in got:
            for j in rec["joints"]:
                if j["sat_pct"] >= SAT:
                    sat_counts[j["joint"]] = sat_counts.get(j["joint"], 0) + 1
        med = lambda ks: round(float(np.median(ks)), 2) if ks else None
        per_wp[wp] = {
            "n": len(got),
            "ik_err_mm_median": med([r["ik_err_mm"] for r in got]),
            "dz_mm_median": med([r["dz_mm"] for r in got]),
            "tip_err_mm_median": med([r["tip_err_mm"] for r in got]),
            "saturated_joint_counts": dict(
                sorted(sat_counts.items(), key=lambda kv: -kv[1])),
            "saturated_joints_per_seed_median": med(
                [sum(1 for j in r["joints"] if j["sat_pct"] >= SAT) for r in got]),
            "seeds_with_arm_contact": sum(1 for r in got if r["arm_contacts"]),
            "contact_geoms": sorted({g for r in got for g in r["arm_contacts"]}),
        }
        top = max(sat_counts, key=sat_counts.get) if sat_counts else None
        per_wp[wp]["top_saturated_joint"] = top
        if top:
            js = [j for r in got for j in r["joints"] if j["joint"] == top]
            per_wp[wp]["top_joint"] = {
                "joint": top,
                "err_mrad_median": med([j["err_mrad"] for j in js]),
                "gravity_nm_median": med([j["gravity_nm"] for j in js]),
                "forcerange_nm": js[0]["forcerange_nm"],
                "qvel_mrad_s_median": med([j["qvel_mrad_s"] for j in js]),
            }
            others = [j for r in got for j in r["joints"]
                      if j["joint"] != top and j["joint"] != "gripper"]
            per_wp[wp]["other_joints_err_mrad_max"] = round(
                max(abs(j["err_mrad"]) for j in others), 2) if others else None

    grasp = [per_wp[w] for w in GRASPS if w in per_wp]
    free = [per_wp[w] for w in FREE if w in per_wp]
    controls = {
        "all_seeds_ok": len(good) == a.seeds,
        "solver_found_the_pose": all(w["ik_err_mm_median"] < 10.0 for w in grasp),
        "tip_misses_by_much_more_than_the_solver": all(
            abs(w["dz_mm_median"]) > 2 * w["ik_err_mm_median"] for w in grasp),
        # The bar is the RECORD's own number.  TECHNICAL_SUMMARY section 5
        # says "three joints all sit at their +/-2.94 N.m limit"; this asks
        # whether that is what happens at the waypoint the claim is used to
        # explain.  A bar picked for convenience would prove nothing.
        "fewer_saturated_joints_than_the_record_claims": all(
            w["saturated_joints_per_seed_median"] < 3.0 for w in grasp),
        "the_saturating_joint_is_not_holding_gravity": all(
            w["top_joint"]["gravity_nm_median"] < 0.5 * w["top_joint"]["forcerange_nm"]
            for w in grasp if w.get("top_joint")),
        "it_has_stopped_moving": all(
            w["top_joint"]["qvel_mrad_s_median"] < STOPPED_MRAD_S
            for w in grasp if w.get("top_joint")),
        "the_arm_is_touching_something_there": all(
            w["seeds_with_arm_contact"] == w["n"] for w in grasp),
        "free_waypoints_are_not_saturated": all(
            w["saturated_joints_per_seed_median"] == 0.0 for w in free),
    }

    out = {
        "probe": "measure_descent_tracking.py",
        "question": ("Is the cutlery stall the reach-driven servo saturation "
                     "the repository attributes it to, or something else?"),
        "finding_id": "F-DESCENT-TRACKING-001",
        "corrects": ("TECHNICAL_SUMMARY.md sections 5 and 8, README.md and "
                     "manifest T1 defect_open, which attribute the fork "
                     "descent's residual to reach-driven saturation of three "
                     "joints"),
        "seeds": a.seeds,
        "saturation_threshold_pct": SAT,
        "controls": controls,
        "waypoints": per_wp,
        "runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n  waypoint         n  ik_err  dz     sat_joints/seed  top joint"
          "        grav/limit   |qvel|  contacts")
    for wp in WAYPOINTS:
        w = per_wp.get(wp)
        if not w:
            continue
        tj = w.get("top_joint") or {}
        gl = (f"{tj.get('gravity_nm_median')}/{tj.get('forcerange_nm')}"
              if tj else "-")
        print(f"  {wp:15} {w['n']:>2}  {w['ik_err_mm_median']:>6} "
              f"{w['dz_mm_median']:>6}  {w['saturated_joints_per_seed_median']:>13}  "
              f"{str(w.get('top_saturated_joint')):15} {gl:>11}  "
              f"{str(tj.get('qvel_mrad_s_median')):>6}  "
              f"{w['seeds_with_arm_contact']}/{w['n']} {w['contact_geoms'][:3]}")
    print("\n  controls:")
    for k, v in controls.items():
        print(f"    {k}: {v}")
    print(f"\n  wrote {a.out}")
    return 0 if all(controls.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
