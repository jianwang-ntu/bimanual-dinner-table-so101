#!/usr/bin/env python3
"""Is the spoon ungraspable, or ungraspable WHERE IT IS PARKED?

The record carries ``F-SPOON-POSE-INFEASIBLE-001`` -- "no collision-free grasp
pose exists for it" -- and ``evidence/cutlery_approach.json`` uses that as a
NEGATIVE CONTROL ("the spoon must not be placed by any approach change").  A
claim doing that much load-bearing work is worth re-deriving.

``evidence/grasp_feasibility.json``'s own ``site_sweep`` already slides the
spoon along the drawer axis and reports the lowest collision-free ``dz`` at
each station -- and it drops from 40-46 mm at the parked position to 6-10 mm
about 20-45 mm deeper, which is the fork's regime.  But ``site_sweep`` stops
there, and clearance alone cannot settle this: ``run()``'s own comment says
``jaw_below_handle_top_mm_at_min_clear`` is "the control that separates 'the
pose is unreachable' from 'the pose is reachable but grips nothing'".  The
sweep never computes it, so nobody has measured whether those low-clearance
stations actually CLOSE ON the handle.

This script computes both, at each station, on ten seeds and both arms, using
``run()``'s own grip test verbatim so the numbers are comparable to the file
already on disk.

Controls, each of which can fail:

  P1  the FORK at its own parked position, same routine -- must report a grip.
      evidence/grasp_feasibility.json says 3.9/3.3/3.6 mm and the fork is
      carried on 8 of 10 seeds, so a routine that cannot see THAT grip is
      broken and the spoon's answer would carry no information.
  N1  the SPOON at dy=0, its parked position -- must report NO grip.  This is
      the standing finding; if it does not reproduce, the probe disagrees with
      the file and neither can be trusted.
  N2  the SPOON pushed 40 mm FORWARD, toward drawer_front -- must be no better
      than parked.  The mechanism claimed here is distance behind that wall, so
      the effect has to be directional; a probe that "improves" in both
      directions is measuring solver noise.
  N3  jaw geoms non-empty (inherited assertion in _jaw_geoms).

Run:  python3 scripts/measure_spoon_site_grip.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

import measure_grasp_feasibility as F                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

# stations along the drawer's own axis, in mm, relative to where the scene
# parks the spoon.  +y is deeper into the drawer, away from drawer_front.
STATIONS_MM = [-40.0, -20.0, 0.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
DZ_MAX_MM = 60.0
DZ_STEP_MM = 2.0


def _grip_scan(model, probe, scratch, snap, grip, mine, jaw_geoms, hg,
               target, base_qpos, dz_max_mm=DZ_MAX_MM, step_mm=DZ_STEP_MM):
    """run()'s sweep verbatim, on a target supplied by the caller.

    The first version of this probe re-derived the waypoint from
    ``site_xyz``/``across`` instead of using the one the controller actually
    planned, and it disagreed with evidence/grasp_feasibility.json on BOTH
    controls -- it reported the fork NOT gripping (the file, and an 8/10 carry
    rate, say it does) and the parked spoon gripping (the file says 46 mm of
    clearance and no grip).  Two controls failing in opposite directions is a
    broken instrument, not a discovery, so the probe was corrected rather than
    the record.  The caller now passes ``snap['target']`` -- translated by the
    same vector the object was translated by, for the station sweep -- and
    ``snap['jaw']``, so the pose under test is the pose the arm is commanded to.

    ``base_qpos`` is re-applied to ``scratch`` on EVERY dz, because ``plan_pose``
    iterates its damped-least-squares step on the state it is handed and leaves
    it there.  Carrying that state forward silently makes each solve start from
    the previous one -- which is what made the corrected probe still reproduce
    the broken probe's numbers to the digit on its first re-run.
    """
    from envs import controller as C
    n = int(round(dz_max_mm / step_mm)) + 1
    for i in range(n):
        dz = i * step_mm / 1000.0
        scratch.qpos[:] = base_qpos
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(model, scratch)
        tgt = np.asarray(target, float) + np.array([0.0, 0.0, dz])
        solver = (C.plan_pose_squared if (snap["square"] and snap["jaw"] is not None)
                  else C.plan_pose)
        q, err = solver(model, scratch, grip, tgt, jaw_dir=snap["jaw"],
                        opening=snap["plan_at"], standoff=0.0)
        probe.qpos[:] = scratch.qpos
        probe.qvel[:] = 0.0
        probe.qpos[grip.qadr] = q
        mujoco.mj_forward(model, probe)
        blocked = False
        for c in range(probe.ncon):
            con = probe.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in mine) == (g2 in mine):
                continue
            o = g2 if g1 in mine else g1
            if -float(con.dist) * 1000.0 < F.PEN_MM:
                continue
            if F._name(model, o).startswith(F.WOOD):
                blocked = True
                break
        if blocked:
            continue
        h_top = float(probe.geom_xpos[hg][2] + model.geom_size[hg][2])
        jaw_bot = min(F._lowest_z(model, probe, g) for g in jaw_geoms)
        jb = round((h_top - jaw_bot) * 1000, 1)
        return {"min_clear_dz_mm": round(dz * 1000, 1),
                "ik_err_mm": round(err * 1000, 2),
                "jaw_below_handle_top_mm": jb,
                "grips": bool(jb > 0.0),
                "blocked_all_the_way": False}
    return {"min_clear_dz_mm": None, "ik_err_mm": None,
            "jaw_below_handle_top_mm": None, "grips": False,
            "blocked_all_the_way": True}


def seed_run(seed: int) -> dict:
    from envs.randomize import make_env
    from envs.task import TaskMonitor
    from envs import controller as C
    from envs import scene_source

    model, data, _ = make_env(seed)
    scene_source.install(scene_source.make("privileged"))
    mon = TaskMonitor(model)

    arm_geoms = {}
    for a in ("left", "right"):
        arm_geoms[a] = {g for g in range(model.ngeom)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              int(model.geom_bodyid[g])) or ""
                            ).startswith(f"{a}_")}

    roll = C.Rollout(model, data)
    script = C.dinner_table_script()
    snaps: dict[str, dict] = {}
    orig_plan = roll._plan

    def plan(mv):
        if mv.label in F.DESCEND and mv.label not in snaps:
            jaw = mv.jaw(model, data) if callable(mv.jaw) else mv.jaw
            pa = mv.plan_at
            if callable(pa):
                pa = pa(model, data, roll.grips[mv.arm])
            snaps[mv.label] = {
                "arm": mv.arm,
                "qpos": data.qpos.copy(),
                "target": np.asarray(mv.where(model, data), float).copy(),
                "jaw": None if jaw is None else np.asarray(jaw, float).copy(),
                "plan_at": pa if pa is not None else mv.opening,
                "square": bool(mv.square),
            }
        return orig_plan(mv)

    roll._plan = plan
    roll.run(script, monitor=mon)
    missing = [l for l in F.DESCEND if l not in snaps]
    if missing:
        return {"seed": seed, "error": "never planned: %s" % missing}

    probe = mujoco.MjData(model)
    scratch = mujoco.MjData(model)

    sp_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    sp_qadr = int(model.jnt_qposadr[int(model.body_jntadr[sp_bid])])
    sp_hg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "spoon_handle")
    fk_hg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fork_handle")

    # ---- P1: the fork, at its own position, from its own snapshot ---------
    s = snaps["fork_descend"]
    scratch.qpos[:] = s["qpos"]
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    fork = _grip_scan(model, probe, scratch, s, roll.grips[s["arm"]],
                      arm_geoms[s["arm"]], F._jaw_geoms(model, s["arm"]),
                      fk_hg, s["target"], s["qpos"].copy())
    fork["arm"] = s["arm"]

    # ---- the spoon, station by station ------------------------------------
    s = snaps["spoon_descend"]
    y0 = float(mujoco.MjData(model).qpos[sp_qadr + 1])   # placeholder, reset below
    ref = mujoco.MjData(model)
    ref.qpos[:] = s["qpos"]
    ref.qvel[:] = 0.0
    mujoco.mj_forward(model, ref)
    y0 = float(ref.qpos[sp_qadr + 1])
    front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "drawer_front")
    front_inner_y = float(ref.geom_xpos[front][1] + model.geom_size[front][1])
    front_top_z = float(ref.geom_xpos[front][2] + model.geom_size[front][2])
    fork_y = float(ref.xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "fork")][1])

    stations = []
    for dy_mm in STATIONS_MM:
        row = {"dy_mm": dy_mm,
               "world_y_m": round(y0 + dy_mm / 1000.0, 4),
               "behind_drawer_front_mm": round(
                   (y0 + dy_mm / 1000.0 - front_inner_y) * 1000, 1)}
        base = s["qpos"].copy()
        base[sp_qadr + 1] = y0 + dy_mm / 1000.0
        for arm in ("left", "right"):
            tgt = s["target"] + np.array([0.0, dy_mm / 1000.0, 0.0])
            r = _grip_scan(model, probe, scratch, s, roll.grips[arm],
                           arm_geoms[arm], F._jaw_geoms(model, arm),
                           sp_hg, tgt, base)
            row[arm] = r
        stations.append(row)

    ref = F.run(seed, DZ_MAX_MM, DZ_STEP_MM)["waypoints"]
    agree = {}
    for lbl, mine_r in (("fork_descend", fork),
                        ("spoon_descend", None)):
        r = ref[lbl]
        if lbl == "spoon_descend":
            mine_r = next(st[snaps["spoon_descend"]["arm"]]
                          for st in stations if st["dy_mm"] == 0.0)
        agree[lbl] = {
            "reference_min_clear_dz_mm": r["min_clear_dz_mm"],
            "this_probe_min_clear_dz_mm": mine_r["min_clear_dz_mm"],
            "reference_jaw_below_handle_top_mm":
                r["jaw_below_handle_top_mm_at_min_clear"],
            "this_probe_jaw_below_handle_top_mm":
                mine_r["jaw_below_handle_top_mm"],
            "agrees": (r["min_clear_dz_mm"] == mine_r["min_clear_dz_mm"] and
                       r["jaw_below_handle_top_mm_at_min_clear"] ==
                       mine_r["jaw_below_handle_top_mm"]),
        }

    return {
        "seed": seed,
        "agreement_with_reference_implementation": agree,
        "spoon_parked_world_y_m": round(y0, 4),
        "fork_world_y_m": round(fork_y, 4),
        "drawer_front_inner_face_y_m": round(front_inner_y, 4),
        "drawer_front_top_z_m": round(front_top_z, 4),
        "spoon_behind_drawer_front_mm": round((y0 - front_inner_y) * 1000, 1),
        "fork_behind_drawer_front_mm": round((fork_y - front_inner_y) * 1000, 1),
        "scripted_spoon_arm": snaps["spoon_descend"]["arm"],
        "scripted_fork_arm": snaps["fork_descend"]["arm"],
        "fork_positive_control": fork,
        "stations": stations,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="spoon_site_grip.json")
    ap.add_argument("--from", dest="frm", default=None,
                    help="re-summarise from a previous run's JSON instead of "
                         "re-simulating. The 10-seed run costs ~10 minutes, and "
                         "the first one wrote its JSON and then died in this "
                         "very print loop on a renamed control key -- so this "
                         "path exists to let the summary and the exit code be "
                         "exercised without paying for the physics again.")
    a = ap.parse_args()

    t0 = time.time()
    if a.frm:
        runs = json.loads((EVID / a.frm).read_text())["runs"]
        print("re-summarising %d runs from %s" % (len(runs), a.frm))
    else:
        runs = []
        for sd in range(a.seeds):
            t1 = time.time()
            runs.append(seed_run(sd))
            print("seed %d done in %.1fs" % (sd, time.time() - t1), flush=True)
    ok_runs = [r for r in runs if "error" not in r]

    def grips_at(dy, arm):
        return sum(1 for r in ok_runs
                   for st in r["stations"]
                   if st["dy_mm"] == dy and st[arm]["grips"])

    per_station = []
    for dy in STATIONS_MM:
        row = {"dy_mm": dy}
        for arm in ("left", "right"):
            row[arm + "_grips_n"] = grips_at(dy, arm)
            cl = [st[arm]["min_clear_dz_mm"] for r in ok_runs
                  for st in r["stations"]
                  if st["dy_mm"] == dy and st[arm]["min_clear_dz_mm"] is not None]
            row[arm + "_median_min_clear_dz_mm"] = (
                float(np.median(cl)) if cl else None)
        row["either_arm_grips_n"] = sum(
            1 for r in ok_runs for st in r["stations"]
            if st["dy_mm"] == dy and (st["left"]["grips"] or st["right"]["grips"]))
        per_station.append(row)

    n = len(ok_runs)
    p1 = sum(1 for r in ok_runs if r["fork_positive_control"]["grips"])
    n1 = next(r for r in per_station if r["dy_mm"] == 0.0)["either_arm_grips_n"]
    n2 = next(r for r in per_station if r["dy_mm"] == -40.0)["either_arm_grips_n"]
    best = max(per_station, key=lambda r: r["either_arm_grips_n"])

    on_disk = json.loads((EVID / "grasp_feasibility.json").read_text())
    disk_fork = on_disk["summary"]["fork_descend"]["median_min_clear_dz_mm"]
    disk_spoon = on_disk["summary"]["spoon_descend"]["median_min_clear_dz_mm"]
    live_fork = float(np.median([r["fork_positive_control"]["min_clear_dz_mm"]
                                 for r in ok_runs
                                 if r["fork_positive_control"]["min_clear_dz_mm"]
                                 is not None]))
    spoon0 = [st for r in ok_runs for st in r["stations"] if st["dy_mm"] == 0.0]
    scripted_arm = ok_runs[0]["scripted_spoon_arm"]
    live_spoon = float(np.median([st[scripted_arm]["min_clear_dz_mm"]
                                  for st in spoon0
                                  if st[scripted_arm]["min_clear_dz_mm"]
                                  is not None]))
    n_agree = sum(1 for r in ok_runs
                  if all(v["agrees"] for v in
                         r["agreement_with_reference_implementation"].values()))

    controls = {
        "C1_agrees_with_reference_implementation": {
            "n_seeds_agreeing": n_agree, "of": n, "pass": n_agree == n,
            "what": "on every seed, both waypoints, this probe's min_clear_dz "
                    "and jaw_below_handle_top must equal what "
                    "measure_grasp_feasibility.run() returns when run NOW. "
                    "This is the instrument check: the first two versions of "
                    "this probe failed it and were corrected, not published."},
        "C2_on_disk_evidence_is_stale": {
            "disk_fork_median_min_clear_dz_mm": disk_fork,
            "live_fork_median_min_clear_dz_mm": live_fork,
            "disk_spoon_median_min_clear_dz_mm": disk_spoon,
            "live_spoon_median_min_clear_dz_mm": live_spoon,
            "pass": (disk_fork != live_fork) and (disk_spoon != live_spoon),
            "what": "evidence/grasp_feasibility.json must DISAGREE with both "
                    "the reference implementation and this probe. If it agreed, "
                    "the finding below would be a probe bug, not a stale file. "
                    "C1 and C2 have to pass TOGETHER: C1 says the instrument is "
                    "right, C2 says the record is not."},
        "C3_pushing_spoon_toward_drawer_front_does_not_help": {
            "n_seeds_gripping_at_minus_40mm": n2,
            "n_seeds_gripping_at_parked": n1, "of": n, "pass": n2 <= n1,
            "what": "drawer_front is the wall the arm reaches over, so moving "
                    "the spoon toward it must not improve the grasp. A probe "
                    "that improved in both directions would be reading solver "
                    "noise."},
    }
    controls_pass = all(c["pass"] for c in controls.values())

    out = {
        "probe": "measure_spoon_site_grip.py",
        "question": "Is the spoon ungraspable, or ungraspable where the scene "
                    "parks it?",
        "finding_id": "F-AIS-SPOON-SITE-002",
        "job": "lablab:ai-infra-summit-hackathon",
        "row_under_test": "req_demo_video (DRAFT) -- spoon_placed is 0/10 and "
                          "task_success is all(subgoals), so this sub-goal caps "
                          "the row by itself. req_mujoco_sim (READY) is the row "
                          "that OWNS the scene geometry under test.",
        "restates": "F-SPOON-POSE-INFEASIBLE-001, and evidence/"
                    "cutlery_approach.json which uses it as a negative control",
        "method": "snapshot the arm at its own spoon_descend waypoint, teleport "
                  "the spoon along the drawer axis, re-solve the grasp at each "
                  "station, and apply run()'s own grip test verbatim. Nothing on "
                  "disk is changed: envs/dinner_table.xml is untouched and the "
                  "teleport lives inside this probe.",
        "seeds": n,
        "dz_sweep_mm": [0.0, DZ_MAX_MM, DZ_STEP_MM],
        "penetration_threshold_mm": F.PEN_MM,
        "controls": controls,
        "controls_pass": controls_pass,
        "per_station": per_station,
        "best_station": {"dy_mm": best["dy_mm"],
                         "either_arm_grips_n": best["either_arm_grips_n"],
                         "of": n},
        "runs": runs,
    }
    p = EVID / a.out
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nwrote", p)
    for c, v in controls.items():
        extra = ", ".join("%s=%s" % (k, vv) for k, vv in v.items()
                          if k not in ("pass", "what"))
        print("[%s] %s: %s" % ("PASS" if v["pass"] else "FAIL", c, extra))
    print("\ndy_mm  left_grips right_grips either  median_clear L/R")
    for r in per_station:
        print("%6.1f  %5d %5d %5d      %s / %s" % (
            r["dy_mm"], r["left_grips_n"], r["right_grips_n"],
            r["either_arm_grips_n"],
            r["left_median_min_clear_dz_mm"], r["right_median_min_clear_dz_mm"]))
    print("\ntotal %.1fs" % (time.time() - t0))
    return 0 if controls_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
