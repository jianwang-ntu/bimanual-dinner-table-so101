#!/usr/bin/env python3
"""Why every route to the cutlery stops at the same absolute height.

``scripts/measure_cutlery_zcross.py`` crossed the descent height with the two
knobs that each halved the arrival gap -- 24 cells, 10 seeds, 240 rollouts --
and found something none of the eight earlier probes could see, because each of
them swept ONE knob and read the gap RELATIVE to a target that moved with it:

    asked tip z   0.7865 -> 0.7985 m   (+12.0 mm, four descent heights)
    lowest tip z  0.8013 -> 0.8032 m   (+ 1.9 mm)

The arm does not stall a distance above the target.  It stops at an ABSOLUTE
height, ~0.800 m, and the "arrival gap" shrinks only because the target rises
toward it.  That is why seven routes that changed WHERE the arm goes and one
that changed the hand's WIDTH all changed nothing: none of them could move a
constant.

This probe measures the constant kinematically, with no dynamics, no servos and
no contact solver, so saturation cannot be the explanation:

  * replay the rollout, capture the joint vector ``plan_pose`` RETURNS for a
    grasp waypoint,
  * put the arm on that vector in a scratch ``MjData`` and run
    ``mj_kinematics`` only,
  * take the jaw MEETING POINT (``Gripper.tip_mid``) and the lowest world z of
    any vertex of any geom on that arm, exactly, from the compiled mesh
    vertices -- and name the geom that attains it.

``drop_mm`` = meeting point z - lowest hand point z.  It is how far the hand
hangs BELOW the point it pinches with.  A hand with drop_mm = D cannot put its
meeting point closer than D to any surface it is resting on, whatever the
trajectory, the opening or the solver.

THE CONTROL is the mug.  ``mug_placed`` scores and ``fork_placed`` does not, so
if this quantity explains the difference it must be the SAME for both -- it is
a property of the hand, not of the object -- and the objects must differ in how
far their graspable feature sits above their own support.  If ``drop_mm`` comes
out different at the two waypoints then this probe is measuring arm posture and
not the hand, and the finding is void.  Both are reported either way.

Writes evidence/hand_floor.json.
Run:  python3 scripts/measure_hand_floor.py --seeds 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

WAYPOINTS = ("fork_descend", "spoon_descend", "plate_descend")
# the mug is picked through ``_shunt``, which tags each re-grasp, so its
# descend labels are ``mug_<arm><i>_descend``.  It is THE control: it is the
# one graspable this entry has ever lifted.
MUG_RE = re.compile(r"^mug_(left|right)\d+_descend$")

# The HAND, not the arm.  ``right_base`` sits on the table and is the lowest
# geom on the whole arm at every pose, which says nothing about what the hand
# can reach into a drawer.  These three bodies are exactly the ones the
# dynamic contact tallies name: geom115 and right_fixed_jaw_box3 on
# ``*_gripper``, right_camera_box2 on ``*_camera_mount``, geom77 and
# left_moving_jaw_box1 on ``*_moving_jaw_so101_v1``.
HAND_SUFFIXES = ("gripper", "camera_mount", "moving_jaw_so101_v1")


def _geom_lowest_z(model, data, g: int) -> float:
    """Exact lowest world z of geom ``g``: mesh vertices when it has them,
    the type's own support otherwise."""
    pos = np.asarray(data.geom_xpos[g], float)
    mat = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
    gtype = int(model.geom_type[g])
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        a = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        v = np.asarray(model.mesh_vert[a:a + n], float).reshape(-1, 3)
        return float((pos + v @ mat.T)[:, 2].min())
    s = np.asarray(model.geom_size[g], float)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        c = np.array([[sx, sy, sz] for sx in (-s[0], s[0])
                      for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
        return float((pos + c @ mat.T)[:, 2].min())
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(pos[2] - s[0])
    if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        ends = np.array([[0, 0, -s[1]], [0, 0, s[1]]])
        return float((pos + ends @ mat.T)[:, 2].min() - (s[0] if gtype ==
                     mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0))
    # planes and anything else: the origin is as good as it gets
    return float(pos[2] - float(model.geom_rbound[g]))


def run(seed: int) -> dict:
    from envs import randomize as R
    from envs.task import TaskMonitor
    from envs import controller as C
    from envs import scene_source

    model, data, _ = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))
    roll = C.Rollout(model, data)
    orig = roll._plan

    hand_geoms = {}
    for a in ("left", "right"):
        want = {f"{a}_{sfx}" for sfx in HAND_SUFFIXES}
        hand_geoms[a] = [g for g in range(model.ngeom)
                         if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                               int(model.geom_bodyid[g])) or "")
                         in want]
        if not hand_geoms[a]:
            raise RuntimeError(f"no hand geoms found for {a}: {want}")

    scratch = mujoco.MjData(model)
    out: dict[str, dict] = {}

    def plan(mv):
        q = orig(mv)
        lb = getattr(mv, "label", "") or ""
        key = "mug_descend" if MUG_RE.match(lb) else lb
        if key in WAYPOINTS + ("mug_descend",) and key not in out:
            g = roll.grips[mv.arm]
            scratch.qpos[:] = data.qpos
            scratch.qvel[:] = 0.0
            scratch.qpos[g.qadr] = np.asarray(q, float)
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            tip = g.tip_mid(model, scratch)
            lows = [(_geom_lowest_z(model, scratch, gg), gg)
                    for gg in hand_geoms[mv.arm]]
            lowz, lowg = min(lows)
            asked = (np.asarray(mv.where(model, data), float)
                     if mv.where is not None else None)
            out[key] = {
                "waypoint": lb, "arm": mv.arm,
                "tip_mid_z_m": round(float(tip[2]), 5),
                "lowest_hand_z_m": round(float(lowz), 5),
                "drop_mm": round(float(tip[2] - lowz) * 1000, 2),
                "lowest_geom": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(lowg)) or f"geom{lowg}",
                "gripper_cmd_rad": round(float(np.asarray(q, float)[-1]), 4),
                "asked_tip_z_m": (round(float(asked[2]), 5)
                                  if asked is not None else None),
            }
        return q

    roll._plan = plan
    roll.run(C.dinner_table_script(), monitor=mon)

    # where each object's graspable feature sits above the surface under it
    def bz(name):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return float(data.xpos[b][2]) if b >= 0 else None

    def gz(name):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            return None
        return {"top_z_m": round(float(data.geom_xpos[g][2]
                                       + model.geom_size[g][2]), 5),
                "centre_z_m": round(float(data.geom_xpos[g][2]), 5)}

    return {"seed": seed, "waypoints": out,
            "surfaces": {n: gz(n) for n in
                         ("drawer_floor", "table_top", "tabletop", "table")},
            "objects": {n: bz(n) for n in ("fork", "spoon", "mug", "plate")}}


def _job(s):
    try:
        return run(s)
    except Exception as exc:                                   # pragma: no cover
        return {"seed": s, "error": f"{type(exc).__name__}: {exc}"}


def _med(v):
    return round(float(np.median(v)), 2) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(EVID / "hand_floor.json"))
    a = ap.parse_args()

    rows = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, range(a.seeds)):
            rows.append(r)
            if "error" in r:
                print(f"  ERROR seed {r['seed']}: {r['error']}")
    good = [r for r in rows if "waypoints" in r]

    per = {}
    for wp in WAYPOINTS + ("mug_descend",):
        got = [r["waypoints"][wp] for r in good if wp in r["waypoints"]]
        if not got:
            continue
        geoms: dict[str, int] = {}
        for g in got:
            geoms[g["lowest_geom"]] = geoms.get(g["lowest_geom"], 0) + 1
        per[wp] = {
            "n": len(got),
            "drop_mm_median": _med([g["drop_mm"] for g in got]),
            "drop_mm_min": min(g["drop_mm"] for g in got),
            "drop_mm_max": max(g["drop_mm"] for g in got),
            "tip_mid_z_m_median": _med([g["tip_mid_z_m"] for g in got]),
            "gripper_cmd_rad_median": _med([g["gripper_cmd_rad"] for g in got]),
            "lowest_geom_counts": dict(sorted(geoms.items(),
                                              key=lambda kv: -kv[1])),
        }

    drops = {k: v["drop_mm_median"] for k, v in per.items()}
    cut = [drops[k] for k in ("fork_descend", "spoon_descend") if k in drops]
    ctl = [drops[k] for k in ("mug_descend", "plate_descend") if k in drops]
    same = (bool(cut) and bool(ctl)
            and abs(float(np.median(cut)) - float(np.median(ctl))) <= 3.0)

    doc = {
        "probe": "measure_hand_floor.py",
        "question": "Is the cutlery stall a property of the HAND -- how far it "
                    "hangs below the point it pinches with -- rather than of "
                    "the trajectory, the opening or the solver?",
        "finding_id": "F-HAND-FLOOR-001",
        "method": "kinematic only: mj_kinematics on the joint vector plan_pose "
                  "returned. No dynamics, no servos, no contact solver.",
        "seeds": a.seeds,
        "controls": {
            "all_seeds_ok": len(good) == a.seeds,
            "cutlery_waypoints_present": all(
                w in per for w in ("fork_descend", "spoon_descend")),
            "control_waypoint_present": "mug_descend" in per,
            "drop_is_a_property_of_the_hand_not_the_waypoint": same,
            "why_that_control_matters":
                "mug_placed scores and fork_placed does not. If drop_mm "
                "differed between them this probe would be measuring posture, "
                "not the hand, and the finding would be void.",
        },
        "waypoints": per,
        "surfaces_seed0": good[0]["surfaces"] if good else None,
        "runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")

    print(f"\n{'waypoint':>16} {'n':>3} {'drop mm':>9} {'min':>7} {'max':>7} "
          f"{'tip z':>8}  lowest geom")
    for wp, v in per.items():
        print(f"{wp:>16} {v['n']:>3} {v['drop_mm_median']:>9} "
              f"{v['drop_mm_min']:>7} {v['drop_mm_max']:>7} "
              f"{v['tip_mid_z_m_median']:>8}  {v['lowest_geom_counts']}")
    print(f"\ncontrols: {doc['controls']['all_seeds_ok']=} "
          f"{doc['controls']['drop_is_a_property_of_the_hand_not_the_waypoint']=}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
