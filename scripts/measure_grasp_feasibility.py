"""Can the gripper reach the cutlery inside the drawer at all, in PLAN?

Every explanation offered so far for ``fork_placed``/``spoon_placed`` being
0/10 has been a control question -- a jaw angle, a standoff, a reach envelope.
This script asks the prior question and answers it with geometry rather than
with a rollout: put the arm in the pose the planner ASKS for, and see whether
that pose is inside the drawer.

Method.  Run the real script until the cutlery descend waypoints are planned,
and snapshot the live state at each -- the drawer where the arm actually left
it, the cutlery where the drawer actually dragged it.  Then, from that
snapshot, plan the grasp pose at a range of heights above the object, teleport
the arm onto each planned pose, and run MuJoCo's own collision pass.  No
dynamics, no servos, no saturation: if the SOLVED pose already has the wrist
inside ``drawer_front``, then no gain, no standoff and no jaw angle can help,
because the controller is chasing a pose that does not fit.

Reports the lowest height above the object at which the planned pose is free
of the woodwork, and what is in the way below it.  Compare that number with
the object's own half-thickness: a pinch needs the jaws AROUND the handle, so
a clearance floor above the handle top is a floor on the grasp, not a margin.

Writes evidence/grasp_feasibility.json.
Run:  python3 scripts/measure_grasp_feasibility.py --seeds 3
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

DESCEND = ("fork_descend", "spoon_descend")
WOOD = ("drawer_", "cab_")
PEN_MM = 0.5          # ignore grazing contacts; a plan is "blocked" when a
                      # geom is at least this far INSIDE another


def _name(model, g: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{g}"


def _lowest_z(model, d, g: int) -> float:
    """Bottom of geom ``g``'s world axis-aligned box.

    Computed from the geom's own rotation rather than from its largest
    half-size: the jaws are boxes and they are not axis-aligned in a grasp
    pose, so a max-radius approximation reports them tens of millimetres
    lower than they are and would make the overlap below look better than it
    is.  Exact for a box, conservative (an over-estimate of the extent, so an
    under-estimate of the overlap) for the capsules.
    """
    R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
    half = np.asarray(model.geom_size[g], float)
    return float(d.geom_xpos[g][2] - float(np.abs(R[2]) @ half))


def _jaw_geoms(model, arm: str) -> set[int]:
    """The gripper's jaw boxes -- the parts that must end up either side of
    the handle.  Asserted non-empty: an empty set makes every overlap below
    report as a vacuous pass rather than as a broken probe."""
    out = {g for g in range(model.ngeom)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) or ""
               ).startswith(f"{arm}_") and "jaw" in _name(model, g)}
    if not out:
        raise RuntimeError(f"no jaw geoms found for arm {arm!r}")
    return out


def run(seed: int, dz_max_mm: float, step_mm: float) -> dict:
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
        if mv.label in DESCEND and mv.label not in snaps:
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

    # --- the geometric sweep, from each snapshot ---------------------------
    probe = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    out = {}
    for label, s in snaps.items():
        grip = roll.grips[s["arm"]]
        mine = arm_geoms[s["arm"]]
        # the object this waypoint pinches, so its own contacts are separable
        body = label.split("_")[0]
        jg = _jaw_geoms(model, s["arm"])
        hg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_handle")
        levels = []
        n = int(round(dz_max_mm / step_mm)) + 1
        for i in range(n):
            dz = i * step_mm / 1000.0
            tgt = s["target"] + np.array([0.0, 0.0, dz])
            scratch.qpos[:] = s["qpos"]
            scratch.qvel[:] = 0.0
            solver = (C.plan_pose_squared if (s["square"] and s["jaw"] is not None)
                      else C.plan_pose)
            q, err = solver(model, scratch, grip, tgt, jaw_dir=s["jaw"],
                            opening=s["plan_at"], standoff=0.0)
            probe.qpos[:] = s["qpos"]
            probe.qvel[:] = 0.0
            probe.qpos[grip.qadr] = q
            mujoco.mj_forward(model, probe)
            wood, own, other = [], [], []
            for c in range(probe.ncon):
                con = probe.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                if (g1 in mine) == (g2 in mine):
                    continue                      # not an arm-vs-world contact
                o = g2 if g1 in mine else g1
                depth = -float(con.dist) * 1000.0
                if depth < PEN_MM:
                    continue
                nm = _name(model, o)
                rec = {"geom": nm, "penetration_mm": round(depth, 2)}
                if nm.startswith(WOOD):
                    wood.append(rec)
                elif nm.startswith(body):
                    own.append(rec)
                else:
                    other.append(rec)
            h_top = float(probe.geom_xpos[hg][2] + model.geom_size[hg][2])
            h_bot = float(probe.geom_xpos[hg][2] - model.geom_size[hg][2])
            jaw_bot = min(_lowest_z(model, probe, g) for g in jg)
            levels.append({
                "dz_mm": round(dz * 1000, 1),
                "ik_err_mm": round(err * 1000, 2),
                "handle_thickness_mm": round((h_top - h_bot) * 1000, 1),
                "jaw_below_handle_top_mm": round((h_top - jaw_bot) * 1000, 1),
                "woodwork_hits": sorted(wood, key=lambda r: -r["penetration_mm"])[:3],
                "own_object_hits": sorted(own, key=lambda r: -r["penetration_mm"])[:2],
                "other_hits": sorted(other, key=lambda r: -r["penetration_mm"])[:2],
            })
        clear = [l for l in levels if not l["woodwork_hits"]]
        out[label] = {
            "arm": s["arm"],
            "asked_z_m": round(float(s["target"][2]), 4),
            "min_clear_dz_mm": clear[0]["dz_mm"] if clear else None,
            "handle_thickness_mm": levels[0]["handle_thickness_mm"],
            # how far the jaws reach past the top face of the handle at the
            # lowest pose that clears the woodwork.  Positive means the jaws
            # are alongside the handle and a pinch there closes ON it; this is
            # the control that separates "the pose is unreachable" from "the
            # pose is reachable but grips nothing".
            "jaw_below_handle_top_mm_at_min_clear": (
                clear[0]["jaw_below_handle_top_mm"] if clear else None),
            "blocked_all_the_way": not clear,
            "levels": levels,
        }
    return {"seed": seed, "waypoints": out}


def cross(seed: int, dz_max_mm: float, step_mm: float) -> dict:
    """The same sweep, for every (arm, object) pair rather than the scripted one.

    ``dinner_table_script`` gives the fork to the RIGHT arm and the spoon to the
    LEFT.  The per-waypoint sweep shows those two are not the same problem --
    the fork clears by millimetres and the spoon by tens of them -- so the
    question is whether that is a property of the OBJECT (where it sits in the
    drawer) or of the ARM that was given it.  Only a cross separates them.

    Planned from one shared state: the drawer where the script left it and the
    cutlery where the drawer dragged it, so all four pairs see one scene.
    """
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
    snap = {}
    orig_plan = roll._plan

    def plan(mv):
        if mv.label == "fork_descend" and not snap:
            snap["qpos"] = data.qpos.copy()
        return orig_plan(mv)

    roll._plan = plan
    roll.run(script, monitor=mon)
    if not snap:
        return {"seed": seed, "error": "fork_descend never planned"}

    probe = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    ref = mujoco.MjData(model)
    ref.qpos[:] = snap["qpos"]
    ref.qvel[:] = 0.0
    mujoco.mj_forward(model, ref)

    pairs = {}
    n = int(round(dz_max_mm / step_mm)) + 1
    for arm in ("left", "right"):
        grip = roll.grips[arm]
        mine = arm_geoms[arm]
        for obj in ("fork", "spoon"):
            base = C.site_xyz(f"{obj}_grasp", dz=0.003)(model, ref)
            jaw = C.across(obj)(model, ref)
            levels = []
            for i in range(n):
                dz = i * step_mm / 1000.0
                tgt = np.asarray(base, float) + np.array([0.0, 0.0, dz])
                scratch.qpos[:] = snap["qpos"]
                scratch.qvel[:] = 0.0
                q, err = C.plan_pose(model, scratch, grip, tgt,
                                     jaw_dir=np.asarray(jaw, float),
                                     opening=C.GRIP_PINCH, standoff=0.0)
                probe.qpos[:] = snap["qpos"]
                probe.qvel[:] = 0.0
                probe.qpos[grip.qadr] = q
                mujoco.mj_forward(model, probe)
                wood = []
                for c in range(probe.ncon):
                    con = probe.contact[c]
                    g1, g2 = int(con.geom1), int(con.geom2)
                    if (g1 in mine) == (g2 in mine):
                        continue
                    o = g2 if g1 in mine else g1
                    depth = -float(con.dist) * 1000.0
                    if depth < PEN_MM:
                        continue
                    nm = _name(model, o)
                    if nm.startswith(WOOD):
                        wood.append({"geom": nm,
                                     "penetration_mm": round(depth, 2)})
                levels.append({"dz_mm": round(dz * 1000, 1),
                               "ik_err_mm": round(err * 1000, 2),
                               "woodwork_hits": sorted(
                                   wood, key=lambda r: -r["penetration_mm"])[:3]})
            clear = [l for l in levels if not l["woodwork_hits"]]
            pairs[f"{arm}:{obj}"] = {
                "min_clear_dz_mm": clear[0]["dz_mm"] if clear else None,
                "scripted": (arm, obj) in (("right", "fork"), ("left", "spoon")),
                "blockers_at_dz0": [h["geom"] for h in levels[0]["woodwork_hits"]],
                "ik_err_mm_at_min_clear": (clear[0]["ik_err_mm"]
                                           if clear else None),
            }
    return {"seed": seed, "pairs": pairs}


def site_sweep(seed: int, dz_max_mm: float, step_mm: float) -> dict:
    """Where in this drawer IS a spoon graspable?

    The cross shows the spoon is refused by both arms and the fork by neither,
    so the variable is where the two objects sit, not which arm is sent.  The
    scene parks the spoon at drawer-local y = -0.032, which is 36 mm behind
    ``drawer_front`` -- the tallest of the four walls, standing 40.5 mm above
    the cutlery it has to be reached over.  The fork sits at +0.018, 86 mm
    behind it.

    This slides the spoon along the drawer's own axis and re-solves the grasp
    at each station, so the next fix is aimed at a measured number rather than
    at a guess.  Nothing here changes the scene on disk: the body is teleported
    inside a probe, and ``envs/dinner_table.py`` is untouched.
    """
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
    snap = {}
    orig_plan = roll._plan

    def plan(mv):
        if mv.label == "fork_descend" and not snap:
            snap["qpos"] = data.qpos.copy()
        return orig_plan(mv)

    roll._plan = plan
    roll.run(script, monitor=mon)
    if not snap:
        return {"seed": seed, "error": "fork_descend never planned"}

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    qadr = int(model.jnt_qposadr[int(model.body_jntadr[bid])])
    fork_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork")

    probe = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    ref = mujoco.MjData(model)
    ref.qpos[:] = snap["qpos"]
    ref.qvel[:] = 0.0
    mujoco.mj_forward(model, ref)
    y0 = float(ref.qpos[qadr + 1])
    fork_y = float(ref.xpos[fork_bid][1])

    n = int(round(dz_max_mm / step_mm)) + 1
    stations = []
    for k in range(-8, 13):
        dy = k * 0.005
        row = {"dy_mm": round(dy * 1000, 1),
               "world_y_m": round(y0 + dy, 4),
               "clearance_to_fork_mm": round(abs((y0 + dy) - fork_y) * 1000, 1)}
        for arm in ("left", "right"):
            grip = roll.grips[arm]
            mine = arm_geoms[arm]
            best = None
            for i in range(n):
                dz = i * step_mm / 1000.0
                scratch.qpos[:] = snap["qpos"]
                scratch.qpos[qadr + 1] = y0 + dy
                scratch.qvel[:] = 0.0
                mujoco.mj_forward(model, scratch)
                base = C.site_xyz("spoon_grasp", dz=0.003)(model, scratch)
                jaw = C.across("spoon")(model, scratch)
                tgt = np.asarray(base, float) + np.array([0.0, 0.0, dz])
                q, err = C.plan_pose(model, scratch, grip, tgt,
                                     jaw_dir=np.asarray(jaw, float),
                                     opening=C.GRIP_PINCH, standoff=0.0)
                probe.qpos[:] = snap["qpos"]
                probe.qpos[qadr + 1] = y0 + dy
                probe.qvel[:] = 0.0
                probe.qpos[grip.qadr] = q
                mujoco.mj_forward(model, probe)
                hit = False
                for c in range(probe.ncon):
                    con = probe.contact[c]
                    g1, g2 = int(con.geom1), int(con.geom2)
                    if (g1 in mine) == (g2 in mine):
                        continue
                    o = g2 if g1 in mine else g1
                    if -float(con.dist) * 1000.0 < PEN_MM:
                        continue
                    if _name(model, o).startswith(WOOD):
                        hit = True
                        break
                if not hit:
                    best = round(dz * 1000, 1)
                    break
            row[arm] = best
        stations.append(row)
    return {"seed": seed, "spoon_world_y_m": round(y0, 4),
            "fork_world_y_m": round(fork_y, 4), "stations": stations}


def _job_site(args):
    seed, dz, step = args
    try:
        return site_sweep(seed, dz, step)
    except Exception as exc:
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def _job(args):
    seed, dz, step = args
    try:
        return run(seed, dz, step)
    except Exception as exc:
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def _job_cross(args):
    seed, dz, step = args
    try:
        return cross(seed, dz, step)
    except Exception as exc:
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dz-max-mm", type=float, default=60.0)
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--site-sweep", action="store_true",
                    help="slide the spoon along the drawer axis and re-solve, "
                         "to find where in the drawer a grasp is feasible")
    ap.add_argument("--cross", action="store_true",
                    help="also sweep every (arm, object) pair, not just the "
                         "pair dinner_table_script() happens to use")
    ap.add_argument("--out", default=str(EVID / "grasp_feasibility.json"))
    a = ap.parse_args()

    jobs = [(s, a.dz_max_mm, a.step_mm) for s in range(a.seeds)]
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(_job, jobs))
        crosses = list(ex.map(_job_cross, jobs)) if a.cross else []
        sites = list(ex.map(_job_site, jobs)) if a.site_sweep else []

    summary = {}
    for lb in DESCEND:
        ws = [r["waypoints"][lb] for r in runs
              if "error" not in r and lb in r["waypoints"]]
        vals = [w["min_clear_dz_mm"] for w in ws]
        got = [v for v in vals if v is not None]
        ov = [w["jaw_below_handle_top_mm_at_min_clear"] for w in ws
              if w["jaw_below_handle_top_mm_at_min_clear"] is not None]
        summary[lb] = {
            "seeds": len(vals),
            "min_clear_dz_mm": got,
            "blocked_seeds": sum(1 for v in vals if v is None),
            "median_min_clear_dz_mm": (round(float(np.median(got)), 1)
                                       if got else None),
            "handle_thickness_mm": (ws[0]["handle_thickness_mm"] if ws
                                    else None),
            "jaw_below_handle_top_mm_at_min_clear": ov,
            "grips_at_min_clear": bool(ov) and all(v > 0 for v in ov),
        }
    cross_summary = {}
    if crosses:
        keys = sorted({k for r in crosses if "pairs" in r for k in r["pairs"]})
        for k in keys:
            vals = [r["pairs"][k]["min_clear_dz_mm"] for r in crosses
                    if "pairs" in r]
            got = [v for v in vals if v is not None]
            cross_summary[k] = {
                "scripted": next(r["pairs"][k]["scripted"] for r in crosses
                                 if "pairs" in r),
                "min_clear_dz_mm": vals,
                "median_min_clear_dz_mm": (round(float(np.median(got)), 1)
                                           if got else None),
                "blocked_seeds": sum(1 for v in vals if v is None),
            }
    out = {
        "question": "is the cutlery grasp pose collision-free in PLAN, before "
                    "any dynamics?",
        "cross_summary": cross_summary,
        "crosses": crosses,
        "site_sweep": sites,
        "method": "solve the grasp pose at dz above the object, teleport the "
                  "arm onto it, run MuJoCo's collision pass; report the lowest "
                  "dz with no woodwork penetration >= %.1f mm" % PEN_MM,
        "penetration_threshold_mm": PEN_MM,
        "summary": summary,
        "runs": runs,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    if sites:
        print("  --- spoon slid along the drawer axis: min clear dz (mm) ---")
        ok = [r for r in sites if "stations" in r]
        if ok:
            for i, st in enumerate(ok[0]["stations"]):
                cells = []
                for arm in ("left", "right"):
                    vs = [r["stations"][i][arm] for r in ok]
                    cells.append(f"{arm}={vs}")
                print(f"  dy={st['dy_mm']:+7.1f}mm "
                      f"fork_gap={st['clearance_to_fork_mm']:5.1f}mm  "
                      + "  ".join(cells))
    if cross_summary:
        print("  --- arm x object, min clear dz (mm) ---")
        for k, v in cross_summary.items():
            print(f"  {k:14} scripted={int(v['scripted'])} "
                  f"median={v['median_min_clear_dz_mm']} "
                  f"per_seed={v['min_clear_dz_mm']}")
    for r in runs:
        if "error" in r:
            print(f"  seed {r['seed']}: ERROR {r['error']}")
            continue
        for lb, w in r["waypoints"].items():
            lo = w["levels"][0]
            print(f"  seed {r['seed']} {lb:15} arm={w['arm']:5} "
                  f"min_clear_dz={w['min_clear_dz_mm']}mm  "
                  f"at dz=0: wood={[h['geom'] for h in lo['woodwork_hits']]} "
                  f"own={[h['geom'] for h in lo['own_object_hits']]}")


if __name__ == "__main__":
    main()
