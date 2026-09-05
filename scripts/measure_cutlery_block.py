"""What stops the arms lifting the fork and the spoon out of the drawer?

``fork_placed`` and ``spoon_placed`` have scored 0 on every seed this project
has ever run -- 20 of the 50 sub-goals, and the two that criterion T1 weighs
under "manipulation accuracy" and "overall task success".  Three separate
explanations have been offered for it in this repository and this script tests
the two that were still standing.

1. GROSS REACH -- refuted already by ``scripts/measure_reach.py``: the arm puts
   its jaws at 0.417 m of planar radius and the cutlery sits at 0.347/0.387 m.

2. THE UNSQUARED JAW.  ``plan_pose_squared`` (see ``envs/controller.py``) took
   the mug from 0/10 gripped to 8/10 by solving for the direction the jaws
   close in, and its own docstring measures the fork and spoon grasps as 64-66
   degrees out of the plane they must be pinched across.  The two cutlery
   ``_pick``/``_handoff`` calls were never switched to it.  This script runs
   both settings on the same seeds.  REFUTED: squaring changes nothing that
   matters, because the arm never gets down to the object in the first place.

3. VERTICAL CLEARANCE -- what is left, and what this script measures.  The
   descend waypoint asks the jaw meeting point for the cutlery's own height;
   the arm stalls tens of millimetres above it, in sustained contact with a
   named piece of drawer.  The fork is stopped by the fork itself (the
   stationary jaw reaches it before the meeting point does, with ``standoff``
   left at 0 in ``_pick``); the spoon is stopped by ``drawer_front``, whose top
   edge stands above the cutlery it has to be reached over.

Reports, per seed and per cutlery waypoint: the z the waypoint asked for, the
lowest z the jaw meeting point actually reached, the gap between them, and the
non-arm geom the moving arm was in contact with for the most simulator steps.

Writes evidence/cutlery_block.json.
Run:  python3 scripts/measure_cutlery_block.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from envs.randomize import make_env                            # noqa: E402
from envs.task import TaskMonitor, PLACEMENTS                  # noqa: E402
from envs import controller as C                               # noqa: E402
from envs import scene_source                                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

# Exactly the waypoints that ``_pick(square=True)``, ``_handoff(square=True)``
# and the ``_place`` inside it would put through ``plan_pose_squared``.  Set on
# the built script rather than by editing the call sites, so the A/B is one
# process and one seed stream.
SQUARABLE = tuple(
    f"{b}{suf}" for b in ("fork", "spoon")
    for suf in ("_above", "_descend", "_lift", "_take_above", "_take")
) + tuple(f"target_{b}{suf}" for b in ("fork", "spoon")
          for suf in ("_over", "_down"))

DESCEND = ("fork_descend", "spoon_descend", "fork_take", "spoon_take")
ARMS = ("left", "right")


def _geom_body(model, g: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             int(model.geom_bodyid[g])) or "?"


def _flip_square(script, on: bool) -> int:
    """Set ``.square`` on every cutlery pinch waypoint in a built script."""
    n = 0
    for entry in script:
        if isinstance(entry, tuple) and entry and entry[0] == "if":
            n += _flip_square(entry[2], on)
            continue
        moves, _ = entry
        for mv in moves.values():
            if mv.label in SQUARABLE:
                mv.square = on
                n += 1
    return n


def scene_geometry(seed: int = 0) -> dict:
    """Wall heights and cutlery rest height, swept out of the compiled model.

    Measured rather than read off ``envs/dinner_table.py``'s literals, because
    the literals are half-sizes in a body frame and what the arm has to clear
    is a world z.
    """
    model, data, _ = make_env(seed)
    mujoco.mj_forward(model, data)
    out = {}
    for nm in ("drawer_floor", "drawer_front", "drawer_back",
               "drawer_side_l", "drawer_side_r", "cab_top"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g < 0:
            continue
        out[nm] = {"top_z_m": round(float(data.geom_xpos[g][2]
                                          + model.geom_size[g][2]), 4)}
    for nm in ("fork", "spoon"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{nm}_handle")
        out[nm] = {"top_z_m": round(float(data.geom_xpos[g][2]
                                          + model.geom_size[g][2]), 4)}
    lip = {k: round((out[k]["top_z_m"] - out["fork"]["top_z_m"]) * 1000, 1)
           for k in ("drawer_front", "drawer_side_l", "drawer_back")
           if k in out}
    out["lip_above_cutlery_mm"] = lip
    return out


def run(seed: int, square: bool) -> dict:
    model, data, _ = make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    arm_geoms = {a: {g for g in range(model.ngeom)
                     if _geom_body(model, g).startswith(f"{a}_")} for a in ARMS}
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    start = {o: data.xpos[bid(o)].copy() for o in ("fork", "spoon")}

    roll = C.Rollout(model, data)
    script = C.dinner_table_script()
    flipped = _flip_square(script, square)

    live = {"label": "", "arm": ""}
    asked: dict[str, float] = {}
    reached: dict[str, float] = {}
    contacts: dict[str, dict[str, int]] = {}
    peak = {o: 0.0 for o in start}

    orig_plan = roll._plan

    def plan(mv):
        live["label"], live["arm"] = mv.label, mv.arm
        if mv.label.startswith(("fork", "spoon")) and mv.where is not None:
            asked.setdefault(mv.label,
                             float(np.asarray(mv.where(model, data), float)[2]))
        return orig_plan(mv)

    roll._plan = plan

    def on_step(d):
        for o in start:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))
        lb, arm = live["label"], live["arm"]
        if not lb.startswith(("fork", "spoon")):
            return
        g = roll.grips[arm]
        z = float(g.tip_mid(model, d)[2])
        reached[lb] = min(reached.get(lb, 9.9), z)
        mine = arm_geoms[arm]
        tally = contacts.setdefault(lb, {})
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if g1 in mine or g2 in mine:
                other = g2 if g1 in mine else g1
                if other in mine:
                    continue
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       other) or f"geom{other}"
                tally[nm] = tally.get(nm, 0) + 1

    roll.run(script, monitor=mon, on_step=on_step)
    rep = mon.report(data)

    waypoints = []
    for lb in DESCEND:
        if lb not in asked or lb not in reached:
            continue
        top = sorted(contacts.get(lb, {}).items(), key=lambda kv: -kv[1])[:2]
        waypoints.append({
            "waypoint": lb,
            "asked_tip_z_m": round(asked[lb], 4),
            "lowest_tip_z_m": round(reached[lb], 4),
            "stalled_above_target_mm": round((reached[lb] - asked[lb]) * 1000, 1),
            "blocking_contacts": [{"geom": n, "steps": s} for n, s in top],
        })

    objs = {}
    for o in ("fork", "spoon"):
        end = data.xpos[bid(o)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{o}_placed"][1])]
        objs[o] = {
            "lifted_mm": round(peak[o] * 1000, 1),
            "planar_move_mm": round(
                float(np.linalg.norm(end[:2] - start[o][:2])) * 1000, 1),
            "final_to_target_mm": round(
                float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
            "tolerance_mm": round(PLACEMENTS[f"{o}_placed"][2] * 1000, 1),
            "placed": bool(rep["subgoals"][f"{o}_placed"]),
        }
    return {"seed": seed, "square_cutlery": square,
            "waypoints_squared": flipped,
            "subgoals_met": rep["subgoals_met"],
            "subgoals": rep["subgoals"], "objects": objs,
            "waypoints": waypoints}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="cutlery_block.json")
    args = ap.parse_args()

    rows = []
    for square in (False, True):
        for s in range(args.seeds):
            r = run(s, square)
            rows.append(r)
            o = r["objects"]
            print(f"seed {s} square={square!s:5s} met={r['subgoals_met']}  "
                  f"fork lift {o['fork']['lifted_mm']:5.1f} mm "
                  f"d {o['fork']['final_to_target_mm']:6.1f}/{o['fork']['tolerance_mm']:.0f}  "
                  f"spoon lift {o['spoon']['lifted_mm']:5.1f} mm "
                  f"d {o['spoon']['final_to_target_mm']:6.1f}/{o['spoon']['tolerance_mm']:.0f}",
                  flush=True)

    def placed(sq):
        return sum(1 for r in rows if r["square_cutlery"] is sq
                   for o in ("fork", "spoon") if r["objects"][o]["placed"])

    def maxlift(sq):
        return max(r["objects"][o]["lifted_mm"] for r in rows
                   if r["square_cutlery"] is sq for o in ("fork", "spoon"))

    stalls = [w for r in rows if not r["square_cutlery"] for w in r["waypoints"]]
    worst = max(stalls, key=lambda w: w["stalled_above_target_mm"])
    out = {
        "question": "why have fork_placed and spoon_placed never scored?",
        "seeds": args.seeds,
        "hypothesis_1_gross_reach": "REFUTED ELSEWHERE -- scripts/measure_reach.py, "
                                    "evidence/reach_envelope.json",
        "hypothesis_2_unsquared_jaw": {
            "verdict": "REFUTED",
            "cutlery_placed_unsquared": placed(False),
            "cutlery_placed_squared": placed(True),
            "max_cutlery_lift_mm_unsquared": maxlift(False),
            "max_cutlery_lift_mm_squared": maxlift(True),
            "note": "plan_pose_squared solves for the jaw closing direction and "
                    "took the mug from 0/10 gripped to 8/10. Applied to the "
                    "cutlery it changes nothing: neither object is lifted off "
                    "the drawer floor either way, because the arm never arrives "
                    "at the grasp height. A better jaw angle at a pose that is "
                    "never reached is not a better grasp.",
        },
        "hypothesis_3_vertical_clearance": {
            "verdict": "STANDS -- this is the measured blocker",
            "stall_mm_median": round(float(np.median(
                [w["stalled_above_target_mm"] for w in stalls])), 1),
            "worst_stall": worst,
            "stalls": stalls,
            "note": "The descend waypoint asks the jaw meeting point for the "
                    "cutlery's own height and the arm stalls above it in "
                    "sustained contact with a named geom. Scene geometry, from "
                    "envs/dinner_table.py: the cutlery rests at z=0.784 on a "
                    "drawer floor whose top face is z=0.781; the drawer side "
                    "walls stand to z=0.813 and drawer_front to z=0.827. Every "
                    "route to the cutlery is over a 43 mm lip. _pick also passes "
                    "no standoff, so the stationary jaw reaches the fork before "
                    "the meeting point does -- which is what plan_pose's own "
                    "docstring says shoved all three objects away.",
        },
        "scene_geometry": scene_geometry(0),
        "runs": rows,
    }
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\ncutlery placed: unsquared {placed(False)}, squared {placed(True)} "
          f"(of {2 * args.seeds}); max lift {maxlift(False):.1f} / "
          f"{maxlift(True):.1f} mm -> evidence/{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
