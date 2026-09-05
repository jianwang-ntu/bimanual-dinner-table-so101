"""Does a non-zero grasp ``standoff`` get the cutlery out of the drawer?

``scripts/measure_cutlery_block.py`` established what stops the fork and the
spoon: the arm never arrives at the grasp height.  It stalls a median 43.0 mm
above the waypoint it asked for, in sustained contact with a named geom.  Two
different geoms, and they are two different problems:

* ``fork_descend`` stalls 21-24 mm high on ``fork_handle`` -- the fork itself.
  ``plan_pose``'s docstring measures why: only one of this gripper's jaws
  moves, so a pose that puts the jaw MEETING POINT on the object puts the
  STATIONARY jaw inside it -- 4.9 mm from the fork handle's centre against a
  6 mm half-width.  ``_pick`` passes no ``standoff`` at all, and the plate is
  the only grasp in the script that does (``PLATE_STANDOFF``).
* ``spoon_descend`` and both ``_take`` waypoints stall on ``drawer_front`` and
  ``drawer_side_l`` -- the woodwork, which stands 40.5 mm and 26.5 mm above
  the objects being reached over.

A standoff can only ever fix the first kind.  This script measures how much of
the problem that is, rather than assuming either way, by sweeping the standoff
over the four cutlery pinch waypoints on a shared seed stream.  The sweep is
crossed with ``square``, because ``measure_cutlery_block`` refuted squaring at
standoff 0 and "refuted at the only setting tried" is not the same claim as
"refuted".

Every variant runs the SAME seeds and the same untouched scorer, and the
baseline (standoff 0, square off) is in the sweep rather than quoted from the
earlier file, so the comparison is within one process pool and one code state.

Writes evidence/cutlery_standoff.json.
Run:  python3 scripts/measure_cutlery_standoff.py --seeds 3
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

# The waypoints that actually pinch the cutlery: the descend that closes the
# jaws in the drawer, and the taker's grasp at the hand-off.  The ``_above``
# via points are deliberately excluded -- they are solved with the jaws OPEN
# and nothing closes there, so a standoff on them only moves a free-space pose.
PINCH = ("fork_descend", "spoon_descend", "fork_take", "spoon_take")
DESCEND = ("fork_descend", "spoon_descend")
# Exactly the set ``measure_cutlery_block.py`` flipped when it recorded the
# squaring hypothesis REFUTED.  It includes the carry waypoints (``_lift``,
# ``_taker_lift``) and the two ``_place`` waypoints, and ``_handoff``'s own
# docstring already measures that squaring a CARRY asks for a wrist a loaded
# arm cannot hold.  Kept here as a named scope so the earlier result is
# reproduced in this sweep rather than argued with.
ALL_SQUARABLE = tuple(
    f"{b}{suf}" for b in ("fork", "spoon")
    for suf in ("_above", "_descend", "_lift", "_take_above", "_take")
) + tuple(f"target_{b}{suf}" for b in ("fork", "spoon")
          for suf in ("_over", "_down"))

SCOPES = {
    "none": (),
    "descend": DESCEND,
    "pinch": PINCH,
    "all": ALL_SQUARABLE,
}
ARMS = ("left", "right")


def _apply(script, standoff: float, scope: str) -> tuple[int, int]:
    """Square the waypoints ``scope`` names; put ``standoff`` on the descents.

    The two axes are deliberately separate.  Squaring solves for the direction
    the jaws close in; the standoff decides where the meeting point sits along
    that direction.  Only the descents pinch inside the drawer, so only they
    get the standoff -- a standoff on a free-space via point just moves a pose
    nothing closes at.
    """
    sq = set(SCOPES[scope])
    n_sq = n_off = 0
    stack = [script]
    while stack:
        cur = stack.pop()
        for entry in cur:
            if isinstance(entry, tuple) and entry and entry[0] == "if":
                stack.append(entry[2])
                continue
            moves, _ = entry
            for mv in moves.values():
                lb = getattr(mv, "label", "")
                if lb in sq:
                    mv.square = True
                    n_sq += 1
                if lb in DESCEND and standoff:
                    mv.standoff = standoff
                    n_off += 1
    return n_sq, n_off


def run(seed: int, standoff: float, scope: str) -> dict:
    from envs.randomize import make_env
    from envs.task import TaskMonitor, PLACEMENTS
    from envs import controller as C
    from envs import scene_source

    model, data, _ = make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    def geom_body(g):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) or "?"

    arm_geoms = {a: {g for g in range(model.ngeom)
                     if geom_body(g).startswith(f"{a}_")} for a in ARMS}
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    start = {o: data.xpos[bid(o)].copy() for o in ("fork", "spoon")}

    roll = C.Rollout(model, data)
    script = C.dinner_table_script()
    n_sq, n_off = _apply(script, standoff, scope)

    live = {"label": "", "arm": ""}
    asked: dict[str, float] = {}
    reached: dict[str, float] = {}
    contacts: dict[str, dict[str, int]] = {}
    peak = {o: 0.0 for o in start}
    orig_plan = roll._plan

    def plan(mv):
        live["label"], live["arm"] = mv.label, mv.arm
        if mv.label in PINCH and mv.where is not None:
            asked.setdefault(mv.label,
                             float(np.asarray(mv.where(model, data), float)[2]))
        return orig_plan(mv)

    roll._plan = plan

    def on_step(d):
        for o in start:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))
        lb, arm = live["label"], live["arm"]
        if lb not in PINCH:
            return
        g = roll.grips[arm]
        reached[lb] = min(reached.get(lb, 9.9), float(g.tip_mid(model, d)[2]))
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
    for lb in PINCH:
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
            "placed": bool(rep["subgoals"][f"{o}_placed"]),
        }
    return {
        "seed": seed, "standoff_m": standoff, "scope": scope,
        "squared_waypoints": n_sq, "standoff_waypoints": n_off,
        "subgoals_met": int(sum(rep["subgoals"].values())),
        "subgoals": rep["subgoals"],
        "objects": objs, "waypoints": waypoints,
    }


def _job(args):
    seed, standoff, scope = args
    try:
        return run(seed, standoff, scope)
    except Exception as exc:                       # keep the sweep going
        return {"seed": seed, "standoff_m": standoff, "scope": scope,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--standoffs", default="0.0,0.006,0.010,0.014")
    ap.add_argument("--scopes", default="none,descend,pinch,all")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(EVID / "cutlery_standoff.json"))
    a = ap.parse_args()

    offs = [float(x) for x in a.standoffs.split(",")]
    sqs = [x.strip() for x in a.scopes.split(",")]
    for q in sqs:
        if q not in SCOPES:
            raise SystemExit(f"unknown scope {q!r}; have {sorted(SCOPES)}")
    jobs = [(s, o, q) for o in offs for q in sqs for s in range(a.seeds)]

    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(_job, jobs))

    variants = []
    for o in offs:
        for q in sqs:
            rs = [r for r in runs if r["standoff_m"] == o and r["scope"] == q]
            ok = [r for r in rs if "error" not in r]
            placed = sum(int(r["objects"][b]["placed"]) for r in ok
                         for b in ("fork", "spoon"))
            lift = max([r["objects"][b]["lifted_mm"] for r in ok
                        for b in ("fork", "spoon")] or [0.0])
            stalls = [w["stalled_above_target_mm"] for r in ok
                      for w in r["waypoints"]]
            variants.append({
                "standoff_m": o, "scope": q, "seeds": len(ok),
                "cutlery_placed": placed,
                "cutlery_placed_of": 2 * len(ok),
                "max_cutlery_lift_mm": round(lift, 1),
                "stall_mm_median": (round(float(np.median(stalls)), 1)
                                    if stalls else None),
                "stall_mm_min": round(min(stalls), 1) if stalls else None,
                "subgoals_total": sum(r["subgoals_met"] for r in ok),
                "subgoals_of": 5 * len(ok),
                "errors": [r["error"] for r in rs if "error" in r],
            })

    base = next(v for v in variants
                if v["standoff_m"] == 0.0 and v["scope"] == "none")
    best = max(variants, key=lambda v: (v["cutlery_placed"],
                                        v["max_cutlery_lift_mm"]))
    out = {
        "question": "does a non-zero grasp standoff lift the cutlery out of "
                    "the drawer?",
        "seeds": a.seeds,
        "pinch_waypoints": list(PINCH),
        "scopes": {k: list(v) for k, v in SCOPES.items()},
        "baseline": {k: base[k] for k in
                     ("standoff_m", "scope", "cutlery_placed",
                      "max_cutlery_lift_mm", "stall_mm_median",
                      "subgoals_total")},
        "best": {k: best[k] for k in
                 ("standoff_m", "scope", "cutlery_placed",
                  "max_cutlery_lift_mm", "stall_mm_median", "subgoals_total")},
        "verdict": ("STANDOFF HELPS" if best["cutlery_placed"] >
                    base["cutlery_placed"] else
                    "NO PLACEMENT GAIN -- see max_cutlery_lift_mm and stalls"),
        "variants": variants,
        "runs": runs,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("baseline", "best", "verdict")}, indent=1))
    for v in variants:
        print(f"  standoff={v['standoff_m']:.3f} scope={v['scope']:8} "
              f"placed={v['cutlery_placed']}/{v['cutlery_placed_of']} "
              f"lift={v['max_cutlery_lift_mm']:6.1f}mm "
              f"stall_med={v['stall_mm_median']} "
              f"subgoals={v['subgoals_total']}/{v['subgoals_of']}"
              + (f" ERRORS={len(v['errors'])}" if v["errors"] else ""))


if __name__ == "__main__":
    main()
