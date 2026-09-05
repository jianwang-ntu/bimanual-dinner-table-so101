"""Does a FEASIBLE spoon pose become a placed spoon?

``scripts/measure_grasp_feasibility.py`` shows the spoon's grasp pose is inside
``drawer_front`` for both arms at the position the scene authors it -- median
40 mm (left) and 30 mm (right) of height had to be given up before the solved
pose cleared the woodwork -- and that sliding the spoon 20 mm deeper into the
drawer collapses that to 6-8 mm, which is what the fork already enjoys.

That is a statement about geometry.  It is not yet a statement about the task:
a reachable pose still has to survive the servos, the contacts and the rest of
the episode.  This script closes that gap by running the REAL rollout and the
untouched scorer with the spoon started at a range of depths.

The spoon is teleported at t=0, before the drawer is opened, which is the same
state the scene would produce if ``envs/dinner_table.py`` had authored the
offset -- the drawer is still shut, and the drawer floor drags the spoon out
exactly as it drags it today.  Nothing on disk is edited, so a negative result
costs no artifact re-derivation.

dy = 0.0 is the shipped scene and is run in the same pool as every other
station, so the comparison is within one code state rather than against a
quoted number.

Writes evidence/spoon_depth.json.
Run:  python3 scripts/measure_spoon_depth.py --seeds 10 --dys 0,0.010,0.020,0.030
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


def run(seed: int, dy: float) -> dict:
    from envs.randomize import make_env
    from envs.task import TaskMonitor, PLACEMENTS
    from envs import controller as C
    from envs import scene_source

    model, data, _ = make_env(seed)
    scene_source.install(scene_source.make("privileged"))
    mon = TaskMonitor(model)

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    if dy:
        adr = int(model.jnt_qposadr[int(model.body_jntadr[bid("spoon")])])
        data.qpos[adr + 1] += dy
        mujoco.mj_forward(model, data)
    start = {o: data.xpos[bid(o)].copy() for o in ("fork", "spoon")}
    peak = {o: 0.0 for o in start}

    roll = C.Rollout(model, data)

    def on_step(d):
        for o in start:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    objs = {}
    for o in ("fork", "spoon"):
        end = data.xpos[bid(o)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{o}_placed"][1])]
        objs[o] = {
            "lifted_mm": round(peak[o] * 1000, 1),
            "final_to_target_mm": round(
                float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
            "tolerance_mm": round(PLACEMENTS[f"{o}_placed"][2] * 1000, 1),
            "placed": bool(rep["subgoals"][f"{o}_placed"]),
        }
    return {"seed": seed, "dy_m": dy, "subgoals": rep["subgoals"],
            "subgoals_met": int(sum(rep["subgoals"].values())), "objects": objs}


def _job(args):
    seed, dy = args
    try:
        return run(seed, dy)
    except Exception as exc:
        return {"seed": seed, "dy_m": dy,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dys", default="0,0.010,0.020,0.030")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=str(EVID / "spoon_depth.json"))
    a = ap.parse_args()

    dys = [float(x) for x in a.dys.split(",")]
    jobs = [(s, d) for d in dys for s in range(a.seeds)]
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(_job, jobs))

    stations = []
    for d in dys:
        ok = [r for r in runs if r["dy_m"] == d and "error" not in r]
        sub = {k: sum(int(r["subgoals"][k]) for r in ok)
               for k in ("drawer_open", "fork_placed", "spoon_placed",
                         "plate_placed", "mug_placed")}
        stations.append({
            "dy_m": d, "seeds": len(ok),
            "subgoals_total": sum(r["subgoals_met"] for r in ok),
            "subgoals_of": 5 * len(ok),
            "per_subgoal": sub,
            "spoon_lift_mm_max": round(max(
                [r["objects"]["spoon"]["lifted_mm"] for r in ok] or [0]), 1),
            "spoon_to_target_mm_min": round(min(
                [r["objects"]["spoon"]["final_to_target_mm"] for r in ok]
                or [0]), 1),
            "errors": [r["error"] for r in runs
                       if r["dy_m"] == d and "error" in r],
        })

    base = next(s for s in stations if s["dy_m"] == 0.0)
    best = max(stations, key=lambda s: (s["per_subgoal"]["spoon_placed"],
                                        s["subgoals_total"]))
    out = {
        "question": "does making the spoon's grasp pose feasible make the "
                    "spoon get placed?",
        "note": "the spoon is teleported at t=0 with the drawer still shut; "
                "envs/dinner_table.py is NOT edited by this script",
        "seeds": a.seeds, "baseline_dy_m": 0.0,
        "baseline": base, "best": best,
        "verdict": ("SPOON PLACED" if best["per_subgoal"]["spoon_placed"] >
                    base["per_subgoal"]["spoon_placed"] else
                    "NO PLACEMENT GAIN"),
        "stations": stations, "runs": runs,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(out["verdict"])
    for s in stations:
        p = s["per_subgoal"]
        print(f"  dy={s['dy_m']*1000:+6.1f}mm  subgoals={s['subgoals_total']:3}/"
              f"{s['subgoals_of']:<3} drawer={p['drawer_open']} "
              f"fork={p['fork_placed']} spoon={p['spoon_placed']} "
              f"plate={p['plate_placed']} mug={p['mug_placed']}  "
              f"spoon_lift_max={s['spoon_lift_mm_max']:6.1f}mm "
              f"spoon_nearest={s['spoon_to_target_mm_min']:6.1f}mm"
              + (f"  ERRORS={len(s['errors'])}" if s["errors"] else ""))


if __name__ == "__main__":
    main()
