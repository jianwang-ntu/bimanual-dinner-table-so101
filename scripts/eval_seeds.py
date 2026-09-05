#!/usr/bin/env python3
"""Run the randomized environment across N seeds and report what happened.

Two modes, and the difference between them is the point:

``--policy none``      (default) the arms hold the home pose.  The task score
                       is 0 on every seed and that is the correct result; what
                       the run establishes is that each randomized episode is a
                       valid, solvable starting point -- objects on the table,
                       inside an arm's reach, not interpenetrating, stable for
                       the whole episode.  It is also the negative control for
                       the scored mode: any subgoal that fires here would mean
                       the scorer, not the controller, produced it.

``--policy scripted``  the scripted bimanual controller in ``envs/controller``
                       drives the same episodes and is scored by the same
                       predicates.

Run:  python3 scripts/eval_seeds.py --seeds 10 --policy scripted
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

from envs.randomize import make_env, ARM_BASES, REACH_MAX      # noqa: E402
from envs.task import TaskMonitor, TABLE_TOP_Z                 # noqa: E402
from envs.controller import run_dinner_table                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
OBJECTS = ("plate", "mug", "bottle", "spoon", "fork")


def episode(seed: int, seconds: float, render: pathlib.Path | None,
            policy: str = "none") -> dict:
    model, data, log = make_env(seed)
    mon = TaskMonitor(model)

    # initial-state validity, measured before a single step is taken
    init = {}
    for nm in OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        xy = data.xpos[bid][:2]
        init[nm] = {
            "z": round(float(data.xpos[bid][2]), 4),
            "nearest_arm_m": round(float(min(np.linalg.norm(xy - b)
                                             for b in ARM_BASES.values())), 3),
        }
    valid = {
        "all_on_table": all(TABLE_TOP_Z - 0.03 < v["z"] < TABLE_TOP_Z + 0.30
                            for v in init.values()),
        "all_reachable": all(v["nearest_arm_m"] < REACH_MAX for v in init.values()),
        "no_initial_penetration": bool(
            data.ncon == 0 or float(np.min(data.contact.dist[:data.ncon])) > -0.002),
    }

    rollout = None
    if policy == "scripted":
        rollout = run_dinner_table(model, data, monitor=mon)
    else:
        steps = int(seconds / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            mon.step(data)

    stable = bool(np.all(np.isfinite(data.qpos))
                  and int(data.warning.number.sum()) == 0)

    if render is not None:
        with mujoco.Renderer(model, height=720, width=1280) as r:
            r.update_scene(data, camera="scene_cam")
            import imageio.v2 as imageio
            imageio.imwrite(render, r.render())

    return {
        "seed": seed,
        "policy": policy,
        "rollout": None if rollout is None else
                   {k: v for k, v in rollout.items() if k != "trace"},
        "randomization": log,
        "initial_state": init,
        "initial_state_valid": valid,
        "episode_stable": stable,
        "warnings": data.warning.number.tolist(),
        "task": mon.report(data),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--policy", choices=("none", "scripted"), default="none")
    ap.add_argument("--out", default=None,
                    help="evidence file to write (default depends on --policy)")
    args = ap.parse_args()

    frames = EVID / "seeds"
    frames.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(args.seeds):
        png = None if args.no_render else frames / f"seed_{s:02d}.png"
        row = episode(s, args.seconds, png, policy=args.policy)
        rows.append(row)
        v = row["initial_state_valid"]
        print(f"seed {s:2d}  on_table={v['all_on_table']!s:5s} "
              f"reachable={v['all_reachable']!s:5s} "
              f"no_penetration={v['no_initial_penetration']!s:5s} "
              f"stable={row['episode_stable']!s:5s} "
              f"subgoals={row['task']['subgoals_met']}/"
              f"{row['task']['subgoals_total']}")

    ok = all(all(r["initial_state_valid"].values()) and r["episode_stable"]
             for r in rows)
    met = [r["task"]["subgoals_met"] for r in rows]
    note = ("No policy is loaded. The arms hold the home pose, so a task score "
            "of 0 is the expected and correct result; this run scores the "
            "ENVIRONMENT, not a controller."
            if args.policy == "none" else
            "envs/controller.py, a scripted IK waypoint state machine. It "
            "holds no learned parameters and reads no camera; it reads the "
            "world pose of the object it is about to touch, which is what a "
            "perception stack would supply. Scored by the same predicates as "
            "the no-policy control.")
    out = {
        "seeds": args.seeds,
        "seconds_per_episode": args.seconds,
        "policy": None if args.policy == "none" else "scripted",
        "policy_note": note,
        "all_episodes_valid_and_stable": ok,
        "subgoals_met_per_seed": met,
        "subgoals_met_mean": round(float(np.mean(met)), 3) if met else 0.0,
        "task_success_count": sum(1 for r in rows if r["task"]["task_success"]),
        "handoff_seeds": [r["seed"] for r in rows if r["task"]["handoff_occurred"]],
        "episodes": rows,
    }
    EVID.mkdir(parents=True, exist_ok=True)
    name = args.out or ("eval_seeds.json" if args.policy == "none"
                        else "eval_seeds_scripted.json")
    (EVID / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{args.seeds} seeds, policy={args.policy}: "
          f"{'all valid and stable' if ok else 'FAILURES PRESENT'}; "
          f"subgoals {sum(met)}/{5 * len(met)}, "
          f"task_success {out['task_success_count']}/{len(met)} "
          f"-> evidence/{name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
