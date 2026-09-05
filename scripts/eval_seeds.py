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

Orthogonal to the policy, ``--scene`` chooses where that controller's object
positions come from (see ``envs/scene_source.py``):

``privileged``  ``MjData``.  The number this entry has always reported.
``perceived``   one ``top_cam`` frame per planning instant through the exported
                OpenVINO IR.  No true object pose reaches the control path.
``blind``       the nominal, un-randomized layout.  The negative control: it
                must score WORSE, or the controller is not consuming the source
                and the perceived number would mean nothing.

The scorer in ``envs/task.py`` is untouched by ``--scene`` and always reads the
simulator.  A run that let the estimate grade itself would not be an evaluation.

Run:  python3 scripts/eval_seeds.py --seeds 10 --policy scripted
      python3 scripts/eval_seeds.py --seeds 10 --policy scripted --scene perceived
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
from envs import scene_source                                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
OBJECTS = ("plate", "mug", "bottle", "spoon", "fork")


def episode(seed: int, seconds: float, render: pathlib.Path | None,
            policy: str = "none", scene: str = "privileged",
            scene_kw: dict | None = None, runner=None,
            policy_steps: int = 0, stride: int = 25) -> dict:
    model, data, log = make_env(seed)
    mon = TaskMonitor(model)
    src = scene_source.make(scene, **(scene_kw or {}))
    scene_source.install(src)

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
    try:
        if policy == "scripted":
            rollout = run_dinner_table(model, data, monitor=mon)
        elif policy == "act":
            # The learned policy drives the same actuators the scripted
            # controller does, at the rate the demonstrations were sampled at.
            # Every number it sees comes through `src`, so --scene decides
            # whether it is looking at the simulator or at a camera.
            from envs import act_policy                        # noqa: PLC0415
            adr = act_policy.actuated_qpos_adr(model)
            lo = model.actuator_ctrlrange[:, 0].copy()
            hi = model.actuator_ctrlrange[:, 1].copy()
            free = model.actuator_ctrllimited == 0
            lo[free], hi[free] = -np.inf, np.inf
            runner.reset()
            for _ in range(policy_steps):
                st, ev = act_policy.observe(model, data, src, adr)
                data.ctrl[:] = np.clip(runner.act(st, ev), lo, hi)
                for _ in range(stride):
                    mujoco.mj_step(model, data)
                    mon.step(data)
            rollout = {"moves": runner.steps,
                       "sim_time_s": round(float(data.time), 3),
                       "max_ik_err_mm": None,
                       "act": runner.report()}
        else:
            steps = int(seconds / model.opt.timestep)
            for _ in range(steps):
                mujoco.mj_step(model, data)
                mon.step(data)
        scene_report = src.report()
    finally:
        src.close()
        scene_source.reset()

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
        "scene_source": scene_report,
        "task": mon.report(data),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--policy", choices=("none", "scripted", "act"),
                    default="none")
    ap.add_argument("--policy-ckpt", default="models/act_policy.pt",
                    help="--policy act: the trained ACT checkpoint")
    ap.add_argument("--policy-steps", type=int, default=0,
                    help="--policy act: policy steps per episode "
                         "(0 = the median demonstration length)")
    ap.add_argument("--stride", type=int, default=25,
                    help="--policy act: simulator steps between policy "
                         "queries; must match collect_demos.py --stride")
    ap.add_argument("--scene", choices=tuple(scene_source.SOURCES),
                    default="privileged",
                    help="where the controller reads object positions from")
    ap.add_argument("--precision", default="FP32",
                    choices=("FP32", "FP16", "INT8"),
                    help="--scene perceived: which exported IR to run")
    ap.add_argument("--device", default="CPU",
                    help="--scene perceived: OpenVINO device")
    ap.add_argument("--backend", default="openvino",
                    choices=("openvino", "torch"),
                    help="--scene perceived: inference runtime")
    ap.add_argument("--out", default=None,
                    help="evidence file to write (default depends on --policy)")
    args = ap.parse_args()

    frames = EVID / "seeds"
    frames.mkdir(parents=True, exist_ok=True)
    scene_kw = ({"precision": args.precision, "device": args.device,
                 "backend": args.backend} if args.scene == "perceived" else {})
    runner, policy_steps = None, args.policy_steps
    if args.policy == "act":
        from envs import act_policy                            # noqa: PLC0415
        runner = act_policy.ActRunner(ROOT / args.policy_ckpt)
        if policy_steps == 0:
            # As long as the demonstrator's own episodes ran, so the learned
            # policy is given the same wall of simulator time to work in and
            # neither run is scored over a longer horizon than the other.
            lens = [-(-e["sim_steps"] // args.stride)
                    for f in sorted((ROOT / "data/demos").glob("*.json"))
                    for e in json.loads(f.read_text())["episodes"]]
            policy_steps = int(np.median(lens))
        clash = sorted(set(runner.train_seeds) & set(range(args.seeds)))
        if clash:
            print(f"REFUSED: checkpoint was trained on evaluation seeds "
                  f"{clash}.", file=sys.stderr)
            return 2

    rows = []
    for s in range(args.seeds):
        png = None if args.no_render else frames / f"seed_{s:02d}.png"
        row = episode(s, args.seconds, png, policy=args.policy,
                      scene=args.scene, scene_kw=scene_kw, runner=runner,
                      policy_steps=policy_steps, stride=args.stride)
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
    SCENE_NOTE = {
        "privileged": "The controller read the world pose of the object it was "
                      "about to touch straight out of MjData. This is the "
                      "privileged control.",
        "perceived":  "The controller read NO true object pose. The planar "
                      "centres of plate, mug and bottle and the drawer opening "
                      "came from one top_cam frame per planning instant through "
                      "the exported OpenVINO IR. Object height, yaw and "
                      "dimensions, and the spoon and fork, are still privileged "
                      "-- envs/scene_source.py lists exactly which.",
        "blind":      "NEGATIVE CONTROL. The controller was handed the nominal, "
                      "un-randomized layout from envs/randomize.py instead of an "
                      "estimate. It must score worse than privileged; if it does "
                      "not, the controller is not consuming the scene source.",
    }
    ACT_NOTE = (
        "A LeRobot ACT policy (lerobot.policies.act.modeling_act.ACTPolicy), "
        "trained by behaviour cloning on the scripted controller's rollouts on "
        "seeds it never sees at evaluation. It holds learned parameters and no "
        "waypoint script: at every policy step it is handed twelve joint "
        "positions and seven scene numbers and returns twelve actuator "
        "targets. It cannot exceed its demonstrator on sub-goals the "
        "demonstrator never achieves. Scored by the same predicates as the "
        "no-policy control and the scripted run. ")
    note = ("No policy is loaded. The arms hold the home pose, so a task score "
            "of 0 is the expected and correct result; this run scores the "
            "ENVIRONMENT, not a controller."
            if args.policy == "none" else
            ACT_NOTE + SCENE_NOTE[args.scene] if args.policy == "act" else
            "envs/controller.py, a scripted IK waypoint state machine. It "
            "holds no learned parameters. Scored by the same predicates as the "
            "no-policy control, and by a scorer that always reads the "
            "simulator whatever --scene is set to. "
            + SCENE_NOTE[args.scene])
    est = [r["scene_source"] for r in rows
           if r["scene_source"].get("worst_object_mean_mm") is not None]
    out = {
        "seeds": args.seeds,
        "seconds_per_episode": args.seconds,
        "policy": None if args.policy == "none" else args.policy,
        "policy_detail": (None if runner is None else
                          {**runner.report(), "policy_steps_per_episode":
                           policy_steps, "sim_steps_per_policy_step":
                           args.stride, "checkpoint": args.policy_ckpt}),
        "policy_note": note,
        "scene_source": args.scene,
        "scene_source_detail": (rows[0]["scene_source"]["source"] if rows
                                else args.scene),
        "scene_source_note": SCENE_NOTE[args.scene],
        "scorer_reads": "MjData -- envs/task.py is never routed through the "
                        "scene source",
        "estimate_error_mm": None if not est else {
            "worst_object_mean_over_seeds": round(
                float(np.mean([e["worst_object_mean_mm"] for e in est])), 3),
            "worst_object_max_over_seeds": round(
                float(np.max([max(e[k]["max"] for k in
                                  ("plate_mm", "mug_mm", "bottle_mm"))
                              for e in est])), 3),
            "at_t0_worst_object_mean": round(
                float(np.mean([max(e[k]["first"] for k in
                                   ("plate_mm", "mug_mm", "bottle_mm"))
                               for e in est])), 3),
            "inferences_total": int(sum(e["inferences"] for e in est)),
            "note": "distance from the true planar centre, in millimetres, at "
                    "every planning instant -- how wrong the controller's view "
                    "of the table was when it planned each waypoint. "
                    + ("For --scene blind there is no inference: this is the "
                       "fixed gap between the nominal layout and the "
                       "randomized one, and it is the control's whole point."
                       if args.scene == "blind" else
                       "t0 is an unoccluded table; later instants have two arms "
                       "over it."),
        },
        "all_episodes_valid_and_stable": ok,
        "subgoals_met_per_seed": met,
        "subgoals_met_mean": round(float(np.mean(met)), 3) if met else 0.0,
        "task_success_count": sum(1 for r in rows if r["task"]["task_success"]),
        "handoff_seeds": [r["seed"] for r in rows if r["task"]["handoff_occurred"]],
        "episodes": rows,
    }
    EVID.mkdir(parents=True, exist_ok=True)
    default = {"none": "eval_seeds.json",
               "scripted": "eval_seeds_scripted.json",
               "act": "eval_seeds_act.json"}[args.policy]
    if args.scene != "privileged" and args.out is None:
        default = f"eval_seeds_{args.policy}_{args.scene}.json"
    name = args.out or default
    (EVID / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{args.seeds} seeds, policy={args.policy}, scene={args.scene}: "
          f"{'all valid and stable' if ok else 'FAILURES PRESENT'}; "
          f"subgoals {sum(met)}/{5 * len(met)}, "
          f"task_success {out['task_success_count']}/{len(met)} "
          f"-> evidence/{name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
