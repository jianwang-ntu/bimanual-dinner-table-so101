#!/usr/bin/env python3
"""Record the scripted controller's own rollouts as an imitation-learning dataset.

Track Objective 4 asks for a policy *trained* in MuJoCo, not for a hand-written
script.  The only demonstrator this project has is the scripted bimanual
controller in ``envs/controller.py``, so the honest route to a learned policy is
behaviour cloning from it: run the script on seeds the evaluation never sees,
log what it observed and what it commanded, and train on that.

What one sample is
------------------
Sampled every ``--stride`` simulator steps (default 25, i.e. 20 Hz at the
scene's 2 ms timestep):

``state``      the 12 actuated joint positions read from ``qpos`` -- the arms'
               own proprioception, which a real cell has.
``env_state``  the 7 numbers ``envs/perception.py`` regresses: plate (x, y),
               mug (x, y), bottle (x, y) and the drawer opening.  Recorded here
               from ``MjData`` because these are the training *labels*; at
               evaluation time the same seven numbers are supplied by
               ``envs/scene_source.py``, which can hand the policy the
               camera-and-OpenVINO estimate instead.  The observation vector is
               identical in both cases -- that is the whole point of the seam.
``action``     the 12 actuator position targets ``data.ctrl`` held at that
               instant.  This is what the policy is asked to reproduce.

Seed hygiene
------------
Demonstrations default to seeds 3000+.  The evaluation seeds are 0..9 and the
perception model's seeds are 1000-1499 and 2000-2005.  No demonstration episode
is an evaluation episode, and the script refuses to start if the requested
range overlaps 0..9.

Run:  python3 scripts/collect_demos.py --seed-start 3000 --count 8 --out data/demos_3000.npz
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

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from envs.randomize import make_env                            # noqa: E402
from envs.task import TaskMonitor                              # noqa: E402
from envs.controller import run_dinner_table                   # noqa: E402
from envs import perception as P                               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_SEEDS = tuple(range(10))

# The seven numbers, in envs/perception.py's own output order.  Derived from the
# perception module rather than retyped, so the two cannot drift apart.
ENV_STATE_NAMES: tuple[str, ...] = tuple(
    f"{o}_{ax}" for o in P.OBJECTS for ax in ("x", "y")) + ("drawer_q",)


def actuated_qpos_adr(model) -> np.ndarray:
    """qpos address of the joint each actuator drives, in actuator order."""
    adr = []
    for i in range(model.nu):
        assert model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT, (
            f"actuator {i} is not a joint transmission")
        jid = int(model.actuator_trnid[i, 0])
        adr.append(int(model.jnt_qposadr[jid]))
    return np.asarray(adr, dtype=int)


def env_state(model, data, obj_bid, drawer_adr) -> np.ndarray:
    out = np.empty(len(ENV_STATE_NAMES), dtype=np.float32)
    for k, bid in enumerate(obj_bid):
        out[2 * k:2 * k + 2] = data.xpos[bid][:2]
    out[-1] = data.qpos[drawer_adr]
    return out


def collect(seed: int, stride: int) -> dict:
    model, data, log = make_env(seed)
    mon = TaskMonitor(model)
    adr = actuated_qpos_adr(model)
    obj_bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o)
               for o in P.OBJECTS]
    drawer_adr = int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")])

    states, envs_, actions, times = [], [], [], []
    n = [0]

    def on_step(d):
        if n[0] % stride == 0:
            states.append(d.qpos[adr].astype(np.float32).copy())
            envs_.append(env_state(model, d, obj_bid, drawer_adr))
            actions.append(d.ctrl.astype(np.float32).copy())
            times.append(float(d.time))
        n[0] += 1

    t0 = time.time()
    roll = run_dinner_table(model, data, monitor=mon, on_step=on_step)
    report = mon.report(data)
    return {
        "seed": seed,
        "state": np.asarray(states, dtype=np.float32),
        "env_state": np.asarray(envs_, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "t": np.asarray(times, dtype=np.float32),
        "sim_time_s": roll["sim_time_s"],
        "sim_steps": n[0],
        "wall_s": round(time.time() - t0, 2),
        "subgoals": report["subgoals"],
        "subgoals_met": report["subgoals_met"],
        "randomization": log,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=3000)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    seeds = list(range(a.seed_start, a.seed_start + a.count))
    clash = sorted(set(seeds) & set(EVAL_SEEDS))
    if clash:
        print(f"REFUSED: demonstration seeds {clash} are evaluation seeds. "
              f"Training on the seeds we report would make the score meaningless.",
              file=sys.stderr)
        return 2

    eps, meta = [], []
    for s in seeds:
        e = collect(s, a.stride)
        eps.append(e)
        meta.append({k: e[k] for k in
                     ("seed", "sim_time_s", "sim_steps", "wall_s",
                      "subgoals", "subgoals_met")})
        print(f"seed {s}: {len(e['state'])} samples, "
              f"{e['subgoals_met']}/5 sub-goals, {e['wall_s']}s", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        state=np.concatenate([e["state"] for e in eps]),
        env_state=np.concatenate([e["env_state"] for e in eps]),
        action=np.concatenate([e["action"] for e in eps]),
        t=np.concatenate([e["t"] for e in eps]),
        episode=np.concatenate([np.full(len(e["state"]), e["seed"], np.int32)
                                for e in eps]),
        frame=np.concatenate([np.arange(len(e["state"]), dtype=np.int32)
                              for e in eps]),
    )
    a.out.with_suffix(".json").write_text(json.dumps({
        "seeds": seeds,
        "stride": a.stride,
        "hz": round(1.0 / (0.002 * a.stride), 3),
        "state_dim": 12,
        "env_state_dim": len(ENV_STATE_NAMES),
        "env_state_names": list(ENV_STATE_NAMES),
        "action_dim": 12,
        "action_is": "data.ctrl -- actuator position targets, same units as state",
        "env_state_source": "MjData (training labels). At evaluation the same "
                            "seven numbers come from envs/scene_source.py.",
        "eval_seeds_excluded": list(EVAL_SEEDS),
        "episodes": meta,
        "total_samples": int(sum(len(e["state"]) for e in eps)),
    }, indent=1) + "\n")
    print(f"wrote {a.out} "
          f"({sum(len(e['state']) for e in eps)} samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
