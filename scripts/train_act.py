#!/usr/bin/env python3
"""Train an ACT policy on the scripted controller's rollouts (behaviour cloning).

Track Objective 4: "Train or fine-tune a robotics policy in MuJoCo using Hugging
Face LeRobot or compatible tooling.  Candidate policies may include SmolVLA,
Pi0.5, ACT, or another appropriate VLA / imitation-learning policy."

This script uses the real thing: ``lerobot.policies.act.modeling_act.ACTPolicy``
from Hugging Face LeRobot, the action-chunking transformer with a CVAE encoder.
Nothing here re-implements it.

Observation / action space
--------------------------
``observation.state``              12 actuated joint positions (proprioception)
``observation.environment_state``   7 scene numbers -- plate (x, y), mug (x, y),
                                    bottle (x, y), drawer opening.  These are the
                                    exact outputs of ``envs/perception.py``, so at
                                    evaluation time the policy can be fed the
                                    camera-and-OpenVINO estimate through
                                    ``envs/scene_source.py`` instead of the truth.
``action``                         12 actuator position targets, chunked.

No image feature is configured, and that is a limitation, not an oversight: ACT's
ResNet backbone over a 128x224 frame at every planning instant is a second
vision stack beside the one this project already exports to OpenVINO.  The camera
enters through ``observation.environment_state`` instead.  Said here so it is not
read off the architecture as an image-conditioned policy.

Normalization is done here, not by the policy: lerobot 0.6 moved normalization
out of ``ACTPolicy`` into its dataset processors, and this project does not use
``LeRobotDataset``.  Mean/std are computed on the TRAIN episodes only and stored
in the checkpoint so evaluation cannot silently use different ones.

Run:  python3 scripts/train_act.py --steps 6000 --out models/act_policy.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

import av                                                       # noqa: F401
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

STATE_DIM, ENV_DIM, ACTION_DIM = 12, 7, 12


def load_demos(d: pathlib.Path) -> dict:
    files = sorted(d.glob("*.npz"))
    if not files:
        raise SystemExit(f"no demonstrations in {d}")
    parts = [np.load(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in
           ("state", "env_state", "action", "episode", "frame")}
    out["files"] = [f.name for f in files]
    return out


def episode_slices(episode: np.ndarray) -> dict[int, tuple[int, int]]:
    """First and last+1 row of each episode, in the order they appear."""
    out, start = {}, 0
    for i in range(1, len(episode) + 1):
        if i == len(episode) or episode[i] != episode[start]:
            out[int(episode[start])] = (start, i)
            start = i
    return out


class ChunkSet(torch.utils.data.Dataset):
    """One sample = the observation at t, and the next `chunk` actions."""

    def __init__(self, d: dict, seeds: list[int], chunk: int, norm: dict):
        self.chunk = chunk
        sl = episode_slices(d["episode"])
        self.spans = [sl[s] for s in seeds]
        self.index = [(k, i) for k, (a, b) in enumerate(self.spans)
                      for i in range(b - a)]
        self.state = (d["state"] - norm["state_mean"]) / norm["state_std"]
        self.env = (d["env_state"] - norm["env_mean"]) / norm["env_std"]
        self.action = (d["action"] - norm["action_mean"]) / norm["action_std"]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, n: int) -> dict:
        k, i = self.index[n]
        a, b = self.spans[k]
        t = a + i
        end = min(t + self.chunk, b)
        act = np.zeros((self.chunk, ACTION_DIM), dtype=np.float32)
        pad = np.ones(self.chunk, dtype=bool)
        act[: end - t] = self.action[t:end]
        pad[: end - t] = False
        if end - t < self.chunk:                  # hold the last commanded pose
            act[end - t:] = self.action[end - 1]
        return {
            OBS_STATE: torch.from_numpy(self.state[t]),
            OBS_ENV_STATE: torch.from_numpy(self.env[t]),
            ACTION: torch.from_numpy(act),
            "action_is_pad": torch.from_numpy(pad),
        }


def make_policy(chunk: int, n_action_steps: int, device: str) -> ACTPolicy:
    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(ENV_DIM,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        chunk_size=chunk,
        n_action_steps=n_action_steps,
        device=device,
    )
    return ACTPolicy(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=pathlib.Path, default=ROOT / "data/demos")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-episodes", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "models/act_policy.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_demos(a.demos)
    seeds = sorted(episode_slices(d["episode"]))
    val_seeds = seeds[-a.val_episodes:] if a.val_episodes else []
    train_seeds = [s for s in seeds if s not in val_seeds]

    sl = episode_slices(d["episode"])
    tr = np.concatenate([np.arange(*sl[s]) for s in train_seeds])
    norm = {
        "state_mean": d["state"][tr].mean(0), "state_std": d["state"][tr].std(0) + 1e-6,
        "env_mean": d["env_state"][tr].mean(0), "env_std": d["env_state"][tr].std(0) + 1e-6,
        "action_mean": d["action"][tr].mean(0), "action_std": d["action"][tr].std(0) + 1e-6,
    }

    ds = ChunkSet(d, train_seeds, a.chunk, norm)
    vs = ChunkSet(d, val_seeds, a.chunk, norm) if val_seeds else None
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True,
                                     num_workers=4, drop_last=True,
                                     persistent_workers=True)

    policy = make_policy(a.chunk, a.n_action_steps, device).to(device)
    n_par = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=1e-4)

    print(f"device={device} params={n_par:,} train_eps={len(train_seeds)} "
          f"val_eps={len(val_seeds)} train_samples={len(ds)}", flush=True)

    def val_l1() -> float | None:
        if vs is None:
            return None
        policy.eval()
        loader = torch.utils.data.DataLoader(vs, batch_size=256, shuffle=False)
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                pred = policy.predict_action_chunk(b)
                m = (~b["action_is_pad"]).unsqueeze(-1)
                tot += float((( (pred - b[ACTION]).abs() * m).sum()))
                n += int(m.sum()) * ACTION_DIM
        policy.train()
        return tot / max(n, 1)

    t0 = time.time()
    hist, step = [], 0
    policy.train()
    stop = False
    while not stop:
        for b in dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            loss, parts = policy.forward(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            step += 1
            if step % 250 == 0 or step == 1:
                hist.append({"step": step, "loss": round(float(loss), 5),
                             **{k: round(v, 5) for k, v in parts.items()},
                             "wall_s": round(time.time() - t0, 1)})
                print(hist[-1], flush=True)
            if step >= a.steps or time.time() - t0 > a.max_seconds:
                stop = True
                break

    v = val_l1()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "norm": {k: np.asarray(x) for k, x in norm.items()},
        "chunk": a.chunk,
        "n_action_steps": a.n_action_steps,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
    }, a.out)

    report = {
        "policy": "ACT (Action Chunking Transformer, CVAE)",
        "implementation": "lerobot.policies.act.modeling_act.ACTPolicy",
        "lerobot_version": __import__("lerobot").__version__,
        "trained_by": "behaviour cloning on envs/controller.py rollouts",
        "parameters": int(n_par),
        "device": device,
        "chunk_size": a.chunk,
        "n_action_steps": a.n_action_steps,
        "batch": a.batch,
        "lr": a.lr,
        "steps_run": step,
        "wall_s": round(time.time() - t0, 1),
        "train_episodes": train_seeds,
        "val_episodes": val_seeds,
        "train_samples": len(ds),
        "val_samples": 0 if vs is None else len(vs),
        "val_l1_normalized": None if v is None else round(v, 5),
        "loss_history": hist,
        "demo_files": d["files"],
        "observation": {
            OBS_STATE: "12 actuated joint positions",
            OBS_ENV_STATE: "plate_x, plate_y, mug_x, mug_y, bottle_x, "
                           "bottle_y, drawer_q -- the outputs of "
                           "envs/perception.py",
        },
        "no_image_feature": "ACT is configured with no camera input. The camera "
                            "reaches the policy only through the seven "
                            "environment-state numbers, which at evaluation "
                            "time envs/scene_source.py can produce from a "
                            "top_cam frame through the OpenVINO IR. This is a "
                            "state-conditioned ACT, not an image-conditioned "
                            "one, and no claim is made otherwise.",
        "normalization": "mean/std over the TRAIN episodes only, stored in the "
                         "checkpoint; lerobot 0.6 does not normalize inside "
                         "ACTPolicy.",
        "not_claimed": "This is imitation of a scripted controller. It cannot "
                       "exceed its demonstrator on the sub-goals the "
                       "demonstrator never achieves, and no language input is "
                       "consumed anywhere.",
    }
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "act_train.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"saved {a.out}  val_l1_normalized={v}  -> evidence/act_train.json",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
