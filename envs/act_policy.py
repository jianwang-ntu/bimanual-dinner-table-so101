#!/usr/bin/env python3
"""Run a trained LeRobot ACT policy in the closed loop.

This is the inference side of ``scripts/train_act.py``.  It is deliberately thin:
the network, the CVAE decoder and the action queue are LeRobot's
``ACTPolicy``; what lives here is the contract between the simulator and the
policy -- which numbers go in, in what order, normalized how, and what comes
back out.

The observation the policy is handed at evaluation time is assembled from a
``SceneSource`` (``envs/scene_source.py``), not from ``MjData`` directly.  So the
same checkpoint can be driven from privileged state, from the camera through the
OpenVINO IR, or from the blind nominal layout, and the difference is measurable.

Deliberately NOT hidden: the actuator positions in ``observation.state`` are read
straight from ``qpos``.  That is proprioception, which a real cell has, and it is
the same privilege the scripted controller already took.
"""
from __future__ import annotations

import pathlib

import numpy as np
import mujoco

from . import perception as P

DRAWER_JOINT = "drawer_slide"


def actuated_qpos_adr(model) -> np.ndarray:
    """qpos address of the joint each actuator drives, in actuator order.

    Duplicated from scripts/collect_demos.py on purpose: training and inference
    must agree on the order of the twelve numbers, and a shared import that one
    of them could quietly stop using would hide a mismatch rather than surface
    it.  scripts/test_act_policy.py asserts the two agree.
    """
    adr = []
    for i in range(model.nu):
        assert model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        adr.append(int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])]))
    return np.asarray(adr, dtype=int)


def observe(model, data, src, adr) -> tuple[np.ndarray, np.ndarray]:
    """(state, env_state) exactly as scripts/collect_demos.py recorded them."""
    state = data.qpos[adr].astype(np.float32)
    env = np.empty(2 * len(P.OBJECTS) + 1, dtype=np.float32)
    for k, name in enumerate(P.OBJECTS):
        env[2 * k:2 * k + 2] = src.body_xpos(model, data, name)[:2]
    env[-1] = src.drawer_q(model, data)
    return state, env


class ActRunner:
    """A checkpoint, its normalization, and the action queue LeRobot manages."""

    def __init__(self, ckpt: pathlib.Path | str, device: str | None = None):
        import torch
        import av                                               # noqa: F401
        from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

        self._torch = torch
        self._keys = (OBS_STATE, OBS_ENV_STATE)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from scripts.train_act import make_policy

        self.policy = make_policy(blob["chunk"], blob["n_action_steps"],
                                  self.device)
        self.policy.load_state_dict(blob["state_dict"])
        self.policy.to(self.device).eval()
        self.norm = {k: np.asarray(v, dtype=np.float32)
                     for k, v in blob["norm"].items()}
        self.chunk = int(blob["chunk"])
        self.n_action_steps = int(blob["n_action_steps"])
        self.train_seeds = list(blob.get("train_seeds", []))
        self.queries = 0
        self.steps = 0

    def reset(self) -> None:
        """Clear the action queue AND the counters.

        The counters are per-episode.  One runner is built once and driven
        across every seed, so a counter that survived reset() would report each
        episode's step count as the running total -- which is exactly what the
        first version of this file did, and what reading its own evidence file
        caught.
        """
        self.policy.reset()
        self.queries = 0
        self.steps = 0

    def act(self, state: np.ndarray, env_state: np.ndarray) -> np.ndarray:
        torch = self._torch
        ks, ke = self._keys
        s = (state - self.norm["state_mean"]) / self.norm["state_std"]
        e = (env_state - self.norm["env_mean"]) / self.norm["env_std"]
        empty = len(self.policy._action_queue) == 0
        batch = {
            ks: torch.from_numpy(np.asarray(s, np.float32))[None].to(self.device),
            ke: torch.from_numpy(np.asarray(e, np.float32))[None].to(self.device),
        }
        with torch.no_grad():
            a = self.policy.select_action(batch)[0].float().cpu().numpy()
        self.queries += int(empty)
        self.steps += 1
        return a * self.norm["action_std"] + self.norm["action_mean"]

    def report(self) -> dict:
        return {
            "policy": "ACT (lerobot.policies.act.modeling_act.ACTPolicy)",
            "device": self.device,
            "chunk_size": self.chunk,
            "n_action_steps": self.n_action_steps,
            "policy_steps": self.steps,
            "network_queries": self.queries,
            "trained_on_seeds": self.train_seeds,
        }
