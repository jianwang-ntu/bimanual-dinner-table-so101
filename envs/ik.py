#!/usr/bin/env python3
"""Damped-least-squares site IK for the SO-101 arms.

Small enough to read in one sitting and used by both the scene builder (to pick
a sane home pose) and the scripted rollout (to reach a Cartesian waypoint).
"""
from __future__ import annotations

import numpy as np
import mujoco

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll")


def arm_dof(model: mujoco.MjModel, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos addresses, dof addresses) of one arm's 5 positioning joints."""
    qadr, vadr = [], []
    for j in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
        if jid < 0:
            raise KeyError(f"joint {prefix + j} not in model")
        qadr.append(model.jnt_qposadr[jid])
        vadr.append(model.jnt_dofadr[jid])
    return np.array(qadr), np.array(vadr)


def site_ik(model: mujoco.MjModel, data: mujoco.MjData, prefix: str,
            site: str, target: np.ndarray, *, iters: int = 400,
            damping: float = 0.06, step: float = 0.5,
            tol: float = 2e-3) -> tuple[np.ndarray, float]:
    """Move one arm so ``site`` reaches ``target``.

    Returns the joint vector and the final position error in metres.  The arm's
    joint limits are respected by clipping every iteration, so a target outside
    the reachable set returns the closest pose the solver found rather than a
    silently illegal one.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
    if sid < 0:
        raise KeyError(f"site {prefix + site} not in model")
    qadr, vadr = arm_dof(model, prefix)
    lo = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)][0] for j in ARM_JOINTS])
    hi = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)][1] for j in ARM_JOINTS])

    jacp = np.zeros((3, model.nv))
    best_q, best_err = data.qpos[qadr].copy(), np.inf
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err = target - data.site_xpos[sid]
        n = float(np.linalg.norm(err))
        if n < best_err:
            best_err, best_q = n, data.qpos[qadr].copy()
        if n < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J = jacp[:, vadr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + step * dq, lo, hi)
    data.qpos[qadr] = best_q
    mujoco.mj_kinematics(model, data)
    return best_q, best_err
