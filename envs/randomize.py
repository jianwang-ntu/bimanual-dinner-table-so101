#!/usr/bin/env python3
"""Seeded domain randomization for the dinner-table scene.

The track brief scores robustness under "randomized object placement, weights,
friction, shapes, lighting, background".  Those split across three stages
because MuJoCo needs a recompile only for the first:

  geometry  object radii, heights, cutlery length      -> varies the MjSpec
  model     masses, friction, lighting, surface colour -> patches the MjModel
  state     object x, y, yaw and drawer opening        -> patches the MjData

``make_env(seed)`` runs all three and returns a compiled (model, data) pair
whose initial state has been rejection-sampled to be collision-free and inside
at least one arm's reach envelope, so a failed episode is a policy failure and
not an impossible scene.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import dinner_table as dt

# Every randomized quantity, as a multiplicative or additive range.  Kept in one
# table so the README can quote it and the evaluation can log it.
RANGES = {
    "plate_r":    (0.046, 0.058),      # m
    "mug_r":      (0.024, 0.032),
    "mug_h":      (0.028, 0.040),
    "bottle_r":   (0.022, 0.030),
    "bottle_h":   (0.045, 0.060),
    "cutlery_l":  (0.032, 0.044),
    "mass_scale": (0.6, 1.6),          # x nominal, per object
    "friction":   (0.6, 1.4),          # x nominal sliding friction
    "light_xy":   (-0.35, 0.35),       # m, key-light offset
    "light_gain": (0.45, 0.95),        # diffuse
    "table_rgb":  (0.45, 0.85),        # background/table lightness
    "place_xy":   (-0.045, 0.045),     # m, object placement jitter
    "place_yaw":  (-np.pi, np.pi),     # rad
    "drawer_q":   (0.0, 0.02),         # m, drawer starts almost but not fully shut
}

GRASPABLES = ("plate", "mug", "bottle", "spoon", "fork")
NOMINAL_XY = {                          # nominal table positions, jittered below
    "plate":  (0.25, 0.115),
    "mug":    (-0.25, 0.115),
    "bottle": (0.35, -0.02),
}
ARM_BASES = {"left": np.array([-dt.ARM_X, dt.ARM_Y]),
             "right": np.array([dt.ARM_X, dt.ARM_Y])}
REACH_MAX = 0.40


def _u(rng, key):
    lo, hi = RANGES[key]
    return float(rng.uniform(lo, hi))


def sample_dims(rng) -> dict:
    return {k: _u(rng, k) for k in
            ("plate_r", "mug_r", "mug_h", "bottle_r", "bottle_h", "cutlery_l")}


def randomize_model(model: mujoco.MjModel, rng) -> dict:
    """Masses, friction, lighting and surface colour.  No recompile needed."""
    log = {"mass_scale": {}, "friction_scale": round(_u(rng, "friction"), 3)}

    for nm in GRASPABLES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        s = _u(rng, "mass_scale")
        model.body_mass[bid] *= s
        model.body_inertia[bid] *= s
        log["mass_scale"][nm] = round(s, 3)

    model.geom_friction[:, 0] *= log["friction_scale"]

    for i in range(model.nlight):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) == "key":
            model.light_pos0[i, 0] += _u(rng, "light_xy")
            model.light_pos0[i, 1] += _u(rng, "light_xy")
            g = _u(rng, "light_gain")
            model.light_diffuse[i, :] = g
            log["light"] = {"pos": model.light_pos0[i].round(3).tolist(),
                            "diffuse": round(g, 3)}

    for mat in ("wood", "grid_mat"):                       # table + background
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
        if mid >= 0:
            v = _u(rng, "table_rgb")
            model.mat_rgba[mid, :3] = np.clip(
                model.mat_rgba[mid, :3] * (v / 0.65), 0.05, 1.0)
            log.setdefault("material_lightness", {})[mat] = round(v, 3)
    return log


def _obj_radius(model, name) -> float:
    """Planar half-extent of a body, used for the overlap test."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    r = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        r = max(r, float(np.linalg.norm(model.geom_pos[g][:2])
                         + max(model.geom_size[g][:2])))
    return r


def randomize_state(model: mujoco.MjModel, data: mujoco.MjData, rng,
                    *, tries: int = 200) -> dict:
    """Place the free objects, rejecting overlapping or out-of-reach draws."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.ctrl[:] = model.key_ctrl[kid]

    radii = {n: _obj_radius(model, n) for n in NOMINAL_XY}
    placed, log = {}, {}
    for name, (nx, ny) in NOMINAL_XY.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        adr = model.jnt_qposadr[jid]
        for _ in range(tries):
            xy = np.array([nx + _u(rng, "place_xy"), ny + _u(rng, "place_xy")])
            if min(np.linalg.norm(xy - b) for b in ARM_BASES.values()) > REACH_MAX:
                continue
            if any(np.linalg.norm(xy - q) < radii[name] + radii[o] + 0.01
                   for o, q in placed.items()):
                continue
            if abs(xy[0]) < 0.12 and xy[1] > -0.02:        # keep the drawer clear
                continue
            break
        else:
            raise RuntimeError(f"could not place {name} in {tries} draws")
        yaw = _u(rng, "place_yaw")
        data.qpos[adr:adr + 2] = xy
        data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        placed[name] = xy
        log[name] = {"xy": xy.round(4).tolist(), "yaw": round(yaw, 3)}

    dj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    q = _u(rng, "drawer_q")
    data.qpos[model.jnt_qposadr[dj]] = q
    log["drawer_slide"] = round(q, 4)

    mujoco.mj_forward(model, data)
    return log


def make_env(seed: int) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile and reset one randomized episode.  Returns (model, data, log)."""
    rng = np.random.default_rng(seed)
    dims = sample_dims(rng)

    spec = dt.build(dims=dims)
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos, ctrl, _ = dt.home_keyframe(model, data)
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    model = spec.compile()
    data = mujoco.MjData(model)

    log = {"seed": seed, "dims": {k: round(v, 4) for k, v in dims.items()}}
    log["model"] = randomize_model(model, rng)
    log["state"] = randomize_state(model, data, rng)
    return model, data, log
