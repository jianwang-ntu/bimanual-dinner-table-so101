#!/usr/bin/env python3
"""The bimanual dinner-table MuJoCo scene, built from the SO-101 description.

Two Robot Studio SO-101 arms are attached to a shared table with the prefixes
``left_`` and ``right_``.  Everything else -- table, drawer cabinet, plate,
mug, bottle, spoon, fork, placement targets and cameras -- is authored here so
the geometry is one reviewable source file rather than a binary.

Output: ``envs/dinner_table.xml``, an MJCF that references the SO-101 meshes
in ``third_party/robotstudio_so101/assets``.

``build()`` returns an uncompiled ``MjSpec`` so the randomizer can vary object
geometry before compiling; ``scripts/build_scene.py`` writes the nominal scene
out as ``envs/dinner_table.xml``.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

import mujoco

ROOT = pathlib.Path(__file__).resolve().parent.parent
SO101 = ROOT / "third_party" / "robotstudio_so101" / "so101.xml"
ASSETS = ROOT / "third_party" / "robotstudio_so101" / "assets"
OUT = ROOT / "envs" / "dinner_table.xml"

# ---------------------------------------------------------------- geometry
TABLE_TOP_Z = 0.75          # m, world height of the working surface
TABLE_HALF = (0.45, 0.35, 0.02)
ARM_X = 0.22                # arms sit at x = -ARM_X and x = +ARM_X
ARM_Y = -0.14
HANDOFF = (0.0, -0.075)     # x, y of the shared hand-off zone
CAB_Y, CAB_Z = 0.16, TABLE_TOP_Z
# Half-height of the cabinet carcass.  It has to leave a gripper room to reach
# down INTO the open drawer: with a low top panel the only way to the cutlery
# is a 25 mm slot between the drawer front and the overhang, which no SO-101
# gripper fits through, and the drawer is decorative rather than part of the
# task.  Measured clearance above the drawer rim at this value: 177 mm.
CAB_H = 0.12
DRAWER_TRAVEL = 0.09        # m
ARM_YAW = 36.0              # deg, left mount; the right mount mirrors it
HOME_XY = 0.10              # gripper x offset of the home pose
HOME_Z = 0.13               # gripper height above the table at home
GRIPPER_OPEN = 1.20         # rad

# Contact parameters that let a parallel jaw actually hold an object.
GRASP = dict(
    friction=[1.0, 0.05, 0.0005],
    solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
    solref=[0.004, 1.0],
    condim=4,
)
SLIDE = dict(friction=[0.9, 0.02, 0.0002], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
             solref=[0.005, 1.0], condim=3)

# Object geometry the randomizer is allowed to vary.  Every value is metres or
# kilograms; ``build(dims=...)`` merges an override dict over these defaults so
# a randomized episode differs in object *shape* as well as in placement.
DEFAULT_DIMS = {
    "plate_r": 0.052, "plate_h": 0.005, "plate_m": 0.090,
    "mug_r": 0.028, "mug_h": 0.033, "mug_m": 0.110,
    "bottle_r": 0.026, "bottle_h": 0.052, "bottle_m": 0.160,
    "cutlery_l": 0.038, "cutlery_m": 0.020,
}

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
CAP = mujoco.mjtGeom.mjGEOM_CAPSULE


def zquat(deg: float) -> list[float]:
    a = math.radians(deg) / 2.0
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def box(parent, name, pos, half, material, **kw):
    return parent.add_geom(name=name, type=BOX, pos=list(pos),
                           size=list(half), material=material, **kw)


def free_body(spec, name, pos):
    b = spec.worldbody.add_body(name=name, pos=list(pos))
    b.add_freejoint(name=f"{name}_free")
    return b


def build(dims: dict | None = None) -> mujoco.MjSpec:
    D = dict(DEFAULT_DIMS)
    D.update(dims or {})

    spec = mujoco.MjSpec()
    spec.modelname = "dinner_table_bimanual_so101"
    spec.compiler.degree = False          # radians, matching so101.xml
    spec.compiler.autolimits = True

    spec.option.timestep = 0.002
    spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    spec.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    spec.option.impratio = 10.0

    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # ------------------------------------------------------------ materials
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.24, 0.26, 0.30], rgb2=[0.30, 0.32, 0.36],
                     width=300, height=300)
    gm = spec.add_material(name="grid_mat", texrepeat=[6, 6], reflectance=0.05)
    gm.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"
    spec.add_material(name="wood", rgba=[0.72, 0.58, 0.42, 1], reflectance=0.06)
    spec.add_material(name="cabinet", rgba=[0.86, 0.86, 0.88, 1], reflectance=0.05)
    spec.add_material(name="porcelain", rgba=[0.95, 0.95, 0.97, 1], reflectance=0.25)
    spec.add_material(name="steel", rgba=[0.72, 0.74, 0.78, 1], reflectance=0.40)
    spec.add_material(name="glass", rgba=[0.55, 0.72, 0.85, 0.75], reflectance=0.30)

    # ------------------------------------------------------------ world
    w = spec.worldbody
    w.add_light(name="key", pos=[0.0, -0.6, 2.0], dir=[0, 0.35, -1],
                diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.8, 0.6, 1.8], dir=[0.4, -0.3, -1],
                diffuse=[0.35, 0.35, 0.35])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[3, 3, 0.05], material="grid_mat")

    # table -------------------------------------------------------------
    t = w.add_body(name="table", pos=[0, 0, 0])
    box(t, "table_top", [0, 0, TABLE_TOP_Z - TABLE_HALF[2]], TABLE_HALF,
        "wood", **SLIDE)
    leg_h = (TABLE_TOP_Z - 2 * TABLE_HALF[2]) / 2.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(t, f"leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                [sx * (TABLE_HALF[0] - 0.04), sy * (TABLE_HALF[1] - 0.04), leg_h],
                [0.02, 0.02, leg_h], "wood")

    # drawer cabinet ----------------------------------------------------
    cab = w.add_body(name="cabinet", pos=[0.0, CAB_Y, CAB_Z])
    box(cab, "cab_back", [0, 0.075, CAB_H], [0.11, 0.006, CAB_H], "cabinet")
    box(cab, "cab_left", [-0.104, 0.0, CAB_H], [0.006, 0.08, CAB_H], "cabinet")
    box(cab, "cab_right", [0.104, 0.0, CAB_H], [0.006, 0.08, CAB_H], "cabinet")
    box(cab, "cab_top", [0, 0.0, 2 * CAB_H + 0.006], [0.11, 0.08, 0.006], "cabinet")

    dr = cab.add_body(name="drawer", pos=[0.0, 0.0, 0.045])
    dr.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                 axis=[0, -1, 0], range=[0.0, DRAWER_TRAVEL], damping=6.0,
                 frictionloss=0.35, armature=0.02)
    box(dr, "drawer_floor", [0, 0, -0.018], [0.094, 0.072, 0.004], "cabinet", **SLIDE)
    box(dr, "drawer_back", [0, 0.068, 0.0], [0.094, 0.004, 0.018], "cabinet")
    box(dr, "drawer_side_l", [-0.090, 0.0, 0.0], [0.004, 0.072, 0.018], "cabinet")
    box(dr, "drawer_side_r", [0.090, 0.0, 0.0], [0.004, 0.072, 0.018], "cabinet")
    box(dr, "drawer_front", [0, -0.073, 0.004], [0.098, 0.005, 0.028], "cabinet")
    dr.add_geom(name="drawer_handle", type=CAP,
                fromto=[-0.035, -0.087, 0.006, 0.035, -0.087, 0.006],
                size=[0.005, 0, 0], material="steel", **GRASP)
    dr.add_site(name="drawer_handle_site", pos=[0.0, -0.087, 0.006], size=[0.004, 0, 0])

    # cutlery, resting inside the drawer --------------------------------
    # Laid ACROSS the drawer rather than along it.  Lengthwise they have to sit
    # near the drawer face, and there the SO-101's wrist camera mount fouls the
    # face on the way down and the arm stalls with its servos saturated --
    # measured, not assumed.  Across, they clear it by 86 mm and still fit the
    # 172 mm interior (longest randomized cutlery spans 132 mm).
    for nm, off in (("spoon", -0.032), ("fork", 0.018)):
        b = free_body(spec, nm, [0.0, CAB_Y + off, CAB_Z + 0.034])
        b.quat = zquat(90.0)
        b.add_geom(name=f"{nm}_handle", type=BOX, pos=[0, 0, 0],
                   size=[0.006, D["cutlery_l"], 0.0025], material="steel",
                   mass=D["cutlery_m"], **GRASP)
        b.add_geom(name=f"{nm}_head", type=BOX, pos=[0, D["cutlery_l"] + 0.010, 0.0],
                   size=[0.011, 0.012, 0.002], material="steel",
                   mass=D["cutlery_m"] * 0.5, **GRASP)
        b.add_site(name=f"{nm}_grasp", pos=[0, 0, 0], size=[0.004, 0, 0])

    # plate -------------------------------------------------------------
    plate = free_body(spec, "plate", [0.25, 0.115, TABLE_TOP_Z + D["plate_h"] + 0.005])
    plate.add_geom(name="plate_base", type=CYL, pos=[0, 0, 0],
                   size=[D["plate_r"], D["plate_h"], 0], material="porcelain",
                   mass=D["plate_m"], **GRASP)
    rim_r = D["plate_r"] - 0.003
    for i in range(12):                       # rim, so the jaw has an edge
        a = 2 * math.pi * i / 12
        plate.add_geom(name=f"plate_rim_{i}", type=BOX,
                       pos=[rim_r * math.cos(a), rim_r * math.sin(a), D["plate_h"] + 0.001],
                       quat=zquat(math.degrees(a)),
                       size=[0.006, 0.014, 0.004], material="porcelain",
                       mass=0.002, **GRASP)
    plate.add_site(name="plate_grasp", pos=[rim_r, 0.0, D["plate_h"] + 0.001],
                   size=[0.004, 0, 0])

    # mug ---------------------------------------------------------------
    mug = free_body(spec, "mug", [-0.25, 0.115, TABLE_TOP_Z + D["mug_h"] + 0.003])
    mug.add_geom(name="mug_wall", type=CYL, pos=[0, 0, 0],
                 size=[D["mug_r"], D["mug_h"], 0], material="porcelain",
                 mass=D["mug_m"], **GRASP)
    mug.add_geom(name="mug_handle", type=CAP,
                 fromto=[D["mug_r"] + 0.002, 0, -0.012,
                         D["mug_r"] + 0.002, 0, 0.012], size=[0.005, 0, 0],
                 material="porcelain", mass=0.010, **GRASP)
    mug.add_site(name="mug_grasp", pos=[0.0, 0.0, D["mug_h"] - 0.005],
                 size=[0.004, 0, 0])

    # bottle ------------------------------------------------------------
    bottle = free_body(spec, "bottle", [0.35, -0.02, TABLE_TOP_Z + D["bottle_h"] + 0.004])
    bottle.add_geom(name="bottle_body", type=CYL, pos=[0, 0, 0],
                    size=[D["bottle_r"], D["bottle_h"], 0], material="glass",
                    mass=D["bottle_m"], **GRASP)
    neck_z = D["bottle_h"] + 0.018
    bottle.add_geom(name="bottle_neck", type=CYL, pos=[0, 0, neck_z],
                    size=[0.011, 0.020, 0], material="glass", mass=0.020, **GRASP)
    bottle.add_site(name="bottle_grasp", pos=[0, 0, neck_z], size=[0.004, 0, 0])
    bottle.add_site(name="bottle_spout", pos=[0, 0, neck_z + 0.022], size=[0.004, 0, 0])

    # placement targets the task is scored against ----------------------
    for nm, (x, y) in (("target_plate", (0.000, -0.045)),
                       ("target_fork", (-0.085, -0.045)),
                       ("target_spoon", (0.085, -0.045)),
                       ("target_mug", (0.140, -0.010))):
        w.add_site(name=nm, type=BOX, pos=[x, y, TABLE_TOP_Z + 0.001],
                   size=[0.012, 0.012, 0.0005], rgba=[0.2, 0.8, 0.3, 0.35])
    w.add_site(name="handoff", pos=[HANDOFF[0], HANDOFF[1], TABLE_TOP_Z + 0.13],
               size=[0.010, 0, 0], rgba=[0.9, 0.6, 0.1, 0.30])

    # cameras -----------------------------------------------------------
    w.add_camera(name="scene_cam", pos=[0.0, -0.78, 1.28],
                 xyaxes=[1, 0, 0, 0, 0.62, 0.78], fovy=50)
    w.add_camera(name="front_cam", pos=[0.0, -0.62, 0.96],
                 xyaxes=[1, 0, 0, 0, 0.42, 0.91], fovy=52)
    w.add_camera(name="top_cam", pos=[0.0, 0.02, 1.42],
                 xyaxes=[1, 0, 0, 0, 1, 0], fovy=48)

    # ------------------------------------------------------------ arms
    # At pan=0 an SO-101 points along its own +x, so each mount is yawed to
    # aim the neutral pose at the shared workspace in front of the drawer.
    for prefix, x, yaw in (("left_", -ARM_X, ARM_YAW), ("right_", ARM_X, 180.0 - ARM_YAW)):
        child = mujoco.MjSpec.from_file(str(SO101))
        frame = w.add_frame(pos=[x, ARM_Y, TABLE_TOP_Z], quat=zquat(yaw))
        spec.attach(child, prefix=prefix, frame=frame)

    return spec


def home_keyframe(model, data):
    """Solve a symmetric ready pose and return (qpos, ctrl) for keyframe 'home'."""
    import numpy as np
    from .ik import site_ik, ARM_JOINTS

    mujoco.mj_resetData(model, data)
    errs = {}
    for prefix, sx in (("left_", -1.0), ("right_", 1.0)):
        target = np.array([sx * HOME_XY, 0.02, TABLE_TOP_Z + HOME_Z])
        _, errs[prefix] = site_ik(model, data, prefix, "gripperframe", target)

    qpos = model.qpos0.copy()
    for prefix in ("left_", "right_"):
        for j in ARM_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
            qpos[model.jnt_qposadr[jid]] = data.qpos[model.jnt_qposadr[jid]]
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "gripper")
        qpos[model.jnt_qposadr[gid]] = GRIPPER_OPEN

    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl[i] = qpos[model.jnt_qposadr[jid]]
    return qpos, ctrl, errs
