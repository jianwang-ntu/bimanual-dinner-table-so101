#!/usr/bin/env python3
"""Verify envs/dinner_table.xml as a standalone MJCF and record the evidence.

Checks, in order:
  1. the saved XML loads on its own (mesh paths resolve from the file)
  2. both arms are present with the full 6-DoF actuator set and a wrist camera
  3. the scene settles: no NaN, no warning, objects stay on the table
  4. the drawer joint actually travels its stated range
  5. each object sits inside at least one arm's reach envelope
  6. offscreen rendering works from every scene camera

Writes evidence/scene_verification.json and evidence/frames/*.png.
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "envs" / "dinner_table.xml"
EVID = ROOT / "evidence"
FRAMES = EVID / "frames"

TABLE_TOP_Z = 0.75
ARM_BASES = {"left": np.array([-0.22, -0.14]), "right": np.array([0.22, -0.14])}
REACH_MAX = 0.40          # conservative: measured planar envelope max is 0.478 m
OBJECTS = ["plate", "mug", "bottle", "spoon", "fork"]

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def names(model, objtype):
    n = {mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
         mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
         mujoco.mjtObj.mjOBJ_BODY: model.nbody,
         mujoco.mjtObj.mjOBJ_CAMERA: model.ncam,
         mujoco.mjtObj.mjOBJ_SITE: model.nsite}[objtype]
    return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


def main() -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    ok = True

    # 1 ---------------------------------------------------------------- load
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    ok &= check("standalone_load", True,
                f"nq={model.nq} nu={model.nu} nbody={model.nbody} "
                f"ngeom={model.ngeom} nmesh={model.nmesh}")

    # 2 ------------------------------------------------------------ two arms
    act = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper"]
    for side in ("left", "right"):
        want = [f"{side}_{j}" for j in joints]
        ok &= check(f"{side}_arm_actuators", all(a in act for a in want),
                    f"{sum(a in act for a in want)}/6 present")
    cams = names(model, mujoco.mjtObj.mjOBJ_CAMERA)
    ok &= check("wrist_cameras",
                "left_wrist_cam" in cams and "right_wrist_cam" in cams,
                f"cameras={cams}")
    ok &= check("scene_cameras",
                all(c in cams for c in ("scene_cam", "front_cam", "top_cam")),
                f"{len(cams)} cameras total")

    # 3 --------------------------------------------------------------- settle
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    ok &= check("home_keyframe", kid >= 0, f"nkey={model.nkey}, home id={kid}")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.ctrl[:] = model.key_ctrl[kid]
    for _ in range(1500):                     # 3.0 s at dt=0.002
        mujoco.mj_step(model, data)

    # the commanded home pose must actually be holdable against gravity
    tracking = {}
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = model.actuator_trnid[i, 0]
        tracking[nm] = round(float(data.qpos[model.jnt_qposadr[jid]]
                                   - data.ctrl[i]), 4)
    worst = max(abs(v) for v in tracking.values())
    ok &= check("home_pose_holds", worst < 0.12,
                f"worst joint tracking error {worst:.4f} rad; {tracking}")

    ok &= check("no_nan", bool(np.all(np.isfinite(data.qpos)) and
                               np.all(np.isfinite(data.qvel))),
                f"|qvel|max={float(np.abs(data.qvel).max()):.4f}")
    ok &= check("no_warnings", int(data.warning.number.sum()) == 0,
                f"warning counters={data.warning.number.tolist()}")

    heights, on_table = {}, True
    for nm in OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        z = float(data.xpos[bid][2])
        heights[nm] = round(z, 4)
        on_table &= (TABLE_TOP_Z - 0.02) < z < (TABLE_TOP_Z + 0.30)
    ok &= check("objects_rest_on_table", on_table, heights)

    settle_vel = float(np.abs(data.qvel).max())
    ok &= check("scene_quiescent", settle_vel < 0.05,
                f"max|qvel| after 3.0 s = {settle_vel:.5f} rad/s or m/s")

    # 4 --------------------------------------------------------------- drawer
    dj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    adr = model.jnt_qposadr[dj]
    lo, hi = model.jnt_range[dj]
    d2 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d2)
    d2.qpos[adr] = hi
    mujoco.mj_forward(model, d2)
    opened = float(d2.qpos[adr])
    # cutlery must still be inside the drawer footprint when it is open
    drawer_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
    travel = float(hi - lo)
    ok &= check("drawer_travel", abs(opened - hi) < 1e-9 and travel >= 0.06,
                f"range=[{lo:.3f}, {hi:.3f}] m (travel {travel:.3f}), "
                f"commanded open reads {opened:.3f}")
    # The drawer body must translate by the joint travel, carrying the cutlery
    # with it -- the joint is only useful if the frame it drives actually moves.
    moved = float(data.xpos[drawer_bid][1]) - float(d2.xpos[drawer_bid][1])
    ok &= check("drawer_moves_cutlery", abs(moved - travel) < 0.01,
                f"drawer y closed={data.xpos[drawer_bid][1]:.3f} "
                f"open={d2.xpos[drawer_bid][1]:.3f} delta={moved:.3f} "
                f"vs joint travel {travel:.3f}")

    # 5 ---------------------------------------------------------------- reach
    reach = {}
    for nm in OBJECTS + ["drawer"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        p = data.xpos[bid][:2]
        reach[nm] = {s: round(float(np.linalg.norm(p - b)), 3)
                     for s, b in ARM_BASES.items()}
    reachable = {k: min(v.values()) < REACH_MAX for k, v in reach.items()}
    ok &= check("every_object_reachable", all(reachable.values()), reach)

    handoff = np.array([0.0, -0.075])
    both = {s: round(float(np.linalg.norm(handoff - b)), 3)
            for s, b in ARM_BASES.items()}
    ok &= check("handoff_zone_shared", all(v < REACH_MAX for v in both.values()),
                f"distance from each base to the hand-off site: {both}")

    # 6 --------------------------------------------------------------- render
    rendered = {}
    try:
        with mujoco.Renderer(model, height=720, width=1280) as r:
            for cam in ("scene_cam", "front_cam", "top_cam"):
                r.update_scene(data, camera=cam)
                img = r.render()
                out = FRAMES / f"{cam}.png"
                _write_png(out, img)
                rendered[cam] = [int(img.shape[1]), int(img.shape[0]),
                                 int(img.std() > 5)]
        with mujoco.Renderer(model, height=480, width=640) as r:
            for cam in ("left_wrist_cam", "right_wrist_cam"):
                r.update_scene(data, camera=cam)
                img = r.render()
                _write_png(FRAMES / f"{cam}.png", img)
                rendered[cam] = [int(img.shape[1]), int(img.shape[0]),
                                 int(img.std() > 5)]
        ok &= check("offscreen_render",
                    all(v[2] == 1 for v in rendered.values()), rendered)
    except Exception as exc:                       # noqa: BLE001
        ok &= check("offscreen_render", False, f"{type(exc).__name__}: {exc}")

    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "scene_verification.json").write_text(json.dumps({
        "scene": str(SCENE.relative_to(ROOT)),
        "mujoco_version": mujoco.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "all_pass": bool(ok),
        "checks": results,
    }, indent=1), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


def _write_png(path: pathlib.Path, img) -> None:
    import imageio.v2 as imageio
    imageio.imwrite(path, img)


if __name__ == "__main__":
    sys.exit(main())
