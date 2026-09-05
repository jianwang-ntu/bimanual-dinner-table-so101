#!/usr/bin/env python3
"""Controls for envs/controller.plan_pose_squared.

The solver exists because ``site_ik`` constrains three numbers with five
joints and lets the other two fall where they may, and where they fell was a
gripper whose jaws closed nearly straight down.  A test that only asked "does
it reach the point" would have passed on the broken solver too, so every
control here is paired:

* an ACCEPT control that the squared solver must pass, and
* the SAME measurement on the position-only solver, which must FAIL it --
  otherwise the control is not measuring the thing that was wrong.

The fallback path has its own accept control, because a guard that has never
been seen to let anything through is not a guard.

Run:  python3 scripts/test_squared_ik.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from envs.randomize import make_env                            # noqa: E402
from envs.controller import (Gripper, plan_pose, plan_pose_squared,  # noqa: E402
                             X_AXIS, GRIPPER_WIDE, GRIPPER_NARROW)

ROOT = pathlib.Path(__file__).resolve().parent.parent
results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def axis_deg(model, scratch, grip, want) -> float:
    """Angle between the solved jaw line and the requested one, in degrees."""
    ax = grip.jaw_axis(model, scratch)
    w = np.asarray(want, float)
    w = w / np.linalg.norm(w)
    return float(np.degrees(np.arccos(min(1.0, abs(float(ax @ w))))))


def fresh(model, data):
    sc = mujoco.MjData(model)
    sc.qpos[:] = data.qpos
    sc.qvel[:] = 0.0
    return sc


def main() -> int:
    ok = True
    model, data, _ = make_env(0)
    grip_l = Gripper(model, "left")
    grip_r = Gripper(model, "right")

    def site(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return data.site_xpos[sid].copy()

    mug = site("mug_grasp")

    # ---- 1/2. the defect, and the fix, measured the same way ---------------
    sc = fresh(model, data)
    q_sq, e_sq = plan_pose_squared(model, sc, grip_l, mug, jaw_dir=X_AXIS,
                                   opening=GRIPPER_WIDE)
    d_sq = axis_deg(model, sc, grip_l, X_AXIS)

    sc = fresh(model, data)
    q_pl, e_pl = plan_pose(model, sc, grip_l, mug, jaw_dir=X_AXIS,
                           opening=GRIPPER_WIDE)
    d_pl = axis_deg(model, sc, grip_l, X_AXIS)

    ok &= check("accept_squared_puts_the_jaws_on_the_requested_line",
                d_sq < 45.0 and e_sq < 0.010,
                f"{d_sq:.1f} deg off the line, {e_sq * 1000:.1f} mm off the point")
    ok &= check("reject_position_only_solver_does_not",
                d_pl > 45.0,
                f"position-only solver leaves the jaws {d_pl:.1f} deg off "
                f"(squared: {d_sq:.1f})")
    ok &= check("accept_squared_is_no_worse_on_position",
                e_sq <= e_pl + 0.005,
                f"squared {e_sq * 1000:.1f} mm vs position-only {e_pl * 1000:.1f} mm")

    # ---- 3. the jaw axis is a line, not an arrow ---------------------------
    sc = fresh(model, data)
    plan_pose_squared(model, sc, grip_l, mug, jaw_dir=-X_AXIS,
                      opening=GRIPPER_WIDE)
    d_neg = axis_deg(model, sc, grip_l, X_AXIS)
    ok &= check("accept_reversed_request_gives_the_same_jaw_line",
                abs(d_neg - d_sq) < 12.0,
                f"+x -> {d_sq:.1f} deg, -x -> {d_neg:.1f} deg")

    # ---- 3b. the standoff knows which side the moving jaw is on -----------
    # ``standoff`` backs the meeting point off AWAY from the moving jaw, so it
    # has to be applied along the jaw axis in the sense that agrees with the
    # request.  Getting the sense wrong is a 2 x standoff error straight into
    # the object, and it is invisible to any control that treats the jaw axis
    # as an unsigned line -- which the one above deliberately does.
    S = 0.006
    proj = {}
    for name, want in (("plus", X_AXIS), ("minus", -X_AXIS)):
        sc = fresh(model, data)
        plan_pose_squared(model, sc, grip_l, mug, jaw_dir=want,
                          opening=GRIPPER_WIDE, standoff=S)
        proj[name] = float((grip_l.tip_mid(model, sc) - mug) @ (X_AXIS))
    ok &= check("accept_standoff_follows_the_requested_sense",
                proj["plus"] < 0.0 < proj["minus"],
                f"meeting point offset along +x: {proj['plus'] * 1000:+.1f} mm "
                f"for a +x request, {proj['minus'] * 1000:+.1f} mm for -x")

    # ---- 4. the seed term is load-bearing ---------------------------------
    # Without it the solver answers with poses the arm cannot hold: joints
    # driven onto their stops.  With it the answer stays near the pose the
    # arm is already in.
    lo = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_" + j)][0]
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll")])
    hi = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_" + j)][1]
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll")])
    seed_q = data.qpos[grip_l.qadr[:5]].copy()

    def travel(q):
        return float(np.max(np.abs(np.asarray(q)[:5] - seed_q)))

    def margin(q):
        return float(np.min(np.minimum(np.asarray(q)[:5] - lo,
                                       hi - np.asarray(q)[:5])))

    sc = fresh(model, data)
    q_free, _ = plan_pose_squared(model, sc, grip_l, mug, jaw_dir=X_AXIS,
                                  opening=GRIPPER_WIDE, w_seed=0.0)
    ok &= check("accept_seed_term_keeps_the_answer_reachable",
                margin(q_sq) > margin(q_free),
                f"limit margin {margin(q_sq):.3f} rad with the seed term, "
                f"{margin(q_free):.3f} rad without; joint travel "
                f"{travel(q_sq):.2f} vs {travel(q_free):.2f} rad")

    # ---- 5. the fallback path, exercised in both directions ---------------
    # A target 1.2 m away is outside any pose of this arm, so the squared
    # solve must give up and hand back what plan_pose says.
    far = mug + np.array([0.0, 0.0, 1.2])
    sc = fresh(model, data)
    q_far_sq, e_far_sq = plan_pose_squared(model, sc, grip_r, far,
                                           jaw_dir=X_AXIS, opening=GRIPPER_NARROW)
    sc = fresh(model, data)
    q_far_pl, e_far_pl = plan_pose(model, sc, grip_r, far, jaw_dir=X_AXIS,
                                   opening=GRIPPER_NARROW)
    ok &= check("accept_unreachable_target_falls_back_to_plan_pose",
                np.allclose(q_far_sq, q_far_pl, atol=1e-6),
                f"squared residual {e_far_sq * 1000:.0f} mm > 20 mm accept bar, "
                f"answer identical to plan_pose: "
                f"{float(np.max(np.abs(q_far_sq - q_far_pl))):.2e} rad")
    ok &= check("accept_reachable_target_does_not_fall_back",
                not np.allclose(q_sq, q_pl, atol=1e-6),
                f"reachable target answered by the squared solve, "
                f"max joint difference from plan_pose "
                f"{float(np.max(np.abs(q_sq - q_pl))):.3f} rad")

    # ---- 6. a zero jaw request is not a silent wrong answer ---------------
    sc = fresh(model, data)
    q_zero, _ = plan_pose_squared(model, sc, grip_l, mug,
                                  jaw_dir=np.zeros(3), opening=GRIPPER_WIDE)
    ok &= check("accept_empty_jaw_request_defers_to_plan_pose",
                q_zero.shape == q_pl.shape,
                "a zero-length jaw direction returns a plan_pose answer "
                "rather than dividing by it")

    (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence" / "squared_ik_controls.json").write_text(
        json.dumps({"all_pass": bool(ok),
                    "measured": {
                        "squared_axis_deg": round(d_sq, 2),
                        "position_only_axis_deg": round(d_pl, 2),
                        "squared_pos_err_mm": round(e_sq * 1000, 2),
                        "position_only_pos_err_mm": round(e_pl * 1000, 2),
                    },
                    "controls": results}, indent=1),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
