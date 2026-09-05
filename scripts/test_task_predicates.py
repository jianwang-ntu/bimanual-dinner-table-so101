#!/usr/bin/env python3
"""Controls for envs/task.py: prove every predicate can say yes AND no.

A scorer that only ever returns 0 is indistinguishable from a broken one, and
the 10-seed run scores 0/5 by design because no policy exists yet.  So each
predicate is driven here from both sides: an ACCEPT control that places the
world in the state the predicate is meant to reward, and a REJECT control that
perturbs exactly one thing and must flip it back to false.

Run:  python3 scripts/test_task_predicates.py
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

from envs.task import TaskMonitor, PLACEMENTS, DRAWER_OPEN_M   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "envs" / "dinner_table.xml"

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def load():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    return model, data


def place(model, data, body, xy, z, *, tilt=False):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = [xy[0], xy[1], z]
    data.qpos[adr + 3:adr + 7] = ([0.7071, 0.7071, 0, 0] if tilt
                                  else [1.0, 0.0, 0.0, 0.0])


def solved_state(model, data, *, drawer=0.09):
    """Put the world in the state the task is asking for."""
    dj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    data.qpos[model.jnt_qposadr[dj]] = drawer
    mujoco.mj_forward(model, data)
    for gid, (body, site, _tol) in PLACEMENTS.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        xy = data.site_xpos[sid][:2].copy()
        z = 0.75 + (0.035 if body == "mug" else 0.012)
        place(model, data, body, xy, z)
    mujoco.mj_forward(model, data)


def main() -> int:
    ok = True

    # ------------------------------------------------------------- ACCEPT
    model, data = load()
    mon = TaskMonitor(model)
    solved_state(model, data)
    sg = mon.subgoals(data)
    ok &= check("accept_all_subgoals", all(sg.values()), sg)
    mon.step(data)
    rep = mon.report(data)
    ok &= check("accept_task_success", rep["task_success"] is True,
                f"subgoals_met={rep['subgoals_met']}/{rep['subgoals_total']}, "
                f"in_order_prefix={rep['in_order_prefix']}")

    # ------------------------------------------------------------- REJECT
    for body, gid in (("fork", "fork_placed"), ("spoon", "spoon_placed"),
                      ("plate", "plate_placed"), ("mug", "mug_placed")):
        model, data = load()
        mon = TaskMonitor(model)
        solved_state(model, data)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_free")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] += 0.10                       # 100 mm off target
        mujoco.mj_forward(model, data)
        sg = mon.subgoals(data)
        others = {k: v for k, v in sg.items() if k != gid}
        ok &= check(f"reject_{gid}_displaced",
                    sg[gid] is False and all(others.values()),
                    f"{gid}={sg[gid]}, every other subgoal still true="
                    f"{all(others.values())}")

    # drawer only part-way open
    model, data = load()
    mon = TaskMonitor(model)
    solved_state(model, data, drawer=DRAWER_OPEN_M - 0.02)
    sg = mon.subgoals(data)
    ok &= check("reject_drawer_half_open", sg["drawer_open"] is False,
                f"travel={DRAWER_OPEN_M - 0.02:.3f} m < threshold {DRAWER_OPEN_M} m")

    # mug on its side at the right place -- position alone must not score
    model, data = load()
    mon = TaskMonitor(model)
    solved_state(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_mug")
    place(model, data, "mug", data.site_xpos[sid][:2].copy(), 0.785, tilt=True)
    mujoco.mj_forward(model, data)
    sg = mon.subgoals(data)
    ok &= check("reject_mug_tipped_over", sg["mug_placed"] is False,
                "mug at the target but rotated 90 degrees")

    # an object that fell off the table must not score, even if its x,y align
    model, data = load()
    mon = TaskMonitor(model)
    solved_state(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_plate")
    place(model, data, "plate", data.site_xpos[sid][:2].copy(), 0.30)
    mujoco.mj_forward(model, data)
    mon.step(data)
    sg = mon.subgoals(data)
    ok &= check("reject_plate_on_the_floor",
                sg["plate_placed"] is False and "plate" in mon.dropped,
                f"dropped={sorted(mon.dropped)}")

    # ---------------------------------------------------- hand-off detector
    # Driven through the real contact path: the fork is put against the left
    # jaw pads, stepped, then against the right jaw pads, stepped.  The event
    # is only recorded if MuJoCo actually reports the contacts -- there is no
    # back door into TaskMonitor.handoffs.
    model, data = load()
    mon = TaskMonitor(model)
    seen = []
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"{side}_fixed_jaw_sph_tip1")
        mujoco.mj_forward(model, data)
        p = data.geom_xpos[gid].copy()
        place(model, data, "fork", p[:2], float(p[2]))
        for _ in range(3):
            mujoco.mj_forward(model, data)
            mon.step(data)
        seen.append({side: sorted(mon.touched["fork"])})
    ok &= check("handoff_detector_fires",
                bool(mon.handoffs) and mon.handoffs[0]["from"] == "left"
                and mon.handoffs[0]["to"] == "right",
                f"touched={seen}, handoffs={mon.handoffs}")

    # negative control for the same detector: one arm only must NOT be a hand-off
    model, data = load()
    mon2 = TaskMonitor(model)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                            "left_fixed_jaw_sph_tip1")
    mujoco.mj_forward(model, data)
    p = data.geom_xpos[gid].copy()
    place(model, data, "fork", p[:2], float(p[2]))
    for _ in range(6):
        mujoco.mj_forward(model, data)
        mon2.step(data)
    ok &= check("reject_single_arm_is_not_a_handoff",
                mon2.touched["fork"] == {"left"} and not mon2.handoffs,
                f"touched={sorted(mon2.touched['fork'])}, "
                f"handoffs={mon2.handoffs}")

    (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence" / "task_predicate_controls.json").write_text(
        json.dumps({"all_pass": bool(ok), "controls": results}, indent=1),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
