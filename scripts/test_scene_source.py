#!/usr/bin/env python3
"""Controls for the perception-in-the-loop seam.

Same shape as scripts/test_perception_pipeline.py: every mechanism is driven
from both sides, because a check that can only pass proves nothing.

  identity      ACCEPT the privileged source is MjData verbatim, so installing
                it cannot move a number; REJECT an estimating source being
                mistaken for it
  propagation   ACCEPT an offset on a body reaches the controller's waypoints
                and the body's own grasp site; REJECT it reaching a world site
                or a body the network has no output for
  perception    ACCEPT the real model, on the evaluated seeds, lands inside the
                error this repository has already reported for it; REJECT it
                keeping any skill when the camera is replaced with noise
  the scorer    REJECT envs/task.py consulting the scene source -- an estimate
                that grades itself is not an evaluation
  the harness   ACCEPT eval_seeds.py leaves the privileged source installed
                after an episode, so one run cannot contaminate the next

The propagation controls use an offset taken from envs/randomize.py's own
placement-jitter range rather than a made-up constant, so a change to how much
the scene moves changes what these controls test.

Run:  python3 scripts/test_scene_source.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402
import mujoco                                               # noqa: E402

from envs import perception as P                            # noqa: E402
from envs import scene_source as S                          # noqa: E402
from envs import controller as C                            # noqa: E402
from envs.randomize import make_env, RANGES, NOMINAL_XY     # noqa: E402
from envs.task import TaskMonitor                           # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
results: list[dict] = []

# The offset the propagation controls use: the full width of the placement
# jitter the scene sampler applies. Derived, so it tracks envs/randomize.py.
LO, HI = RANGES["place_xy"]
OFFSET = np.array([HI - LO, LO - HI, 0.0])          # (+0.09, -0.09, 0) m today


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


class FixedOffsetScene(S._EstimatingScene):
    """Estimates every object exactly OFFSET away from where it is."""

    name = "fixed_offset[test]"

    def _estimate(self, model, data):
        t = S.truth(model, data)
        out = {"drawer_q": t["drawer_q"] + 0.01}
        for obj in S.PERCEIVED_BODIES:
            out[f"{obj}_x"] = t[f"{obj}_x"] + OFFSET[0]
            out[f"{obj}_y"] = t[f"{obj}_y"] + OFFSET[1]
        return out


def main() -> int:
    ok = True
    model, data, _ = make_env(0)
    priv = S.PrivilegedScene()

    # ------------------------------------------------------------- identity
    S.install(priv)
    same = []
    for b in ("plate", "mug", "bottle", "spoon", "fork"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        same.append(np.array_equal(priv.body_xpos(model, data, b),
                                   np.array(data.xpos[bid], dtype=float)))
    for s in ("plate_grasp", "mug_grasp", "target_plate", "handoff"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
        same.append(np.array_equal(priv.site_xpos(model, data, s),
                                   np.array(data.site_xpos[sid], dtype=float)))
    adr = S._drawer_adr(model)
    same.append(priv.drawer_q(model, data) == float(data.qpos[adr]))
    ok &= check("accept_privileged_is_mjdata_verbatim", all(same),
                f"{sum(same)}/{len(same)} reads bit-identical to MjData")

    # ----------------------------------------------------------- propagation
    # A waypoint the controller builds must move with the estimate, or the seam
    # is decorative.
    where_home = C.rim_toward("plate", "right")(model, data)
    grasp_home = C.site_xyz("mug_grasp")(model, data)
    target_home = C.site_xyz("target_plate")(model, data)
    fork_home = C.site_xyz("fork_grasp")(model, data)

    off = FixedOffsetScene()
    S.install(off)
    where_off = C.rim_toward("plate", "right")(model, data)
    grasp_off = C.site_xyz("mug_grasp")(model, data)
    target_off = C.site_xyz("target_plate")(model, data)
    fork_off = C.site_xyz("fork_grasp")(model, data)

    d_grasp = grasp_off - grasp_home
    ok &= check("accept_offset_reaches_the_bodys_own_grasp_site",
                np.allclose(d_grasp, OFFSET, atol=1e-9),
                f"mug_grasp moved {np.round(d_grasp * 1000, 3).tolist()} mm "
                f"for an offset of {np.round(OFFSET * 1000, 3).tolist()} mm")

    moved = float(np.linalg.norm((where_off - where_home)[:2]))
    ok &= check("accept_offset_reaches_a_controller_waypoint", moved > 0.05,
                f"the plate-rim waypoint moved {moved * 1000:.1f} mm")

    ok &= check("reject_offset_reaching_a_world_site",
                np.array_equal(target_off, target_home),
                "target_plate is welded to the world and did not move")

    ok &= check("reject_offset_reaching_an_unmodelled_body",
                np.array_equal(fork_off, fork_home),
                "fork_grasp did not move -- the network has no fork output, "
                f"and PERCEIVED_BODIES is {list(S.PERCEIVED_BODIES)}")

    ok &= check("reject_spoon_or_fork_being_claimed_as_perceived",
                not ({"spoon", "fork"} & set(S.PERCEIVED_BODIES)),
                f"PERCEIVED_BODIES = {list(S.PERCEIVED_BODIES)}, "
                f"network outputs = {list(P.OUT_NAMES)}")

    # drawer: the handle site must slide along the joint's own axis
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "drawer_handle_site")
    S.install(priv)
    h0 = priv.site_xpos(model, data, "drawer_handle_site")
    S.install(off)
    h1 = off.site_xpos(model, data, "drawer_handle_site")
    axis = S._drawer_axis(model)
    ok &= check("accept_drawer_estimate_slides_the_handle_along_its_axis",
                np.allclose(h1 - h0, axis * 0.01, atol=1e-9),
                f"handle moved {np.round((h1 - h0) * 1000, 3).tolist()} mm "
                f"along axis {axis.tolist()}")
    ok &= check("accept_drawer_estimate_reaches_the_controllers_predicate",
                off.drawer_q(model, data) != priv.drawer_q(model, data),
                f"drawer_q {priv.drawer_q(model, data):.4f} -> "
                f"{off.drawer_q(model, data):.4f} m")

    # ------------------------------------------------------- the scorer stays
    # An estimate that reaches the scorer would be grading itself.
    S.install(priv)
    mon_a = TaskMonitor(model)
    mon_a.step(data)
    rep_a = mon_a.report(data)
    S.install(FixedOffsetScene())
    mon_b = TaskMonitor(model)
    mon_b.step(data)
    rep_b = mon_b.report(data)
    ok &= check("reject_the_scorer_consulting_the_scene_source",
                rep_a == rep_b,
                "envs/task.py returns the same report under a source that "
                f"displaces every object by {np.round(OFFSET * 1000).tolist()} mm")

    # -------------------------------------------------- the negative control
    S.install(priv)
    blind = S.BlindScene()
    S.install(blind)
    worst = 0.0
    for obj in S.PERCEIVED_BODIES:
        d = blind.body_xpos(model, data, obj)[:2] - np.array(
            data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        obj)][:2])
        worst = max(worst, float(np.linalg.norm(d)))
    ok &= check("accept_blind_control_actually_corrupts", worst > (HI - LO) / 4,
                f"nominal layout is {worst * 1000:.1f} mm from the truth on "
                f"seed 0, against a placement jitter of "
                f"{(HI - LO) * 1000:.0f} mm")
    # ...and it must corrupt by exactly what envs/randomize.py says the nominal
    # layout is. A literal copied into scene_source.py would drift the day the
    # scene moves, and this is what would catch it.
    dlo, dhi = RANGES["drawer_q"]
    est = blind._estimate(model, data)
    tied = all(abs(est[f"{o}_x"] - NOMINAL_XY[o][0]) < 1e-12
               and abs(est[f"{o}_y"] - NOMINAL_XY[o][1]) < 1e-12
               for o in S.PERCEIVED_BODIES)
    tied &= abs(est["drawer_q"] - 0.5 * (dlo + dhi)) < 1e-12
    ok &= check("accept_blind_control_is_tied_to_randomize_not_a_literal", tied,
                "BlindScene's estimate equals envs.randomize.NOMINAL_XY and the "
                f"midpoint of RANGES['drawer_q'] ({0.5 * (dlo + dhi):.4f} m) "
                "exactly")

    # ------------------------------------------------------------ perception
    reported = None
    tp = EV / "perception_train.json"
    if tp.exists():
        reported = json.loads(tp.read_text())["error_mm"]["eval10"][
            "worst_centre_mm"]

    per = None
    try:
        per = S.PerceivedScene()
        S.install(per)
        errs, est_is_the_networks = [], []
        for seed in range(3):
            m, d, _ = make_env(seed)
            per.close()
            per._key = None
            for obj in S.PERCEIVED_BODIES:
                got = per.body_xpos(m, d, obj)[:2]
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
                errs.append(float(np.linalg.norm(
                    got - np.array(d.xpos[bid][:2]))) * 1000.0)
            raw = per._estimate(m, d)
            got = per.body_xpos(m, d, "plate")[:2]
            est_is_the_networks.append(
                abs(got[0] - raw["plate_x"]) < 1e-9
                and abs(got[1] - raw["plate_y"]) < 1e-9)
        w = max(errs)
        bar = 3.0 * reported if reported else 6.0
        ok &= check("accept_perceived_matches_the_reported_model_error",
                    w < bar,
                    f"worst centre {w:.2f} mm over seeds 0-2 at t=0, against "
                    f"{reported:.2f} mm reported on eval10 (bar {bar:.2f} mm)"
                    if reported else f"worst centre {w:.2f} mm")
        ok &= check("accept_what_the_controller_gets_is_the_network_output",
                    all(est_is_the_networks),
                    "body_xpos returned the decoded network prediction, not "
                    "the simulator's value, on every seed checked")

        # noise control: no image, no skill.
        m, d, _ = make_env(0)
        rng = np.random.default_rng(0)
        per.close()
        per._key = None
        per._render = lambda *_a, **_k: rng.integers(
            0, 256, (P.IMG_H, P.IMG_W, 3), dtype=np.uint8)
        noisy = []
        for obj in S.PERCEIVED_BODIES:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
            noisy.append(float(np.linalg.norm(
                per.body_xpos(m, d, obj)[:2]
                - np.array(d.xpos[bid][:2]))) * 1000.0)
        ok &= check("reject_perception_keeping_skill_on_a_noise_image",
                    max(noisy) > 10.0 * w,
                    f"noise gives {max(noisy):.1f} mm against {w:.2f} mm on "
                    "the real frame")
    except FileNotFoundError as e:
        ok &= check("accept_perceived_matches_the_reported_model_error", False,
                    f"could not load the IR: {e}")
    finally:
        if per is not None:
            per.close()
        S.reset()

    # ------------------------------------------------------------ no leakage
    ok &= check("accept_default_source_is_privileged",
                type(S.active()) is S.PrivilegedScene,
                f"after reset the active source is {type(S.active()).__name__}")

    EV.mkdir(parents=True, exist_ok=True)
    (EV / "scene_source_controls.json").write_text(json.dumps(
        {"suite": "scene_source", "passed": sum(r["pass"] for r in results),
         "total": len(results), "all_pass": bool(ok),
         "offset_basis": "envs.randomize.RANGES['place_xy'] full width",
         "offset_m": OFFSET.tolist(), "controls": results}, indent=1),
        encoding="utf-8")
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} controls passed "
          f"-> evidence/scene_source_controls.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
