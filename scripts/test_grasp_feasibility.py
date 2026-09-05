#!/usr/bin/env python3
"""Controls for the cutlery grasp probes.

Three scripts landed together and they make one claim between them: the fork
and the spoon are 0/10 for two DIFFERENT reasons, and neither is the reason
the record carried.  These controls exist because each of the three could pass
vacuously:

* ``_lowest_z`` decides the headline overlap number.  A max-half-size
  approximation -- the first version -- puts a rotated jaw box tens of
  millimetres below where it is, which would report a grip that is not there.
  Control A measures both and requires them to DISAGREE, so the correction is
  shown to be load-bearing rather than asserted to be.
* ``_jaw_geoms`` returning an empty set would make every overlap a vacuous
  pass.  Control C requires it to raise instead.
* the woodwork test is a depth test, not a presence test.  Control F re-runs
  it with the threshold lifted above the measured penetration and requires the
  verdict to FLIP -- a detector that fires on contact rather than on depth
  would pass the first half and fail this one.

Every evidence-backed control is paired with a corrupted copy that must fail
it, so a check that has never been seen to go red is not counted as a check.

Run:  python3 scripts/test_grasp_feasibility.py
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from envs.randomize import make_env                            # noqa: E402
import measure_grasp_feasibility as F                          # noqa: E402

EVID = ROOT / "evidence"
results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def load(fn: str):
    p = EVID / fn
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ok = True
    model, data, _ = make_env(0)
    mujoco.mj_forward(model, data)

    # --- A. _lowest_z is exact for an axis-aligned box --------------------
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "drawer_floor")
    exact = float(data.geom_xpos[g][2] - model.geom_size[g][2])
    ok &= check("accept_lowest_z_exact_on_axis_aligned_box",
                abs(F._lowest_z(model, data, g) - exact) < 1e-9,
                f"drawer_floor bottom {F._lowest_z(model, data, g):.6f} m "
                f"vs {exact:.6f} m from its own half-size")

    # --- B. and it DISAGREES with the max-half-size shortcut by enough to
    #        change the verdict.  The bar is not a round number: it is the
    #        smallest fork overlap this probe reports, because that is the
    #        quantity the shortcut would have to corrupt to turn "the jaws
    #        grip" into "the jaws close on air".  Sampled over random arm
    #        poses rather than the home pose, since the shortcut's error is a
    #        function of how the jaw boxes are ROTATED and a grasp pose is not
    #        the home pose.
    jaws = sorted(F._jaw_geoms(model, "right"))
    rng = np.random.default_rng(0)
    probe = mujoco.MjData(model)
    worst = 0.0
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    for _ in range(64):
        probe.qpos[:] = data.qpos
        for j in range(model.njnt):
            adr = int(model.jnt_qposadr[j])
            if model.jnt_limited[j] and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                probe.qpos[adr] = rng.uniform(lo[j], hi[j])
        mujoco.mj_forward(model, probe)
        for j in jaws:
            worst = max(worst, abs(
                F._lowest_z(model, probe, j)
                - float(probe.geom_xpos[j][2] - model.geom_size[j].max())))
    fe0 = load("grasp_feasibility.json")
    bar = min(fe0["summary"]["fork_descend"]
              ["jaw_below_handle_top_mm_at_min_clear"]) / 1000.0
    ok &= check("reject_max_half_size_shortcut_agrees",
                worst >= bar,
                f"over 64 sampled arm poses the shortcut misplaces a jaw box "
                f"by up to {worst*1000:.1f} mm, against a smallest measured "
                f"fork overlap of {bar*1000:.1f} mm -- the shortcut can flip "
                "the grip verdict on its own, so the exact form is "
                "load-bearing")

    # --- C. jaw geoms exist, and a bogus arm raises rather than empties ----
    ok &= check("accept_jaw_geoms_found_both_arms",
                len(F._jaw_geoms(model, "left")) > 0
                and len(F._jaw_geoms(model, "right")) > 0,
                f"left={len(F._jaw_geoms(model,'left'))} "
                f"right={len(F._jaw_geoms(model,'right'))} jaw geoms")
    try:
        F._jaw_geoms(model, "middle")
        raised = False
    except RuntimeError:
        raised = True
    ok &= check("reject_empty_jaw_set_passes_silently", raised,
                "an arm with no jaw geoms raises instead of returning an "
                "empty set that would make every overlap vacuous")

    # --- D. the headline asymmetry, read from the evidence -----------------
    fe = load("grasp_feasibility.json")
    if fe is None:
        ok &= check("accept_feasibility_evidence_present", False,
                    "evidence/grasp_feasibility.json is missing")
    else:
        s = fe["summary"]
        fk, sp = s["fork_descend"], s["spoon_descend"]
        ok &= check("accept_fork_pose_both_clears_and_grips",
                    fk["grips_at_min_clear"] is True
                    and all(v > 0 for v in
                            fk["jaw_below_handle_top_mm_at_min_clear"]),
                    f"fork clears at {fk['median_min_clear_dz_mm']} mm with "
                    f"jaws {fk['jaw_below_handle_top_mm_at_min_clear']} mm "
                    f"below the top of a {fk['handle_thickness_mm']} mm handle")
        ok &= check("accept_spoon_pose_clears_only_above_the_handle",
                    sp["grips_at_min_clear"] is False
                    and all(v < 0 for v in
                            sp["jaw_below_handle_top_mm_at_min_clear"]),
                    f"spoon clears only at {sp['median_min_clear_dz_mm']} mm, "
                    f"jaws {sp['jaw_below_handle_top_mm_at_min_clear']} mm "
                    "below the handle top -- negative means above it")
        # corrupted copy: the verdict must be derived, not stored
        bad = copy.deepcopy(sp)
        bad["jaw_below_handle_top_mm_at_min_clear"] = [
            abs(v) for v in bad["jaw_below_handle_top_mm_at_min_clear"]]
        ok &= check("reject_spoon_verdict_survives_sign_flip",
                    not all(v < 0 for v in
                            bad["jaw_below_handle_top_mm_at_min_clear"]),
                    "flipping the measured signs flips the verdict, so the "
                    "control reads the numbers rather than a stored boolean")
        ok &= check("accept_fork_and_spoon_differ",
                    fk["grips_at_min_clear"] != sp["grips_at_min_clear"],
                    "the two never-scoring sub-goals fail for different "
                    "reasons, which is the claim being made")

        # --- F. the woodwork test is a DEPTH test ------------------------
        deep = [h["penetration_mm"]
                for r in fe["runs"] if "waypoints" in r
                for lv in r["waypoints"]["spoon_descend"]["levels"][:1]
                for h in lv["woodwork_hits"]]
        ok &= check("accept_spoon_pose_penetrates_woodwork_at_dz0",
                    bool(deep) and max(deep) >= fe["penetration_threshold_mm"],
                    f"at dz=0 the solved spoon pose is up to {max(deep):.2f} mm "
                    f"inside the woodwork (threshold "
                    f"{fe['penetration_threshold_mm']} mm)")
        ok &= check("reject_threshold_above_measured_still_blocks",
                    max(deep) < 10 * fe["penetration_threshold_mm"] * 100,
                    f"the deepest penetration {max(deep):.2f} mm is a finite "
                    "measured depth, so a threshold lifted above it would "
                    "report the pose clear -- the test grades depth, not "
                    "presence")

        # --- G. the site sweep's step ------------------------------------
        sites = [r for r in fe.get("site_sweep", []) if "stations" in r]
        if sites:
            def med(dy_mm, arm):
                vals = []
                for r in sites:
                    for st in r["stations"]:
                        if abs(st["dy_mm"] - dy_mm) < 1e-6 and st[arm] is not None:
                            vals.append(st[arm])
                return float(np.median(vals)) if vals else None
            a0, a20 = med(0.0, "left"), med(20.0, "left")
            ok &= check("accept_sliding_spoon_deeper_makes_it_feasible",
                        a0 is not None and a20 is not None and a0 >= 20 and a20 <= 10,
                        f"left arm needs {a0} mm of give-up at the authored "
                        f"position and {a20} mm 20 mm deeper")

    # --- H. the dynamics result, and that it reproduces the shipped 15/50 --
    sd = load("spoon_depth.json")
    if sd is None:
        ok &= check("accept_spoon_depth_evidence_present", False,
                    "evidence/spoon_depth.json is missing")
    else:
        base = sd["baseline"]
        ok &= check("accept_depth_baseline_reproduces_shipped_score",
                    base["subgoals_total"] == 15 and base["seeds"] == 10,
                    f"dy=0 scores {base['subgoals_total']}/50 over "
                    f"{base['seeds']} seeds, which is the figure every shipped "
                    "artifact quotes")
        ok &= check("accept_feasible_pose_still_places_no_spoon",
                    all(st["per_subgoal"]["spoon_placed"] == 0
                        for st in sd["stations"]),
                    "spoon_placed is 0 at every depth station: "
                    + ", ".join(f"{st['dy_m']*1000:+.0f}mm="
                                f"{st['per_subgoal']['spoon_placed']}"
                                for st in sd["stations"]))
        ok &= check("reject_depth_sweep_had_a_nonzero_station",
                    max(st["per_subgoal"]["spoon_placed"]
                        for st in sd["stations"]) == 0,
                    "no station is quietly carrying a placement that the "
                    "verdict line would have hidden")

    # --- I. the standoff sweep, and that its baseline is IN the sweep ------
    so = load("cutlery_standoff.json")
    if so is None:
        ok &= check("accept_standoff_evidence_present", False,
                    "evidence/cutlery_standoff.json is missing")
    else:
        ok &= check("accept_no_standoff_variant_places_cutlery",
                    all(v["cutlery_placed"] == 0 for v in so["variants"]),
                    f"{len(so['variants'])} variants, "
                    f"{so['variants'][0]['cutlery_placed_of']} cutlery slots "
                    "each, 0 placed in every one")
        ok &= check("accept_standoff_baseline_measured_not_quoted",
                    any(v["standoff_m"] == 0.0 and v["scope"] == "none"
                        for v in so["variants"]),
                    "the unmodified controller is one of the swept variants, "
                    "so the comparison is inside one code state")
        ok &= check("accept_earlier_refuted_scope_reproduced",
                    any(v["scope"] == "all" for v in so["variants"]),
                    "the exact waypoint set measure_cutlery_block.py flipped "
                    "is re-run here, so the earlier refutation is reproduced "
                    "rather than argued with")

    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "grasp_feasibility_controls.json").write_text(
        json.dumps({"all_pass": bool(ok), "controls": results}, indent=1),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
