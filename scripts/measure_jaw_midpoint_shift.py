#!/usr/bin/env python3
"""Two loose ends the envelope sweep left, both pure kinematics.

1. WHICH PART OF THE HAND.  ``measure_gripper_envelope.py`` names the first
   contact of each descend hold on both sides, but several jaw geoms in this
   model are unnamed and print as ``geomNNN``, which is not an answer to
   "which part of the hand lands first".  This resolves every geom id on both
   arms to its BODY, and checks the ids are the same in every seed's model
   before offering the table as a key -- randomization moves bodies, and an id
   map that shifted per seed would mis-name every contact in the sweep.

2. WHERE THE MIDPOINT ACTUALLY GOES.  ``Move.__init__``'s docstring justifies
   ``plan_at`` with "the jaws meet ~41 mm nearer the wrist closed than open".
   The sweep measured the displacement between the two widths the cutlery
   descent actually uses -- 7.0 mm and 50.44 mm of jaw separation -- as 21.8 mm
   of which -0.08 mm is along the approach (wrist) axis.  Those two statements
   are about different width pairs, so this sweeps the WHOLE calibration range
   against the closed width and reports the approach-axis and jaw-axis
   components at each, from the home keyframe pose.  Then the docstring's claim
   is either true somewhere in this gripper's range or it is not, and this says
   which without extending a measurement past where it was taken.

Writes evidence/jaw_midpoint_shift.json.  Changes no behaviour.
Run:  python3 scripts/measure_jaw_midpoint_shift.py
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
SEEDS = (0, 3, 7)
CLOSED_SEP_MM = 7.0        # what pinch() asks for on the fork handle


def geom_table(model) -> dict:
    out = {}
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) or "?"
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        out[str(g)] = {"geom": name, "body": body}
    return out


def main() -> int:
    from envs.randomize import make_env
    from envs import controller as C

    tables = {}
    for s in SEEDS:
        model, data, _ = make_env(s)
        tables[s] = geom_table(model)
    ref = tables[SEEDS[0]]
    stable = all(tables[s] == ref for s in SEEDS)

    model, data, _ = make_env(SEEDS[0])
    grip = C.Gripper(model, "right")
    scratch = mujoco.MjData(model)

    def read(q):
        mujoco.mj_resetDataKeyframe(model, scratch, 0)
        scratch.qpos[grip.grip_q] = float(q)
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        return (grip.tip_mid(model, scratch).copy(),
                grip.jaw_axis(model, scratch).copy(),
                grip.approach_axis(scratch).copy(),
                grip.sep_for_q(float(q)))

    q_closed = grip.q_for_sep(CLOSED_SEP_MM / 1000.0)
    t0, jaw0, app0, sep0 = read(q_closed)

    rows = []
    for q in np.linspace(float(grip.lo[-1]), float(grip.hi[-1]), 25):
        t, _, _, sep = read(q)
        d = t - t0
        rows.append({
            "q_rad": round(float(q), 4),
            "sep_mm": round(sep * 1000, 2),
            "shift_mm": round(float(np.linalg.norm(d)) * 1000, 2),
            "along_approach_mm": round(float(d @ app0) * 1000, 2),
            "along_jaw_mm": round(float(d @ jaw0) * 1000, 2),
            "half_sep_change_mm": round((sep - sep0) / 2 * 1000, 2),
        })

    max_app = max(abs(r["along_approach_mm"]) for r in rows)
    worst = max(rows, key=lambda r: abs(r["along_approach_mm"]))
    # Does the midpoint shift equal half the separation change?  If the fixed
    # jaw really is fixed, it must, everywhere -- and that is a much stronger
    # statement than agreeing at one width.
    resid = max(abs(r["shift_mm"] - abs(r["half_sep_change_mm"])) for r in rows)
    # The value the cutlery descent actually executes at.
    at_narrow = min(rows, key=lambda r: abs(r["q_rad"] - C.GRIPPER_NARROW))

    controls = {
        "geom_ids_are_stable_across_seeds": stable,
        "geom_table_covers_every_geom": len(ref) == model.ngeom,
        "midpoint_shift_is_half_the_separation_change": resid < 0.6,
        "approach_axis_component_never_reaches_41mm": max_app < 41.0,
        "the_swept_range_reaches_the_open_width": max(
            r["sep_mm"] for r in rows) > 100.0,
    }

    out = {
        "probe": "measure_jaw_midpoint_shift.py",
        "finding_id": "F-JAW-MIDPOINT-001",
        "closed_reference": {"q_rad": round(float(q_closed), 4),
                             "sep_mm": round(sep0 * 1000, 2)},
        "seeds_checked_for_id_stability": list(SEEDS),
        "geom_ids_stable": stable,
        "ngeom": int(model.ngeom),
        "max_along_approach_mm": round(max_app, 2),
        "max_along_approach_at": worst,
        "shift_vs_half_separation_max_resid_mm": round(resid, 3),
        "at_gripper_narrow": at_narrow,
        "controls": controls,
        "rows": rows,
        "geom_names": ref,
    }
    (EVID / "jaw_midpoint_shift.json").write_text(json.dumps(out, indent=1))

    print(f"  geom ids stable across seeds {SEEDS}: {stable}  (ngeom {model.ngeom})")
    print(f"  closed reference: {round(sep0*1000,2)} mm at q={round(float(q_closed),4)}")
    print("\n   sep_mm   shift  along_approach  along_jaw  half_sep_change")
    for r in rows[::3]:
        print(f"  {r['sep_mm']:>7}  {r['shift_mm']:>6}  "
              f"{r['along_approach_mm']:>14}  {r['along_jaw_mm']:>9}  "
              f"{r['half_sep_change_mm']:>15}")
    print(f"\n  largest |along_approach| anywhere in the range: {round(max_app,2)} mm "
          f"at {worst['sep_mm']} mm separation")
    print(f"  at GRIPPER_NARROW ({C.GRIPPER_NARROW}): {at_narrow}")
    print("\n  controls:")
    for k, v in controls.items():
        print(f"    {k}: {v}")
    return 0 if all(controls.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
