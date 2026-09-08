#!/usr/bin/env python3
"""Controls for F-HAND-FLOOR-001, and the predictive test that decides it.

The claim is not "the cutlery is hard".  It is:

    a pinch grasp is possible only where the object's graspable feature stands
    at least as far above the surface under it as the hand hangs below the
    point it pinches with.

That is falsifiable on this scene's own four graspables, and one of them --
the mug -- is a POSITIVE case: it is the one object this entry has ever
lifted.  A rule that only ever predicts failure is not a rule, so the mug is
what makes this a test rather than a story.

Run:  python3 scripts/test_hand_floor.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def main() -> int:
    floor = json.loads((EVID / "hand_floor.json").read_text())
    cross = json.loads((EVID / "cutlery_zcross.json").read_text())

    wps = floor["waypoints"]
    ctl = floor["controls"]

    # ---- the probes' own controls, read back rather than restated
    check("hand_floor: all seeds ran", ctl["all_seeds_ok"])
    check("hand_floor: both cutlery waypoints present",
          ctl["cutlery_waypoints_present"])
    check("hand_floor: the positive control (mug) is present",
          ctl["control_waypoint_present"])
    check("zcross: every cell ran all seeds",
          cross["controls"]["every_cell_ran_all_seeds"])
    check("zcross: the shipped cell is IN the grid",
          cross["controls"]["shipped_cell_in_grid"])
    check("zcross: the shipped cell reproduces fork 0/10",
          cross["controls"]["shipped_cell_reproduces_fork_0_of_10"])
    check("zcross: the shipped cell reproduces spoon 0/10",
          cross["controls"]["shipped_cell_reproduces_spoon_0_of_10"])

    # ---- this probe's control FIRED, and that is recorded, not hidden.
    # drop_mm is NOT one number for the whole hand: it depends on the wrist
    # pose the solver chose.  The finding is stated per waypoint because of
    # this, and the check asserts the control's verdict rather than the
    # convenient one.
    check("hand_floor: drop is per-pose, not one hand constant "
          "(the probe's own control refuted the simpler claim)",
          ctl["drop_is_a_property_of_the_hand_not_the_waypoint"] is False,
          f"drops mm: " + ", ".join(
              f"{k} {v['drop_mm_median']}" for k, v in wps.items()))

    # ---- the absolute floor: what the arm ACHIEVES does not follow what it
    # is ASKED for.  This is the observation the eight earlier probes could
    # not make, because each read the gap relative to a target that moved.
    runs = [r for r in cross["runs"] if "error" not in r]
    import statistics as st

    def med_low(z, p, o):
        got = [r for r in runs if r["descend_z_m"] == z
               and r["plan_at_open"] == p and r["opening"] == o]
        return (st.median([r["bodies"]["fork"]["asked_tip_z_m"] for r in got]),
                st.median([r["bodies"]["fork"]["lowest_tip_z_m"] for r in got]))

    a_lo, l_lo = med_low(0.003, True, 0.15)
    a_hi, l_hi = med_low(0.015, True, 0.15)
    asked_mm = (a_hi - a_lo) * 1000
    got_mm = (l_hi - l_lo) * 1000
    check("zcross: asking 12 mm higher moves the achieved height under 4 mm",
          asked_mm > 11.0 and abs(got_mm) < 4.0,
          f"asked +{asked_mm:.1f} mm -> achieved +{got_mm:.1f} mm")

    # ---- the two independent measurements agree
    drawer_top = floor["surfaces_seed0"]["drawer_floor"]["top_z_m"]
    dynamic_clearance_mm = (l_lo - drawer_top) * 1000
    kinematic_min_mm = wps["fork_descend"]["drop_mm_min"]
    check("dynamic floor equals the kinematic minimum drop, within 2 mm",
          abs(dynamic_clearance_mm - kinematic_min_mm) < 2.0,
          f"dynamic {dynamic_clearance_mm:.2f} mm vs kinematic min "
          f"{kinematic_min_mm:.2f} mm")

    # ---- THE PREDICTIVE TEST, including the object that succeeds
    obj = floor["runs"][0]["objects"]
    surf = floor["surfaces_seed0"]
    drawer = surf["drawer_floor"]["top_z_m"]
    table = surf["table_top"]["top_z_m"]

    feature = {
        # graspable feature height above the surface that supports it
        "fork_descend":  (obj["fork"] - drawer) * 1000,
        "spoon_descend": (obj["spoon"] - drawer) * 1000,
        "plate_descend": (obj["plate"] - table) * 1000,
        "mug_descend":   (wps["mug_descend"]["tip_mid_z_m_median"] - table) * 1000,
    }
    # what the entry has measured about each object, from its own reports
    pinched = {"fork_descend": False, "spoon_descend": False,
               "plate_descend": False, "mug_descend": True}

    rows = []
    for wp, h in feature.items():
        drop = wps[wp]["drop_mm_median"]
        predicted = h >= drop
        rows.append((wp, round(h, 1), drop, predicted, pinched[wp]))
        check(f"prediction matches outcome: {wp}", predicted == pinched[wp],
              f"feature {h:.1f} mm vs drop {drop} mm -> "
              f"predicted pinchable={predicted}, observed={pinched[wp]}")

    check("the rule is not vacuous: it predicts at least one SUCCESS",
          any(p for _, _, _, p, _ in rows),
          "the mug is the positive case")

    # ---- and the sweep produced no withheld gain
    cells = cross["cells"]
    tot = [c["subgoals_met_total"] for c in cells]
    shipped = cross["shipped_cell"]["subgoals_met_total"]
    check("no cell in the cross placed any cutlery",
          all(c["fork_placed"] + c["spoon_placed"] == 0 for c in cells))
    check("the best cell's total is inside the sweep's own spread, so it is "
          "not a withheld gain",
          (max(tot) - shipped) <= 2 * st.pstdev(tot),
          f"shipped {shipped}, cells {min(tot)}-{max(tot)}, "
          f"sd {st.pstdev(tot):.1f}")

    print(f"\n{'waypoint':>16} {'feature mm':>11} {'hand drop mm':>13} "
          f"{'predicted':>10} {'observed':>9}")
    for wp, h, d, pr, ob in rows:
        print(f"{wp:>16} {h:>11.1f} {d:>13} {str(pr):>10} {str(ob):>9}")

    bad = [c for c in CHECKS if not c[1]]
    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"  -- {det}" if det else ""))
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
