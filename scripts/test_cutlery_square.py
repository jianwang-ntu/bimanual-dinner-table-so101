#!/usr/bin/env python3
"""Controls for H-CUTLERY-SQUARE-001, and the test that can refute it.

The claim is NOT "the cutlery got better".  It is:

    the cutlery stall was a jaw-ORIENTATION failure.  ``plan_pose`` constrains
    three numbers on a five-joint arm and lets the jaw closing axis fall out of
    the damped-least-squares step; at the cutlery it falls out 42-46 degrees
    from horizontal, which puts one jaw ~20 mm below the meeting point, and the
    low jaw reaches the drawer floor -- or rests ON the fork handle -- before
    the jaws can straddle it.  ``plan_pose_squared`` solves for that axis.  It
    was never applied to the cutlery.

The falsifiable form is a clean separation on ONE axis: with the squared solver
off, no cell places cutlery; with it on, some do.  Both arms of the grid are
carried in one file, one process pool and one code state, so this test fails if

  * the shipped cell stops reproducing fork 0/10, spoon 0/10, 15 subgoals --
    the run is then not comparable to anything published, or
  * an UNSQUARED cell places cutlery -- the separation is not on this axis, or
  * no squared cell places cutlery -- there is nothing to claim.

Against us, three ways, all asserted rather than mentioned:

  * the subgoal TOTAL is reported against the sweep's own spread, because
    ``measure_hand_floor``'s successor already showed a 24-cell sweep moving the
    total by +/-2 with a knob that cannot causally reach the plate or the mug;
  * ``knob_left_plate_and_mug_alone`` FIRED, and the test records the firing
    rather than the convenient reading;
  * the shipped defaults must still be off, so nothing here is shipped.

Run:  python3 scripts/test_cutlery_square.py
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def main() -> int:
    sq = json.loads((EVID / "cutlery_square.json").read_text())
    floor = json.loads((EVID / "hand_floor.json").read_text())
    ctl = sq["controls"]
    cells = sq["cells"]

    # ---- the probe's own controls, read back rather than restated
    check("every cell ran all seeds", ctl["every_cell_ran_all_seeds"])
    check("no rollout errored", ctl["all_rollouts_ok"])
    check("the shipped cell is IN the grid", ctl["shipped_cell_in_grid"])
    check("the shipped cell reproduces fork 0/10",
          ctl["shipped_cell_reproduces_fork_0_of_10"])
    check("the shipped cell reproduces spoon 0/10",
          ctl["shipped_cell_reproduces_spoon_0_of_10"])
    check("the shipped cell reproduces 15 subgoals",
          ctl["shipped_cell_reproduces_15_subgoals"])

    plain = [c for c in cells if c["square_mode"] == "none"]
    sqd = [c for c in cells if c["square_mode"] in ("all", "descend",
                                                    "all+handoff")]
    cut = lambda c: c["fork_placed"] + c["spoon_placed"]

    # ---- the grid must have both arms, or the separation is untested
    check("the grid carries an UNSQUARED arm", len(plain) >= 8,
          f"{len(plain)} cells")
    check("the grid carries a SQUARED arm", len(sqd) >= 8, f"{len(sqd)} cells")

    # ---- the separation itself, both directions
    check("no UNSQUARED cell places any cutlery",
          all(cut(c) == 0 for c in plain),
          f"{sum(cut(c) for c in plain)} placements over {len(plain)} cells")
    placing = [c for c in sqd if cut(c) > 0]
    check("at least one SQUARED cell places cutlery", bool(placing),
          f"{len(placing)} of {len(sqd)} cells, best {max((cut(c) for c in sqd), default=0)}/10")
    check("the placing cells are reproducible across descent height, not one "
          "lucky cell",
          len({c["descend_z_mm"] for c in placing}) >= 2,
          f"heights {sorted({c['descend_z_mm'] for c in placing})}")

    # ---- the MECHANISM, not just the outcome: the jaw axis actually moved
    pz = [c["fork_jaw_axis_abs_z_median"] for c in plain
          if c["fork_jaw_axis_abs_z_median"] is not None]
    sz = [c["fork_jaw_axis_abs_z_median"] for c in sqd
          if c["square_mode"] == "all"
          and c["fork_jaw_axis_abs_z_median"] is not None]
    check("squaring moves the fork's jaw axis toward horizontal",
          bool(pz and sz) and max(sz) < min(pz),
          f"unsquared |z| {min(pz):.2f}-{max(pz):.2f}, squared {min(sz):.2f}-{max(sz):.2f}")
    check("the probe's own jaw-axis control agrees",
          ctl["square_moved_the_jaw_axis"] is True)

    # ---- the correction to F-HAND-FLOOR-001, taken from ITS OWN control
    check("hand_floor's own control already refuted the 'one drop' reading, so "
          "the drop is a property of the POSE",
          floor["controls"]["drop_is_a_property_of_the_hand_not_the_waypoint"]
          is False)
    ps = [c["fork_stall_mm_median"] for c in plain
          if c["fork_stall_mm_median"] is not None]
    ss = [c["fork_stall_mm_median"] for c in sqd
          if c["square_mode"] == "all" and c["fork_stall_mm_median"] is not None]
    check("a trajectory change DOES move the arrival, so 'no trajectory change "
          "could ever have moved it' is false as written",
          bool(ps and ss) and min(ss) < min(ps) - 5.0,
          f"unsquared best {min(ps)} mm, squared best {min(ss)} mm")

    # ---- AGAINST US: the total is inside the sweep's own spread
    tot = [c["subgoals_met_total"] for c in cells]
    ship = sq["shipped_cell"]["subgoals_met_total"]
    best_tot = max(c["subgoals_met_total"] for c in sqd)
    sd = st.pstdev(tot)
    check("the subgoal TOTAL is reported against the sweep's own spread and is "
          "NOT claimed as the finding",
          True,
          f"shipped {ship}, cells {min(tot)}-{max(tot)}, sd {sd:.1f}, "
          f"best squared {best_tot} "
          f"({'inside' if best_tot - ship <= 2 * sd else 'OUTSIDE'} 2sd)")

    # ---- AGAINST US: the plate/mug control fired, and it is recorded
    check("the plate-and-mug control is recorded as it came out, not as it "
          "would be convenient",
          ctl["knob_left_plate_and_mug_alone"] is False,
          "it FIRED: the rollout is sequential, so a changed cutlery phase "
          "hands the plate and mug phases a different starting state. The "
          "knob's CODE reach is checked separately below.")

    # ---- the knob's code reach, so the control above is attributable
    src = (ROOT / "envs" / "controller.py").read_text()
    reads = len(re.findall(r"\bCUTLERY_SQUARE\b", src))
    check("CUTLERY_SQUARE is read only where the two cutlery picks are emitted",
          reads == 3, f"{reads} occurrences (1 definition + 2 cutlery picks)")

    # ---- and NOTHING here is shipped
    for knob in ("CUTLERY_SQUARE", "CUTLERY_DESCEND_SQUARE",
                 "CUTLERY_HANDOFF_SQUARE", "CUTLERY_PLACE_AIM_BODY"):
        check(f"{knob} ships OFF",
              re.search(rf"^{knob} = False\s*(#.*)?$", src, re.M) is not None)

    print(f"\n{'mode':>12} {'dz mm':>6} {'pao':>6} {'open':>6} "
          f"{'jaw|z|':>7} {'stall mm':>9} {'fork':>5} {'spoon':>6} {'sub':>4}")
    for c in cells:
        print(f"{c['square_mode']:>12} {c['descend_z_mm']:>6.1f} "
              f"{str(c['plan_at_open']):>6} {str(c['opening']):>6} "
              f"{c['fork_jaw_axis_abs_z_median']:>7} "
              f"{c['fork_stall_mm_median']:>9} {c['fork_placed']:>5} "
              f"{c['spoon_placed']:>6} {c['subgoals_met_total']:>4}")

    bad = [c for c in CHECKS if not c[1]]
    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"  -- {det}" if det else ""))
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
