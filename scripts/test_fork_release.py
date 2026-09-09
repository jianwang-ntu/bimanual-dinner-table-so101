#!/usr/bin/env python3
"""Controls for F-FORK-HANDOFF-001, and the test that can refute it.

The claim is NOT "the fork placement got worse".  The published number does not
move at all.  The claim is about the MECHANISM behind a number this repository
already ships:

    `fork_placed` 3 / 10 is not a placement.  The left arm -- the arm that runs
    `_place(target_fork)` -- never holds the fork, on any seed, at any waypoint.
    The right arm picks it out of the drawer and holds it firmly over the
    hand-off site; the left arm never arrives; the fork falls to the table
    during the hand-off; and `_place` then runs its whole over / down / release
    / retreat sequence on an EMPTY GRIPPER, arriving accurately at a target it
    is carrying nothing to.

The falsifiable form is a contact record with a working positive control:

  * if the placing arm's jaws ever touch the fork, the claim is false;
  * if the detector never returns True anywhere, the claim is unsupported --
    which is why the PICKER's own grasp has to register, and does;
  * if the fork is airborne rather than resting once the giver lets go, the
    "it fell" reading is wrong.

Against us, three ways, all asserted rather than mentioned:

  * the tick's own first hypothesis -- that the taker knocks the fork out of
    the giver's jaws -- is REFUTED by this probe's own geometry, and the test
    asserts the refutation rather than quietly dropping it;
  * `fork_placed` 3 / 10 must be UNCHANGED, so the correction cannot be a
    rewrite of an inconvenient figure;
  * the one height/contact disagreement is asserted to be exactly a fork
    mid-fall, not a hovering unheld fork, because the second would mean the
    contact test is missing a grasp.

Run:  python3 scripts/test_fork_release.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
# scripts/mutate_fork_release.py points this at a mutated COPY so the mutation
# campaign never writes into evidence/.  The path actually read is printed, so
# a run against a copy can never be mistaken for a run against the tree.
FORK_RELEASE = pathlib.Path(
    os.environ.get("FORK_RELEASE_EVIDENCE", str(EVID / "fork_release.json")))

CHECKS: list[tuple[str, bool, str]] = []

PLACE_LABELS = ("target_fork_over", "target_fork_down", "target_fork_release",
                "target_fork_retreat")
# envs/task.py TABLE_TOP_Z = 0.75 + the fork handle's 2.5 mm half-thickness;
# and its drawer pose z=0.784 in envs/dinner_table.xml.
TABLE_MM, DRAWER_MM = 752.5, 783.5


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def _call_args(src: str, fn: str) -> list[str]:
    """The argument text of every call to ``fn`` that is not its definition."""
    out, i = [], 0
    while True:
        i = src.find(fn + "(", i)
        if i < 0:
            return out
        if src[max(0, i - 4):i] == "def ":
            i += 1
            continue
        j, depth = i + len(fn), 0
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(src[i:j + 1])
        i = j


def main() -> int:
    print(f"  reading {FORK_RELEASE}")
    fr = json.loads(FORK_RELEASE.read_text())
    ctl, summ, rows = fr["controls"], fr["summary"], fr["per_seed"]
    ok_rows = [r for r in rows if "error" not in r]

    # ---- the probe's own controls, read back rather than restated ----------
    check("the probe reproduces the adopted cell's published per-seed column",
          ctl["reproduces_published"]["pass"],
          f"delta {ctl['reproduces_published']['abs_delta_mm']}")
    check("the release decomposition closes",
          ctl["decomposition_closes"]["pass"],
          f"max residual {ctl['decomposition_closes']['max_residual_mm']} mm")
    check("every seed reached every watched waypoint",
          ctl["labels_present"]["pass"], str(ctl["labels_present"]["missing"]))
    check("POSITIVE CONTROL: the contact detector can say yes",
          ctl["detector_can_say_yes"]["pass"],
          "the picker's own grasp registers; without this a 'never held' "
          "reading proves nothing")

    # ---- the claim --------------------------------------------------------
    check("the placing arm NEVER holds the fork, on any seed",
          summ["seeds_where_placer_ever_holds_the_fork"] == [],
          f"seeds where it does: "
          f"{summ['seeds_where_placer_ever_holds_the_fork']}")

    never_held_in_place = all(
        not ((r["waypoints"].get(l) or {}).get("held_by"))
        for r in ok_rows for l in PLACE_LABELS)
    check("_place(target_fork) runs on an empty gripper on every seed",
          never_held_in_place)

    resting = {s: v for s, v in summ["fork_resting_at_taker_lift"].items()}
    check("by the taker's lift the fork is already resting, on every seed",
          all(resting.values()), str(resting))

    zs = summ["fork_z_at_taker_lift_mm"]
    check("that resting height is the table or the drawer, nothing between",
          all(abs(z - TABLE_MM) < 6 or abs(z - DRAWER_MM) < 6
              for z in zs.values()), str(zs))

    # The empty gripper still arrives: this is what made the error look like a
    # release problem for four ticks.
    carries = [v["carry_mm"] for v in summ["at_down"].values()
               if v["carry_mm"] is not None]
    check("the empty jaws still arrive accurately at target_fork "
          "(median carry well inside tolerance)",
          sorted(carries)[len(carries) // 2] < 45.0,
          f"carry_mm {sorted(carries)}")

    # ---- AGAINST US, 1: this tick's own first hypothesis, and its limit ----
    # WITHDRAWN 2026-09-09 by F-FORK-EJECT-001.  This check used to read
    # "REFUTED, and kept: the taker does not knock the fork out -- its jaws
    # never come within a jaw half-span of it".  The NUMBER is still true and
    # is still asserted; the CONCLUSION drawn from it was not, because 62.3 mm
    # is a minimum over waypoint ENDS and the taker crosses the gap between
    # samples.  At every simulator step it reaches 18.6 mm
    # (evidence/fork_knockout.json).  The check now asserts only what its own
    # data supports, and asserts that the correction exists rather than letting
    # the reader infer the old reading from the old number.
    closest = summ["taker_closest_approach_mm"]
    check("the taker's jaws stay clear of the fork AT EVERY WAYPOINT END -- "
          "which is all this sampling can see, and NOT a refutation of a knock",
          closest is not None and closest > 45.0,
          f"closest taker approach over all seeds and waypoint ends: "
          f"{closest} mm; at step resolution it is 18.6 mm, see "
          f"scripts/measure_fork_knockout.py")
    ko = EVID / "fork_knockout.json"
    check("and the step-resolution measurement that withdraws that reading is "
          "actually present, so the correction cannot be lost",
          ko.exists() and json.loads(ko.read_text())["controls"][
              "link_geoms_are_new"]["pass"],
          str(ko))

    # ---- AGAINST US, 2: the published number is untouched ------------------
    check("fork_placed is UNCHANGED at 3/10 -- the correction is to the "
          "mechanism, not to the figure",
          sum(1 for v in summ["placed"].values() if v) == 3,
          str(summ["placed"]))
    check("the three scoring seeds are the published ones",
          sorted(int(s) for s, v in summ["placed"].items() if v) == [2, 7, 9])

    drift = {r["seed"]: r.get("drift_after_release_mm") for r in ok_rows
             if r["placed"]}
    check("on every scoring seed the fork MOVES after the jaws open, so the "
          "score is not the placing motion",
          all((d or 0) > 45.0 for d in drift.values()), str(drift))

    # ---- AGAINST US, 3: the one cross-check disagreement is a fork mid-fall
    dis = ctl["height_agrees_with_held"]["disagreements"]
    check("every height/contact disagreement is a fork strictly BETWEEN the "
          "two resting heights, i.e. falling",
          all(TABLE_MM < d["fork_z_mm"] < DRAWER_MM for d in dis),
          str(dis))

    # ---- the code fact the finding turns on --------------------------------
    src = (ROOT / "envs" / "controller.py").read_text()
    calls = _call_args(src, "_handoff")
    check("_handoff has a taker_jaw parameter", "taker_jaw=None" in src)
    check("NO _handoff call site passes taker_jaw -- the remedy its own "
          "docstring names is dead code",
          bool(calls) and all("taker_jaw" not in c for c in calls),
          f"{len(calls)} call sites")

    # ---- and nothing was shipped this tick ---------------------------------
    check("the probe touched no controller knob",
          fr["knobs_touched"] == [] and fr["shipped_controller"] is True)

    print(f"\n  taker closest approach to the fork, all seeds: {closest} mm")
    print(f"  giver holds it at: "
          f"{ctl['detector_can_say_yes']['labels_where_picker_holds_the_fork']}")
    print(f"  fork z at taker lift: {zs}")

    bad = [c for c in CHECKS if not c[1]]
    print()
    for n, okc, det in CHECKS:
        print(f"  [{'PASS' if okc else 'FAIL'}] {n}" + (f"  -- {det}" if det else ""))
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
