#!/usr/bin/env python3
"""Is scripts/test_fork_release.py capable of going red?

Seventeen passing checks say nothing until each one has been shown to fail on
an input that violates the thing it asserts.  This drives the checker against
mutated COPIES of evidence/fork_release.json -- the real file is opened
read-only and never written -- and requires every mutant to be killed.

Each mutant corresponds to one claim of F-FORK-HANDOFF-001 and inverts exactly
that claim, so a survivor names the check that is decorative.

Run:  python3 scripts/mutate_fork_release.py
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
SRC = EVID / "fork_release.json"
CHECKER = ROOT / "scripts" / "test_fork_release.py"

PLACE_LABELS = ("target_fork_over", "target_fork_down", "target_fork_release",
                "target_fork_retreat")


def m_placer_holds(d):
    """The placing arm DOES hold the fork -- the finding's central negation."""
    d["summary"]["seeds_where_placer_ever_holds_the_fork"] = [2]
    d["per_seed"][2]["waypoints"]["target_fork_down"]["held_by"] = ["left"]
    return d


def m_place_not_empty(d):
    """A place waypoint with the fork in the jaws."""
    d["per_seed"][7]["waypoints"]["target_fork_release"]["held_by"] = ["left"]
    return d


def m_detector_never_yes(d):
    """The positive control fails: the detector never returns True."""
    d["controls"]["detector_can_say_yes"]["pass"] = False
    return d


def m_fork_hovering(d):
    """The fork is unheld and NOT resting -- i.e. it never fell, it hovers."""
    d["controls"]["height_agrees_with_held"]["disagreements"] = [
        {"seed": 3, "label": "target_fork_down", "fork_z_mm": 900.0}]
    return d


def m_not_resting_at_lift(d):
    d["summary"]["fork_resting_at_taker_lift"]["4"] = False
    return d


def m_z_between_rest_heights(d):
    d["summary"]["fork_z_at_taker_lift_mm"]["4"] = 770.0
    return d


def m_taker_collides(d):
    """The taker DOES reach the fork -- the refutation this tick kept."""
    d["summary"]["taker_closest_approach_mm"] = 3.0
    return d


def m_fork_placed_moved(d):
    """The published figure quietly changes."""
    d["summary"]["placed"]["3"] = True
    return d


def m_scoring_seeds_moved(d):
    d["summary"]["placed"]["2"] = False
    d["summary"]["placed"]["1"] = True
    return d


def m_no_drift(d):
    """A scoring seed whose fork did NOT move after release."""
    for r in d["per_seed"]:
        if r["seed"] == 9:
            r["drift_after_release_mm"] = 1.0
    return d


def m_repro_broken(d):
    d["controls"]["reproduces_published"]["pass"] = False
    return d


def m_decomposition_open(d):
    d["controls"]["decomposition_closes"]["pass"] = False
    return d


def m_labels_missing(d):
    d["controls"]["labels_present"]["pass"] = False
    return d


def m_carry_is_the_blocker(d):
    """The empty jaws do NOT arrive -- the reading four ticks assumed."""
    for k in d["summary"]["at_down"]:
        d["summary"]["at_down"][k]["carry_mm"] = 120.0
    return d


def m_knob_shipped(d):
    d["knobs_touched"] = ["CUTLERY_HANDOFF_SQUARE"]
    return d


MUTANTS = [
    ("placer holds the fork", m_placer_holds),
    ("a place waypoint is not empty-gripper", m_place_not_empty),
    ("positive control fails", m_detector_never_yes),
    ("unheld fork hovers instead of falling", m_fork_hovering),
    ("fork not resting at the taker lift", m_not_resting_at_lift),
    ("fork z between the two resting heights", m_z_between_rest_heights),
    ("taker DOES reach the fork", m_taker_collides),
    ("fork_placed silently becomes 4/10", m_fork_placed_moved),
    ("the scoring seeds change identity", m_scoring_seeds_moved),
    ("a scoring seed has no post-release drift", m_no_drift),
    ("reproduction of the published column fails", m_repro_broken),
    ("the decomposition does not close", m_decomposition_open),
    ("a seed is missing a waypoint", m_labels_missing),
    ("the carry IS the blocker after all", m_carry_is_the_blocker),
    ("a controller knob was shipped", m_knob_shipped),
]


def run_checker(path: pathlib.Path) -> int:
    env = dict(os.environ, FORK_RELEASE_EVIDENCE=str(path))
    p = subprocess.run([sys.executable, str(CHECKER)], env=env,
                       capture_output=True, text=True)
    return p.returncode


def main() -> int:
    base = json.loads(SRC.read_text())

    with tempfile.TemporaryDirectory() as td:
        clean = pathlib.Path(td) / "clean.json"
        clean.write_text(json.dumps(base))
        rc = run_checker(clean)
        # A mutation campaign whose baseline is red kills every mutant for the
        # wrong reason; this is the guard that makes the score mean something.
        if rc != 0:
            print(f"  BASELINE IS RED (rc={rc}) -- no kill count below would "
                  f"mean anything. Aborting.")
            return 2
        print("  baseline green")

        killed, survived = [], []
        for name, fn in MUTANTS:
            mp = pathlib.Path(td) / "mutant.json"
            mp.write_text(json.dumps(fn(copy.deepcopy(base))))
            r = run_checker(mp)
            (killed if r != 0 else survived).append(name)
            print(f"  [{'KILLED' if r != 0 else 'SURVIVED'}] {name}")

    print(f"\n{len(killed)}/{len(MUTANTS)} mutants killed")
    if survived:
        print("SURVIVORS (each names a check that cannot fail):")
        for s in survived:
            print(f"  - {s}")
    print(f"\nevidence/fork_release.json was not written: "
          f"still {SRC.stat().st_size} bytes")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
