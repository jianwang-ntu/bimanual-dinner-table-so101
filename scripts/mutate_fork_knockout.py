#!/usr/bin/env python3
"""Is scripts/test_fork_knockout.py capable of going red?

Eighteen passing checks say nothing until each one has been shown to fail on an
input that violates the thing it asserts.  This drives the checker against
mutated COPIES of evidence/fork_knockout.json -- the real file is opened
read-only and never written -- and requires every mutant to be killed.

Each mutant inverts exactly one claim of F-FORK-EJECT-001, including the two
that run AGAINST us (the correction to F-FORK-HANDOFF-001's refutation, and the
refusal to replace it with a knock story).  A survivor names a check that is
decorative.

Run:  python3 scripts/mutate_fork_knockout.py
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
SRC = EVID / "fork_knockout.json"
CHECKER = ROOT / "scripts" / "test_fork_knockout.py"


def _carried(d):
    return [r for r in d["per_seed"]
            if "error" not in r and r["carry_run_steps"]]


# ---- the probe's own controls ---------------------------------------------
def m_repro_broken(d):
    """The probe drifted off the shipped configuration."""
    d["controls"]["reproduces_published"]["pass"] = False
    return d


def m_links_not_new(d):
    """The 'links are now instrumented' claim is false."""
    d["controls"]["link_geoms_are_new"]["pass"] = False
    return d


def m_detector_never_yes(d):
    """The positive control fails: the detector never returns True."""
    d["controls"]["detector_can_say_yes"]["pass"] = False
    return d


def m_spurious_body(d):
    """The attribution names a body it cannot be seeing."""
    d["controls"]["unrelated_body_absent"]["pass"] = False
    return d


def m_loss_before_lift(d):
    """The 'loss' found is the pick failing, not the hand-off."""
    d["controls"]["loss_is_after_lift"]["pass"] = False
    return d


# ---- the claim -------------------------------------------------------------
def m_never_carried(d):
    """The giver does not carry the fork at all -- the central negation."""
    for r in d["per_seed"]:
        r["carry_run_steps"] = None
    return d


def m_carry_is_a_brush(d):
    """The carry is a light touch, not a loaded grip."""
    for r in _carried(d):
        r["carry_grip_force_N"]["max"] = 3.0
        r["carry_grip_force_N"]["last_20_steps"] = [3.0] * 20
    return d


def m_force_survives_loss(d):
    """The jaws are still loaded at the loss step -- nothing was released."""
    for r in _carried(d):
        r["carry_grip_force_N"]["at_loss"] = 61.0
    return d


def m_collapse_is_slow(d):
    """A slow slide, not a release: force decays over many steps."""
    for r in _carried(d):
        v = [90.0] + [80.0 - 4 * i for i in range(19)]
        r["carry_grip_force_N"]["last_20_steps"] = v
    return d


def m_jaws_opened(d):
    """The jaws DID move: the loss is a grip opening, which is a different bug."""
    for r in _carried(d):
        r["carry_jaw_sep_mm"]["at_loss"] = r["carry_jaw_sep_mm"]["max"]
    return d


def m_no_interference(d):
    """The jaws sit outside the handle -- the squeeze story is wrong."""
    for r in _carried(d):
        r["carry_jaw_sep_mm"]["at_loss"] = 11.5
        r["carry_jaw_sep_mm"]["min"] = 11.5
    return d


# ---- the two that run against us ------------------------------------------
def m_taker_stays_far(d):
    """The previous tick's refutation was right after all: the taker is far.

    If this survives, the checker is not actually asserting the correction and
    the finding could quietly drop an inconvenient earlier claim.
    """
    for r in d["per_seed"]:
        r["min_taker_tip_mm"] = 99.0
    return d


def m_taker_never_touches(d):
    """No taker body ever touches the fork."""
    for r in d["per_seed"]:
        r["first_taker_contact_step"] = None
        r["first_taker_contact_t"] = None
        r["first_taker_contact_bodies"] = []
    return d


def m_knock_explains_everything(d):
    """Every carried seed has taker contact -- a knock story with no exception.

    If this survives, the checker would let the finding be replaced by the very
    story it says it is refusing.
    """
    for r in _carried(d):
        if r["first_taker_contact_step"] is None:
            r["first_taker_contact_step"] = 9000
            r["first_taker_contact_t"] = 18.0
            r["first_taker_contact_bodies"] = ["left_gripper"]
    return d


def m_knob_shipped(d):
    """The probe swept a knob and reported as if it had not."""
    d["knobs_touched"] = ["CUTLERY_SQUEEZE"]
    return d


MUTANTS = [
    ("repro_broken", m_repro_broken),
    ("links_not_new", m_links_not_new),
    ("detector_never_yes", m_detector_never_yes),
    ("spurious_body", m_spurious_body),
    ("loss_before_lift", m_loss_before_lift),
    ("never_carried", m_never_carried),
    ("carry_is_a_brush", m_carry_is_a_brush),
    ("force_survives_loss", m_force_survives_loss),
    ("collapse_is_slow", m_collapse_is_slow),
    ("jaws_opened", m_jaws_opened),
    ("no_interference", m_no_interference),
    ("taker_stays_far", m_taker_stays_far),
    ("taker_never_touches", m_taker_never_touches),
    ("knock_explains_everything", m_knock_explains_everything),
    ("knob_shipped", m_knob_shipped),
]


def run_checker(path: pathlib.Path) -> int:
    env = dict(os.environ, FORK_KNOCKOUT_EVIDENCE=str(path))
    p = subprocess.run([sys.executable, str(CHECKER)], env=env,
                       capture_output=True, text=True)
    return p.returncode


def main() -> int:
    before = SRC.stat().st_size
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
    after = SRC.stat().st_size
    print(f"\nevidence/fork_knockout.json was not written: "
          f"{before} bytes before, {after} bytes after")
    return 1 if survived or after != before else 0


if __name__ == "__main__":
    sys.exit(main())
