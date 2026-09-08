#!/usr/bin/env python3
"""Mutation campaign for ``scripts/test_gripper_envelope.py``.

The suite protects two claims that are easy to assert vacuously:

  "the two new knobs ship at the values every published figure was measured
  at" -- which a suite that reads no knob at all also reports; and
  "the width mismatch is real but explains nothing" -- which a sweep that
  never varied the width, or that quietly lost its rollouts, also reports.

So every mutant below breaks exactly one thing the suite says it protects, and
the suite has to go red, ideally on the control that names it.

Families:

  code      (7)  the two new knobs drifting off the shipped controller, the
                 plan_at wiring going inert, and the emitted-width reader
                 being replaced by the constant it is supposed to check
                 independently.  These are the mutants that would silently
                 invalidate every figure in the repository.
  evidence (12)  one per outcome the suite reads out of the three probe files,
                 including the two failure modes this tree has actually had
                 before: a control that stops discriminating, and a sweep that
                 lost rollouts.
  suite     (4)  a control neutered in place, a control's negative half
                 removed, the arming rebind removed and the tautology guard
                 blinded.  A
                 campaign that cannot notice its own controls disappearing is
                 measuring the wrong thing.

Each mutant runs in its own fresh ``mkdtemp`` tree, COPIED not hard-linked, so
no mutant can write the real evidence files.  The baseline is asserted GREEN in
a copied tree BEFORE and AFTER -- a campaign whose baseline is already red
scores 100% and means nothing.

A mutation whose output is byte-identical to its input RAISES and is reported
BROKEN, not SURVIVED.  That guard exists because this tree has twice scored a
no-op mutant as a survivor of a control that was never exercised.

Writes evidence/gripper_envelope_mutants.json.
Run:  python3 scripts/mutate_gripper_envelope.py
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
SUITE = "scripts/test_gripper_envelope.py"
ENV = "evidence/gripper_envelope.json"
MID = "evidence/jaw_midpoint_shift.json"
IND = "evidence/cutlery_approach.json"
CTRL = "envs/controller.py"


def stage(dst: pathlib.Path) -> None:
    for d in ("envs", "scripts", "third_party"):
        shutil.copytree(ROOT / d, dst / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "evidence").mkdir()
    for f in (ENV, MID, IND):
        shutil.copy2(ROOT / f, dst / f)


def run_suite(tree: pathlib.Path) -> tuple[int, str]:
    """Return the FULL output, not a tail.

    It used to return the last 10,000 characters, and this suite's detail lines
    are long enough that the first controls fell off the end -- so a mutant
    killed by ``defaults`` was recorded as killed by nothing, and the
    "by the control that names them" tally was wrong for every early control.
    Truncation happens where the text is STORED, never before it is parsed.
    """
    p = subprocess.run([sys.executable, SUITE], cwd=tree,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def failed_controls(out: str) -> list[str]:
    return re.findall(r"^\s*FAIL (\S+) --", out, flags=re.M)


class NoOp(Exception):
    """A mutation that changed nothing.  Not a survivor -- a broken mutant."""


def _sub(path: pathlib.Path, old: str, new: str) -> None:
    """Rewrite `old` -> `new`; refuse if it is absent or changes nothing."""
    s = path.read_text()
    if old not in s:
        raise NoOp(f"{old!r} not found in {path.name}")
    t = s.replace(old, new, 1)
    if t == s:
        raise NoOp(f"replacement is byte-identical in {path.name}")
    path.write_text(t)


def _edit(path: pathlib.Path, fn) -> None:
    """Apply `fn` to a loaded JSON document; refuse if the bytes do not move."""
    before = path.read_text()
    d = json.loads(before)
    fn(d)
    after = json.dumps(d, indent=1)
    if after == before:
        raise NoOp(f"json edit is byte-identical in {path.name}")
    path.write_text(after)


def _cell(d, opening, plan_at_open):
    for c in d["cells"]:
        same = (c["opening"] is None if opening is None
                else (c["opening"] is not None
                      and abs(c["opening"] - opening) < 1e-12))
        if same and bool(c["plan_at_open"]) is bool(plan_at_open):
            return c
    raise NoOp(f"no cell {opening}/{plan_at_open}")


# --- code mutants ----------------------------------------------------------

def m_opening_knob_drifted(t):
    """The executed opening ships at a swept value instead of GRIPPER_NARROW."""
    _sub(t / CTRL, "CUTLERY_DESCEND_OPENING = GRIPPER_NARROW",
         "CUTLERY_DESCEND_OPENING = 0.30")


def m_plan_knob_drifted(t):
    """plan_at ships dropped -- the sweep's own alternative shipped by mistake."""
    _sub(t / CTRL, "CUTLERY_PLAN_AT_OPEN = False",
         "CUTLERY_PLAN_AT_OPEN = True")


def m_plan_wiring_inert(t):
    """The knob exists but the descend move ignores it."""
    _sub(t / CTRL, "plan_at=None if plan_at_open else close_to,",
         "plan_at=close_to,")


def m_opening_wiring_inert(t):
    """The opening knob exists but the fork pick ignores it."""
    _sub(t / CTRL, "open_to=CUTLERY_DESCEND_OPENING, descend_z=CUTLERY_DESCEND_Z",
         "open_to=GRIPPER_NARROW, descend_z=CUTLERY_DESCEND_Z")


def m_reader_reads_the_constant(t):
    """The suite stops reading the EMITTED move and reads the constant instead.

    This is the mutant that matters most: a reader like this reports the
    shipped pair correctly forever, including after the wiring is cut.
    """
    _sub(t / pathlib.Path(SUITE),
         "op = mv.opening(model, data, g) if callable(mv.opening) else mv.opening",
         "op = C.CUTLERY_DESCEND_OPENING")


# --- evidence mutants ------------------------------------------------------

def m_offset_zero(t):
    """The mismatch introduces no displacement at all."""
    def f(d):
        for k in ("fork_descend", "spoon_descend"):
            d["offset_by_waypoint"][k]["offset_mm_median"] = 0.0
    _edit(t / ENV, f)


def m_offset_doubled(t):
    """The displacement stops being half the width change."""
    def f(d):
        for k in ("fork_descend", "spoon_descend"):
            d["offset_by_waypoint"][k]["offset_mm_median"] *= 2
    _edit(t / ENV, f)


def m_control_offsets_too(t):
    """The no-plan_at waypoints show the same offset, so the probe's own
    negative control stops discriminating and the number means nothing."""
    def f(d):
        for k in ("fork_above", "spoon_above"):
            d["offset_by_waypoint"][k]["offset_mm_max"] = 21.8
            d["offset_by_waypoint"][k]["offset_mm_median"] = 21.8
    _edit(t / ENV, f)


def m_control_has_plan_at(t):
    """The approach waypoints are recorded as carrying a plan_at, which would
    make the two populations the same and the comparison meaningless."""
    def f(d):
        for k in ("fork_above", "spoon_above"):
            d["offset_by_waypoint"][k]["has_plan_at"] = True
    _edit(t / ENV, f)


def m_offset_is_fore_aft(t):
    """The displacement is recorded along the wrist axis, which is the world
    the shipped docstring describes and the measurement refutes."""
    def f(d):
        for k in ("fork_descend", "spoon_descend"):
            o = d["offset_by_waypoint"][k]
            o["along_jaw_mm_median"], o["along_approach_mm_median"] = (
                o["along_approach_mm_median"], o["along_jaw_mm_median"])
    _edit(t / ENV, f)


def m_midpoint_disagrees(t):
    """The independent full-range sweep no longer agrees at the same opening."""
    def f(d):
        d["at_gripper_narrow"]["along_jaw_mm"] = 2.0
        d["at_gripper_narrow"]["along_approach_mm"] = 21.0
    _edit(t / MID, f)


def m_shipped_baseline_drift(t):
    """The shipped cell stops reproducing the independently published stall."""
    def f(d):
        _cell(d, None, False)["fork_stall_mm_median"] += 9.0
    _edit(t / ENV, f)


def m_shipped_total_drift(t):
    """The shipped cell stops reproducing the published 15/50."""
    def f(d):
        _cell(d, None, False)["subgoals_met_total"] = 19
    _edit(t / ENV, f)


def m_placement_claimed(t):
    """A cell places the fork, which would make the verdict SUPPORTED."""
    def f(d):
        _cell(d, 0.15, True)["fork_placed"] = 3
    _edit(t / ENV, f)


def m_verdict_flipped(t):
    """The stored verdict stops following its own table."""
    def f(d):
        d["verdict"] = "SUPPORTED"
    _edit(t / ENV, f)


def m_placement_without_lift(t):
    """A placement is claimed with no lift behind it."""
    def f(d):
        c = _cell(d, 0.30, False)
        c["spoon_placed"] = 2
        c["spoon_lifted_mm_median"] = 0.0
    _edit(t / ENV, f)


def m_run_dropped(t):
    """A rollout is silently lost from one cell."""
    def f(d):
        _cell(d, 0.00, True)["n"] = 9
    _edit(t / ENV, f)


def m_probe_control_false(t):
    """The probe's own control is False and the suite ignores it."""
    def f(d):
        d["controls"]["no_offset_where_the_widths_are_equal"] = False
    _edit(t / ENV, f)


def m_contact_one_sided(t):
    """First contacts stop naming the hand side -- the shape every earlier
    probe in this tree recorded, and the reason this one was rewritten."""
    def f(d):
        for r in d["sweep_runs"]:
            for b in ("fork", "spoon"):
                fc = r.get("bodies", {}).get(b, {}).get("first_contact")
                if fc:
                    fc["pair"] = fc["pair"].split(" | ")[1]
    _edit(t / ENV, f)


# --- suite mutants ---------------------------------------------------------

def m_control_neutered(t):
    """Neuter one control IN PLACE, keeping the count.  Pinning the count does
    not catch this -- the AST guard has to."""
    _sub(t / pathlib.Path(SUITE), '    check("offset_control", ok,',
         '    check("offset_control", True,')


def m_rebind_removed(t):
    """Remove the rebind that ARMS reader_reads_the_move.  Without it the
    constant and the emitted move agree and the control passes vacuously --
    the "a fix disarms the probe" failure, so the arming half must fire."""
    _sub(t / pathlib.Path(SUITE),
         "        if rebind_after is not None:\n"
         "            C.CUTLERY_DESCEND_OPENING = rebind_after\n",
         "")


def m_taut_guard_blinded(t):
    """The tautology guard always reports nothing.  Its own negative control
    has to notice."""
    _sub(t / pathlib.Path(SUITE), "    import ast\n\n    out = []",
         "    import ast\n\n    return []\n    out = []")


def m_reject_deleted(t):
    """Delete a control's negative half.  The reject pin has to notice."""
    _sub(t / pathlib.Path(SUITE), '    check("offset_control/reject",',
         '    _skip = lambda *a, **k: None\n    _skip("offset_control/reject",')


MUTANTS = [
    ("opening_knob_drifted", "CUTLERY_DESCEND_OPENING ships at 0.30",
     "defaults", m_opening_knob_drifted),
    ("plan_knob_drifted", "CUTLERY_PLAN_AT_OPEN ships True",
     "defaults", m_plan_knob_drifted),
    ("plan_wiring_inert", "the descend move ignores the plan_at knob",
     "knob_live", m_plan_wiring_inert),
    ("opening_wiring_inert", "the fork pick ignores the opening knob",
     "knob_live", m_opening_wiring_inert),
    ("reader_reads_the_constant", "the suite reads the constant instead of the "
     "emitted move", "reader_reads_the_move", m_reader_reads_the_constant),
    ("rebind_removed", "the rebind that arms reader_reads_the_move is removed",
     "reader_reads_the_move", m_rebind_removed),
    ("taut_guard_blinded", "the tautology guard always reports nothing",
     "no_tautological_checks/reject", m_taut_guard_blinded),
    ("offset_zero", "the mismatch introduces no displacement",
     "offset_is_half", m_offset_zero),
    ("offset_doubled", "the displacement stops being half the width change",
     "offset_is_half", m_offset_doubled),
    ("control_offsets_too", "the no-plan_at control shows the same offset",
     "offset_control", m_control_offsets_too),
    ("control_has_plan_at", "the approach waypoints are recorded with a plan_at",
     "offset_control", m_control_has_plan_at),
    ("offset_is_fore_aft", "the displacement is recorded along the wrist axis",
     "offset_sideways", m_offset_is_fore_aft),
    ("midpoint_disagrees", "the independent sweep disagrees at the same opening",
     "offset_sideways", m_midpoint_disagrees),
    ("shipped_baseline_drift", "the shipped cell's stall drifts 9 mm off the "
     "independently published one", "reproduces", m_shipped_baseline_drift),
    ("shipped_total_drift", "the shipped cell stops reproducing 15/50",
     "reproduces", m_shipped_total_drift),
    ("placement_claimed", "a cell places the fork 3/10",
     "verdict", m_placement_claimed),
    ("verdict_flipped", "the stored verdict stops following its own table",
     "verdict", m_verdict_flipped),
    ("placement_without_lift", "a placement is claimed with no lift",
     "placement_lift", m_placement_without_lift),
    ("run_dropped", "a rollout is silently lost from one cell",
     "identity", m_run_dropped),
    ("probe_control_false", "the probe's own control is False",
     "probe_controls_envelope", m_probe_control_false),
    ("contact_one_sided", "first contacts stop naming the hand side",
     "contact_named", m_contact_one_sided),
    ("control_neutered", "one control is turned into a tautology in place, "
     "keeping the count", "no_tautological_checks", m_control_neutered),
    ("reject_deleted", "one control's negative half is removed",
     "reject_count_pinned", m_reject_deleted),
]


def _baseline(tag: str) -> tuple[bool, str]:
    d = tempfile.mkdtemp(prefix=f"grenv_{tag}_")
    tree = pathlib.Path(d) / "tree"
    stage(tree)
    rc, out = run_suite(tree)
    shutil.rmtree(d, ignore_errors=True)
    return rc == 0, out


def main() -> int:
    green_before, out = _baseline("base_before")
    print(f"baseline in a fresh copied tree BEFORE: "
          f"{'GREEN' if green_before else 'RED -- campaign is void'}")
    if not green_before:
        print(out[-4000:])   # only the tail is PRINTED; parsing sees it all
        raise SystemExit("baseline must be green before any mutant is scored")

    results, killed, by_named, broken = [], 0, 0, 0
    for name, what, expect, fn in MUTANTS:
        tmp = tempfile.mkdtemp(prefix=f"grenv_{name}_")
        tree = pathlib.Path(tmp) / "tree"
        stage(tree)
        try:
            fn(tree)
        except NoOp as exc:
            broken += 1
            results.append({"mutant": name, "breaks": what,
                            "expected_control": expect, "status": "BROKEN",
                            "reason": str(exc)})
            print(f"  BROKEN   {name:26s} -> {exc}")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        rc, out = run_suite(tree)
        fails = failed_controls(out)
        red = rc != 0
        named = expect in fails
        killed += bool(red)
        by_named += bool(named)
        results.append({"mutant": name, "breaks": what,
                        "expected_control": expect, "rc": rc,
                        "status": "KILLED" if red else "SURVIVED",
                        "killed_by_expected_control": bool(named),
                        "failed_controls": fails,
                        "tail": out[-2000:]})
        print(f"  {'KILLED  ' if red else 'SURVIVED'} {name:26s} -> "
              f"{', '.join(fails) or '(none)'}")
        shutil.rmtree(tmp, ignore_errors=True)

    green_after, _ = _baseline("base_after")
    print(f"baseline in a fresh copied tree AFTER: "
          f"{'GREEN' if green_after else 'RED'}")

    scored = len(MUTANTS) - broken
    doc = {
        "campaign": "mutate_gripper_envelope.py",
        "suite": SUITE,
        "baseline_green_before": green_before,
        "baseline_green_after": green_after,
        "isolation": "each mutant runs in its own fresh mkdtemp tree, copied "
                     "not hard-linked, so a mutant cannot write the real "
                     "evidence files",
        "no_op_guard": "a mutation whose output is byte-identical to its input "
                       "raises and is reported BROKEN, never SURVIVED",
        "n": len(MUTANTS), "scored": scored, "broken": broken,
        "killed": killed, "survived": scored - killed,
        "killed_by_the_control_that_names_them": by_named,
        "not_claimed": "A mutation score measures how sensitive this suite is "
                       "to these edits. It is not evidence that the controller "
                       "is correct, that the refutation is right, or that any "
                       "cutlery was placed -- none was.",
        "results": results,
    }
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "gripper_envelope_mutants.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{killed}/{scored} killed ({broken} broken), {by_named} by the "
          f"control that names them")
    return 0 if (killed == scored and green_after and broken == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
