#!/usr/bin/env python3
"""Mutation campaign for ``scripts/test_cutlery_approach.py``.

The suite's job is to make a REFUTATION checkable.  A refutation is the easiest
kind of claim to assert vacuously: "nothing was placed" is what a suite that
reads nothing at all also reports.  So each mutant below breaks exactly one
thing the suite says it protects, and the suite has to go red for it.

Ten mutants, in two families:

  controller (2)  the new ``CUTLERY_APPROACH`` knob -- its default drifting off
                  the shipped pure +z, and the knob being wired up inertly so
                  the sweep swept nothing.  These are the mutants that would
                  invalidate every published figure in the repository, which is
                  why the knob was added with a control rather than without.
  evidence   (8)  one per outcome the suite reads out of
                  ``evidence/cutlery_approach.json``.

Run in an isolated COPY of the tree, never the tree itself: a mutant that runs
in place can overwrite the real evidence file, and a hard-linked copy is the
same hazard wearing a different name.  The baseline is asserted GREEN in the
copy before any mutant runs -- a campaign whose baseline is already red scores
100% and means nothing.

Writes evidence/cutlery_approach_mutants.json.
Run:  python3 scripts/mutate_cutlery_approach.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
SUITE = "scripts/test_cutlery_approach.py"
DOC = "evidence/cutlery_approach.json"
CTRL = "envs/controller.py"


def stage(dst: pathlib.Path) -> None:
    """A fresh tree with only what the suite needs, copied not linked."""
    for d in ("envs", "scripts", "third_party"):
        shutil.copytree(ROOT / d, dst / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "evidence").mkdir()
    shutil.copy2(ROOT / DOC, dst / DOC)


def run_suite(tree: pathlib.Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, SUITE], cwd=tree,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


# --- mutants ---------------------------------------------------------------

def m_default_drift(t: pathlib.Path) -> None:
    f = t / CTRL
    s = f.read_text()
    s = s.replace("CUTLERY_APPROACH = (0.0, 0.0, 0.075)",
                  "CUTLERY_APPROACH = (0.0, -0.040, 0.075)")
    f.write_text(s)


def m_knob_inert(t: pathlib.Path) -> None:
    f = t / CTRL
    s = f.read_text()
    s = s.replace("approach=CUTLERY_APPROACH, lift=(0.0, 0.0, 0.085)))",
                  "approach=(0.0, 0.0, 0.075), lift=(0.0, 0.0, 0.085)))")
    f.write_text(s)


def _doc(t: pathlib.Path):
    p = t / DOC
    return p, json.loads(p.read_text())


def m_identity(t):
    p, d = _doc(t)
    for v in d["variants"]:
        v["shipped"] = abs(v["dy_m"] - (-0.040)) < 1e-12 and not v["square"]
    p.write_text(json.dumps(d, indent=1))


def m_reproduces(t):
    p, d = _doc(t)
    for v in d["variants"]:
        if v["shipped"]:
            v["fork_stall_mm_median"] = 21.0
    p.write_text(json.dumps(d, indent=1))


def m_refutation(t):
    p, d = _doc(t)
    d["variants"][0]["fork_placed"] = 2
    p.write_text(json.dumps(d, indent=1))


def m_neg_control(t):
    p, d = _doc(t)
    d["variants"][0]["spoon_placed"] = 1
    p.write_text(json.dumps(d, indent=1))


def m_not_inert(t):
    p, d = _doc(t)
    for v in d["variants"]:
        v["fork_blocking_contacts"] = [{"geom": "fork_handle", "steps": 5288}]
        v["fork_shove_mm_median"] = 4.1
    p.write_text(json.dumps(d, indent=1))


def m_wall_order(t):
    p, d = _doc(t)
    for v in d["variants"]:
        v["fork_stall_mm_median"] = 40.0 + 100.0 * v["dy_m"]
        v["fork_blocking_contacts"] = [{"geom": "drawer_back", "steps": 4000}]
    p.write_text(json.dumps(d, indent=1))


def m_near_miss(t):
    p, d = _doc(t)
    d["variants"][0]["fork_lifted_mm_median"] = 12.0
    p.write_text(json.dumps(d, indent=1))


def m_no_claim(t):
    p, d = _doc(t)
    d["variants"][0]["subgoals_met_total"] = 19
    p.write_text(json.dumps(d, indent=1))


def m_error_rows(t):
    """A sweep that silently lost runs still reads 0/10 everywhere."""
    p, d = _doc(t)
    d["runs"][0] = {"seed": 0, "dy_m": 0.0, "square": False,
                    "error": "RuntimeError: dropped"}
    p.write_text(json.dumps(d, indent=1))


MUTANTS = [
    ("default_drift", "CUTLERY_APPROACH default moves off the shipped +z",
     "defaults", m_default_drift),
    ("knob_inert", "the cutlery _pick calls ignore the knob",
     "knob_live", m_knob_inert),
    ("identity", "the shipped flag is moved to another variant",
     "identity", m_identity),
    ("reproduces", "the shipped stall no longer matches the published 24.5 mm",
     "reproduces", m_reproduces),
    ("refutation", "one variant places the fork 2/10",
     "refutation", m_refutation),
    ("neg_control", "one variant places the spoon",
     "neg_control", m_neg_control),
    ("not_inert", "every variant wears the same wall and the same shove",
     "not_inert", m_not_inert),
    ("wall_order", "stalls ordered the way the hypothesis predicted",
     "wall_is_not_y", m_wall_order),
    ("near_miss", "one variant lifts the fork 12 mm",
     "not_near_miss", m_near_miss),
    ("no_claim", "one variant beats the shipped 15/50",
     "no_claim", m_no_claim),
    ("error_rows", "a run is silently dropped with an error",
     "identity", m_error_rows),
]


def main() -> int:
    results = []
    base_dir = tempfile.mkdtemp(prefix="cutapp_base_")
    base = pathlib.Path(base_dir) / "tree"          # fresh, never pre-existing
    stage(base)
    rc, out = run_suite(base)
    baseline_green = rc == 0
    print(f"baseline in copied tree: rc={rc} "
          f"{'GREEN' if baseline_green else 'RED -- campaign is void'}")
    if not baseline_green:
        print(out)
        shutil.rmtree(base_dir, ignore_errors=True)
        raise SystemExit("baseline must be green before any mutant is scored")
    shutil.rmtree(base_dir, ignore_errors=True)

    killed = 0
    for name, what, expect, fn in MUTANTS:
        tmp = tempfile.mkdtemp(prefix=f"cutapp_{name}_")
        tree = pathlib.Path(tmp) / "tree"
        stage(tree)
        fn(tree)
        rc, out = run_suite(tree)
        red = rc != 0
        named = f"FAILED:" in out and expect in out.split("FAILED:")[-1]
        killed += bool(red)
        results.append({"mutant": name, "breaks": what,
                        "expected_control": expect, "rc": rc,
                        "killed": bool(red),
                        "killed_by_expected_control": bool(named),
                        "failed_controls": (out.split("FAILED:")[-1].strip()
                                            if "FAILED:" in out else "")})
        print(f"  {'KILLED ' if red else 'SURVIVED'} {name:14s} "
              f"-> {results[-1]['failed_controls'] or '(none)'}")
        shutil.rmtree(tmp, ignore_errors=True)

    doc = {
        "campaign": "mutate_cutlery_approach.py",
        "suite": SUITE,
        "baseline_green_in_copied_tree": baseline_green,
        "isolation": "each mutant runs in its own fresh mkdtemp tree, copied "
                     "not hard-linked, so a mutant cannot write the real "
                     "evidence file",
        "n": len(MUTANTS), "killed": killed,
        "killed_by_the_control_that_names_them": sum(
            r["killed_by_expected_control"] for r in results),
        "not_claimed": "A mutation score measures how sensitive this suite is "
                       "to these eleven edits. It is not evidence that the "
                       "refutation is correct, and it is not a placement.",
        "results": results,
    }
    (EVID / "cutlery_approach_mutants.json").write_text(json.dumps(doc, indent=1))
    print(f"\n{killed}/{len(MUTANTS)} killed, "
          f"{doc['killed_by_the_control_that_names_them']}/{len(MUTANTS)} by "
          "the control that names them")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
