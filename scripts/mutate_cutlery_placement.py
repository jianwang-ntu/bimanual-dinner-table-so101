#!/usr/bin/env python3
"""Mutation campaign for ``scripts/test_cutlery_placement.py``.

The suite protects two claims that are easy to assert vacuously:

  "the randomizer ships off, so no published figure moved" -- which a suite
  that reads no constant at all also reports; and
  "no seat places the cutlery" -- which a sweep that never varied the seat,
  or that quietly lost its rollouts, also reports.

So every mutant below breaks exactly one thing the suite says it protects, and
the suite has to go red, ideally on the control that names it.

Twenty-three mutants in three families:

  code      (5)  the three new switches drifting off the shipped scene, and
                 two ways of disarming the penetration test.  These are the mutants
                 that would silently invalidate every figure in the
                 repository, which is why the switches were added with
                 controls rather than without.
  evidence (15)  one per outcome the suite reads out of the three probe files,
                 including the two that this tree has actually got wrong
                 before: an infeasible cell relabelled as legal so its
                 placement counts, and a sweep that lost rollouts.
  suite     (1)  a control deleted from the suite itself.  A campaign that
                 cannot notice its own controls disappearing is measuring the
                 wrong thing.

Each mutant runs in its own fresh ``mkdtemp`` tree, COPIED not hard-linked, so
no mutant can write the real evidence files.  The baseline is asserted GREEN in
a copied tree first -- a campaign whose baseline is already red scores 100% and
means nothing.

Writes evidence/cutlery_placement_mutants.json.
Run:  python3 scripts/mutate_cutlery_placement.py
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
SUITE = "scripts/test_cutlery_placement.py"
PLACEMENT = "evidence/cutlery_placement.json"
SEAT = "evidence/cutlery_seat.json"
RAND = "envs/randomize.py"
TRACKING = "evidence/descent_tracking.json"


def stage(dst: pathlib.Path) -> None:
    for d in ("envs", "scripts"):
        shutil.copytree(ROOT / d, dst / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "evidence").mkdir()
    for f in (PLACEMENT, SEAT, TRACKING):
        shutil.copy2(ROOT / f, dst / f)


def run_suite(tree: pathlib.Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, SUITE], cwd=tree,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout + p.stderr)[-8000:]


def failed_controls(out: str) -> list[str]:
    return re.findall(r"\[FAIL\] ([a-z0-9_]+):", out)


def _sub(path: pathlib.Path, old: str, new: str) -> None:
    """Rewrite `old` -> `new`, and refuse to run if `old` was not there.

    A mutant whose edit silently misses is a mutant the suite "kills" for free.
    That happened once in this campaign already, on 2026-09-08: the string had
    moved and the no-op mutant was scored as a survivor of a control that was
    never exercised.
    """
    s = path.read_text()
    if old not in s:
        raise SystemExit(f"mutant is a no-op: {old!r} not found in {path.name}")
    path.write_text(s.replace(old, new, 1))


def _load(t: pathlib.Path, rel: str):
    p = t / rel
    return p, json.loads(p.read_text())


# --- code mutants ----------------------------------------------------------

def m_spread_on(t):
    _sub(t / RAND, "CUTLERY_PLACE_SPREAD = 0.0", "CUTLERY_PLACE_SPREAD = 1.0")


def m_seat_moved(t):
    _sub(t / RAND, "CUTLERY_SEAT_XY = (0.0, 0.0)",
         "CUTLERY_SEAT_XY = (0.0, -0.02)")


def m_splay_on(t):
    _sub(t / RAND, "CUTLERY_SPLAY_X = 0.0", "CUTLERY_SPLAY_X = 0.018")


def m_no_penetration_test(t):
    """The accept stops being guarded by the contact pass."""
    _sub(t / RAND, "if pen >= PENETRATION_TOL_M:", "if True:")


def m_penetration_never_computed(t):
    """The guard survives but the value it reads is a constant."""
    _sub(t / RAND,
         "drawn, pen = (dx, dy, dyaw), _penetrates(model, data, bid)",
         "drawn, pen = (dx, dy, dyaw), 0.0")


# --- placement-evidence mutants --------------------------------------------

def m_baseline_drift(t):
    p, d = _load(t, PLACEMENT)
    for v in d["variants"]:
        if v["shipped_before"]:
            v["per_goal"]["plate_placed"] = 6
    p.write_text(json.dumps(d, indent=1))


def m_zero_arm_moved(t):
    p, d = _load(t, PLACEMENT)
    for v in d["variants"]:
        if v["shipped_before"]:
            v["fork_start_spread_mm"] = {"x": 3.0, "y": 3.0}
    p.write_text(json.dumps(d, indent=1))


def m_spread_arms_flat(t):
    p, d = _load(t, PLACEMENT)
    for v in d["variants"]:
        if not v["shipped_before"]:
            v["fork_start_spread_mm"] = {"x": 0.0, "y": 0.0}
            v["spoon_start_spread_mm"] = {"x": 0.0, "y": 0.0}
    p.write_text(json.dumps(d, indent=1))


def m_upstream_reshuffled(t):
    p, d = _load(t, PLACEMENT)
    for r in d["runs"]:
        if not r.get("shipped_before"):
            r["upstream"]["drawer_slide"] = 0.0123
            break
    p.write_text(json.dumps(d, indent=1))


def m_cost_misquoted(t):
    p, d = _load(t, PLACEMENT)
    for v in d["variants"]:
        if v["spread"] == 1.0:
            v["subgoals_met_total"] = 15
    p.write_text(json.dumps(d, indent=1))


def m_placement_claimed(t):
    p, d = _load(t, PLACEMENT)
    d["variants"][-1]["per_goal"]["fork_placed"] = 3
    p.write_text(json.dumps(d, indent=1))


def m_probe_control_false(t):
    p, d = _load(t, PLACEMENT)
    d["controls"]["upstream_untouched"] = False
    p.write_text(json.dumps(d, indent=1))


def m_run_dropped(t):
    p, d = _load(t, PLACEMENT)
    d["runs"][0] = {"seed": 0, "spread": 0.0, "error": "RuntimeError: dropped"}
    p.write_text(json.dumps(d, indent=1))


# --- seat-evidence mutants --------------------------------------------------

def m_seat_baseline_drift(t):
    p, d = _load(t, SEAT)
    for c in d["cells"]:
        if c["shipped"]:
            c["per_goal"]["mug_placed"] = 3
    p.write_text(json.dumps(d, indent=1))


def m_feasible_success(t):
    p, d = _load(t, SEAT)
    for c in d["cells"]:
        if c["worst_penetration_mm"] >= -2.0 and not c["shipped"]:
            c["per_goal"]["fork_placed"] = 2
            break
    p.write_text(json.dumps(d, indent=1))


def m_infeasible_relabelled(t):
    """The one cell that placed a spoon is relabelled as a legal scene.

    This is the mutant that matters most: it is the shape of the mistake a
    reader makes on their own, by seeing `spoon 1` in the grid and reporting a
    placement without looking at the penetration column beside it.
    """
    p, d = _load(t, SEAT)
    for c in d["cells"]:
        if c["per_goal"]["fork_placed"] or c["per_goal"]["spoon_placed"]:
            c["worst_penetration_mm"] = 0.0
    p.write_text(json.dumps(d, indent=1))


def m_reach_flattened(t):
    p, d = _load(t, SEAT)
    for c in d["cells"]:
        c["fork_reach_mm"] = 386.7
    p.write_text(json.dumps(d, indent=1))


def m_stall_tracks_reach(t):
    p, d = _load(t, SEAT)
    for c in d["cells"]:
        c["fork_stall_mm_median"] = round(c["fork_reach_mm"] / 10.0, 1)
    p.write_text(json.dumps(d, indent=1))


def m_squaring_no_better(t):
    p, d = _load(t, SEAT)
    for c in d["cells"]:
        if c["square"]:
            c["fork_stall_mm_median"] = 30.0
    p.write_text(json.dumps(d, indent=1))


def m_no_infeasible_cells(t):
    """Drop every cell the drawer rejected, so the legal span looks like the
    whole span anybody ever tried."""
    p, d = _load(t, SEAT)
    d["cells"] = [c for c in d["cells"] if c["worst_penetration_mm"] >= -2.0]
    keep = {(c["seat_dy"], c["splay"], c["square"]) for c in d["cells"]}
    d["runs"] = [r for r in d["runs"]
                 if (r.get("seat_dy"), r.get("splay"), r.get("square")) in keep]
    p.write_text(json.dumps(d, indent=1))


# --- tracking-evidence mutants ----------------------------------------------

def m_solver_blamed(t):
    p, d = _load(t, TRACKING)
    for k in ("fork_descend", "spoon_descend"):
        d["waypoints"][k]["ik_err_mm_median"] = 30.0
    p.write_text(json.dumps(d, indent=1))


def m_control_saturates_too(t):
    """The free via-points saturate as well, so the probe discriminates
    nothing -- and the correction it supports would be unfounded."""
    p, d = _load(t, TRACKING)
    for k in ("fork_above", "spoon_above"):
        d["waypoints"][k]["saturated_joints_per_seed_median"] = 2.0
    p.write_text(json.dumps(d, indent=1))


# --- suite mutant -----------------------------------------------------------

def m_control_deleted(t):
    """Delete one control from the suite.  The pin has to notice."""
    _sub(t / SUITE, '''    ok &= check("cutlery_is_still_never_placed",''',
         '''    if False: ok &= check("cutlery_is_still_never_placed",''')


MUTANTS = [
    ("spread_on", "CUTLERY_PLACE_SPREAD ships at 1.0", "switches_ship_off",
     m_spread_on),
    ("seat_moved", "CUTLERY_SEAT_XY ships 20 mm forward", "switches_ship_off",
     m_seat_moved),
    ("splay_on", "CUTLERY_SPLAY_X ships at 18 mm", "switches_ship_off",
     m_splay_on),
    ("no_penetration_test", "the accept stops being guarded by the contact "
     "pass", "draws_are_rejected_by_the_contact_pass", m_no_penetration_test),
    ("penetration_never_computed", "the guard reads a constant instead of the "
     "contact pass", "draws_are_rejected_by_the_contact_pass",
     m_penetration_never_computed),
    ("baseline_drift", "the spread-0 baseline no longer scores the published "
     "plate 4/10", "default_reproduces_the_published_tally", m_baseline_drift),
    ("zero_arm_moved", "the spread-0 arm reports a non-zero spread",
     "spread_zero_really_is_zero", m_zero_arm_moved),
    ("spread_arms_flat", "the jittered arms report no spread at all",
     "spread_one_really_moves", m_spread_arms_flat),
    ("upstream_reshuffled", "one seed's drawer draw differs between arms",
     "plate_mug_bottle_drawer_are_untouched", m_upstream_reshuffled),
    ("cost_misquoted", "full spread is recorded as costing nothing",
     "the_cost_of_switching_on_is_measured", m_cost_misquoted),
    ("placement_claimed", "a jittered arm places the fork 3/10",
     "cutlery_is_still_never_placed", m_placement_claimed),
    ("probe_control_false", "the probe's own control is False and ignored",
     "probe_controls_all_pass", m_probe_control_false),
    ("run_dropped", "a rollout is silently lost from the sweep",
     "no_run_was_silently_dropped", m_run_dropped),
    ("seat_baseline_drift", "the shipped seat cell no longer reproduces",
     "seat_baseline_reproduces_the_published_tally", m_seat_baseline_drift),
    ("feasible_success", "a legal seat places the fork 2/10",
     "every_feasible_cell_places_no_cutlery", m_feasible_success),
    ("infeasible_relabelled", "the buried cell that placed a spoon is called "
     "legal", "every_feasible_cell_places_no_cutlery", m_infeasible_relabelled),
    ("reach_flattened", "every cell has the baseline's reach",
     "the_sweep_actually_moved_the_reach", m_reach_flattened),
    ("stall_tracks_reach", "the stall is made proportional to the reach",
     "the_stall_does_not_track_the_reach", m_stall_tracks_reach),
    ("squaring_no_better", "squaring stops lowering the stall",
     "squaring_lowers_the_stall_and_still_places_nothing", m_squaring_no_better),
    ("no_infeasible_cells", "the rejected wider seats are dropped from the grid",
     "wider_seats_were_tried_and_are_infeasible", m_no_infeasible_cells),
    ("solver_blamed", "the IK residual is as large as the miss",
     "the_solver_is_not_the_failure", m_solver_blamed),
    ("control_saturates_too", "the free via-points saturate as well",
     "only_the_waypoints_in_contact_saturate", m_control_saturates_too),
    ("control_deleted", "one control is removed from the suite",
     "control_count_is_pinned", m_control_deleted),
]


def main() -> int:
    base_dir = tempfile.mkdtemp(prefix="cutplace_base_")
    base = pathlib.Path(base_dir) / "tree"
    stage(base)
    rc, out = run_suite(base)
    green = rc == 0
    print(f"baseline in copied tree: rc={rc} "
          f"{'GREEN' if green else 'RED -- campaign is void'}")
    if not green:
        print(out[-3000:])
        shutil.rmtree(base_dir, ignore_errors=True)
        raise SystemExit("baseline must be green before any mutant is scored")
    shutil.rmtree(base_dir, ignore_errors=True)

    results, killed, by_named = [], 0, 0
    for name, what, expect, fn in MUTANTS:
        tmp = tempfile.mkdtemp(prefix=f"cutplace_{name}_")
        tree = pathlib.Path(tmp) / "tree"
        stage(tree)
        fn(tree)
        rc, out = run_suite(tree)
        fails = failed_controls(out)
        red = rc != 0
        named = expect in fails
        killed += bool(red)
        by_named += bool(named)
        results.append({"mutant": name, "breaks": what,
                        "expected_control": expect, "rc": rc,
                        "killed": bool(red),
                        "killed_by_expected_control": bool(named),
                        "failed_controls": fails})
        print(f"  {'KILLED ' if red else 'SURVIVED'} {name:22s} -> "
              f"{', '.join(fails) or '(none)'}")
        shutil.rmtree(tmp, ignore_errors=True)

    doc = {
        "campaign": "mutate_cutlery_placement.py",
        "suite": SUITE,
        "baseline_green_in_copied_tree": green,
        "isolation": "each mutant runs in its own fresh mkdtemp tree, copied "
                     "not hard-linked, so a mutant cannot write the real "
                     "evidence files",
        "n": len(MUTANTS), "killed": killed,
        "killed_by_the_control_that_names_them": by_named,
        "not_claimed": "A mutation score measures how sensitive this suite is "
                       "to these twenty-three edits. It is not evidence that the "
                       "randomizer is correct, that the seat sweep's "
                       "refutation is right, or that anything was placed.",
        "results": results,
    }
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "cutlery_placement_mutants.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{killed}/{len(MUTANTS)} killed, {by_named} by the control that "
          f"names them")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
