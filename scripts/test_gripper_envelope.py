#!/usr/bin/env python3
"""Controls for the gripper-envelope sweep and the jaw-midpoint measurement.

``scripts/measure_gripper_envelope.py`` tested the eighth and last candidate
explanation of the cutlery stall named in ``TECHNICAL_SUMMARY.md`` section 8:
the descent is EXECUTED at one gripper opening and SOLVED at another, so the
jaw meeting point ``plan_pose`` puts on the target is not the meeting point
that goes down.  The mechanism is REAL and is measured here.  As an explanation
of the stall it is REFUTED: closing it in every direction places no cutlery.
This suite makes both halves checkable rather than readable.

The control that matters most is ``defaults``.  The sweep needed two new
module-level knobs, ``CUTLERY_DESCEND_OPENING`` and ``CUTLERY_PLAN_AT_OPEN``,
and a knob whose default silently moved the shipped behaviour would invalidate
every published figure in this repository at once.  So the suite drives
``dinner_table_script()`` itself and reads the ``opening`` and ``plan_at`` the
emitted cutlery DESCEND moves actually carry, rather than reading the constants
or the source text.

  defaults        ACCEPT the committed defaults emit descend moves carrying
                  plan_at present and opening=GRIPPER_NARROW for BOTH the fork
                  and the spoon -- the exact pair every published figure was
                  measured at; REJECT the same reader with a knob moved
  knob_live       ACCEPT each knob actually changes what is emitted -- an inert
                  knob would produce this sweep's null result for a reason with
                  nothing to do with the gripper; REJECT a reader that ignores
                  the knob
  mismatch_is_real ACCEPT the shipped descend move's two widths are DIFFERENT
                  numbers, read off the emitted move, so the thing this probe
                  is named after exists before anything is measured about it;
                  REJECT a reader whose tolerance is too wide to see the gap
  identity        ACCEPT exactly one cell is flagged shipped, it is
                  (opening=shipped, plan_at_open=False), every cell ran the
                  full seed count and no run errored; REJECT the flag moved
  reproduces      ACCEPT the shipped cell reproduces what an INDEPENDENT probe
                  already publishes -- fork 0/10, spoon 0/10, 15/50 sub-goals,
                  24.5 mm and 36.4 mm median stalls in
                  evidence/cutlery_approach.json -- so this harness is shown to
                  agree with the published numbers before it argues anything;
                  REJECT a shifted copy
  offset_is_half  ACCEPT the executed-vs-solved displacement equals HALF the
                  change in jaw separation, derived from the artifact's own two
                  width numbers rather than from a literal -- which is what
                  "only one jaw moves" predicts and is a far stronger statement
                  than agreeing at one width; REJECT a doubled copy
  offset_control  ACCEPT the approach waypoints -- same arithmetic, same hand,
                  same seeds, no plan_at -- show an offset of exactly zero, so
                  the descend number is the mismatch and not this probe's
                  algebra; REJECT a copy where the control also shows an offset
  offset_sideways ACCEPT the displacement is along the JAW axis, not the
                  approach axis, and agrees with the independent full-range
                  sweep in evidence/jaw_midpoint_shift.json; REJECT a copy with
                  the two components swapped
  contact_named   ACCEPT every recorded first contact names both sides, hand
                  geom and world geom; REJECT the same predicate against a
                  one-sided copy built from a real entry
  verdict         ACCEPT the verdict stored in the evidence is the one its own
                  cell table forces; REJECT a copy with one placement planted,
                  which must force the other verdict
  placement_lift  ACCEPT no cell claims a placement it has no lift for; REJECT
                  a planted cell that places without lifting
  gain_not_cutlery ACCEPT the best cell's advantage over shipped is entirely
                  outside the cutlery -- every cell is 0/10 on both pieces --
                  and the plate count's own spread across the sweep contains
                  it, so nothing is adopted on it; REJECT a copy where a cell
                  really does place cutlery
  probe_controls  ACCEPT both probes' own controls are green

Standard library only, except ``defaults``/``knob_live``/``mismatch_is_real``,
which must import the controller because the claim is about what the controller
emits.  Check and control counts are PINNED: a control that VANISHES is itself
a failure.
Run:  python3 scripts/test_gripper_envelope.py
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DOC = EVID / "gripper_envelope.json"
MID = EVID / "jaw_midpoint_shift.json"
INDEP = EVID / "cutlery_approach.json"

DESC = ("fork_descend", "spoon_descend")
ABOVE = ("fork_above", "spoon_above")

# Pinned counts, excluding the two pinned-count checks themselves.
EXPECT_CHECKS = 33
EXPECT_REJECTS = 15

FAILED: list[str] = []
PASSED: list[str] = []

# Set by ``descend_widths`` to the value of ``CUTLERY_DESCEND_OPENING`` at the
# moment the emitted moves were read.  It is what ARMS ``reader_reads_the_move``:
# without it, removing the rebind would leave that control passing vacuously,
# which is the "a fix disarms the probe that proved the defect" failure.
LAST_CONSTANT_AT_READ: float | None = None


def check(name: str, ok: bool, detail) -> bool:
    (PASSED if ok else FAILED).append(name)
    print(("  ok   " if ok else "  FAIL ") + name + " -- " + str(detail))
    return ok


def cell(d: dict, opening, plan_at_open: bool) -> dict:
    for c in d["cells"]:
        same = (c["opening"] is None if opening is None
                else (c["opening"] is not None
                      and abs(c["opening"] - opening) < 1e-12))
        if same and bool(c["plan_at_open"]) is bool(plan_at_open):
            return c
    raise SystemExit(f"no cell opening={opening} plan_at_open={plan_at_open}")


def tautological_checks(src: str) -> list[str]:
    """Names of ``check()`` calls whose verdict is a literal, from the AST.

    A text scan is not enough: a comment or a docstring explaining the fix
    would satisfy one.  This walks the parsed tree and looks at the second
    positional argument of every ``check(...)`` call.
    """
    import ast

    out = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "check"
                and len(node.args) >= 2):
            continue
        v = node.args[1]
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not):
            v = v.operand
        if isinstance(v, ast.Constant):
            name = (node.args[0].value if isinstance(node.args[0], ast.Constant)
                    else "?")
            out.append(str(name))
    return out


def verdict_of(d: dict) -> str:
    """The verdict the cell table forces, recomputed here from the table."""
    return ("SUPPORTED"
            if any(c["fork_placed"] + c["spoon_placed"] > 0 for c in d["cells"])
            else "REFUTED")


# --- controls that must import the controller -------------------------------

def descend_widths(opening_knob=None, plan_knob=None,
                   rebind_after=None) -> dict[str, tuple]:
    """(opening, plan_at) each emitted cutlery descend move actually carries.

    Read out of the emitted script, not out of the constants: the claim is
    about what ``dinner_table_script()`` produces.  Callables are resolved the
    same way ``Rollout._plan`` resolves them.

    ``rebind_after`` moves ``CUTLERY_DESCEND_OPENING`` AFTER the script has
    been built and before the moves are read.  The emitted moves captured the
    old value, so a reader that really reads the move returns it and a reader
    that has quietly fallen back to the constant returns the new one.  Without
    that separation the two readers agree on every input and the distinction
    this function exists for is untestable -- which is how a mutant that
    replaced the move read with the constant survived a campaign on
    2026-09-08.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    from envs.randomize import make_env
    from envs import controller as C, scene_source

    model, data, _ = make_env(0)
    scene_source.install(scene_source.make("privileged"))
    keep = (C.CUTLERY_DESCEND_OPENING, C.CUTLERY_PLAN_AT_OPEN)
    try:
        if opening_knob is not None:
            C.CUTLERY_DESCEND_OPENING = opening_knob
        if plan_knob is not None:
            C.CUTLERY_PLAN_AT_OPEN = plan_knob
        script = C.dinner_table_script()
        if rebind_after is not None:
            C.CUTLERY_DESCEND_OPENING = rebind_after
        global LAST_CONSTANT_AT_READ
        LAST_CONSTANT_AT_READ = float(C.CUTLERY_DESCEND_OPENING)
        grips = {a: C.Gripper(model, a) for a in ("left", "right")}
        out = {}
        for entry in script:
            if isinstance(entry, tuple) and entry and entry[0] == "if":
                continue
            for mv in entry[0].values():
                lb = getattr(mv, "label", "")
                if lb not in DESC:
                    continue
                g = grips[mv.arm]
                op = mv.opening(model, data, g) if callable(mv.opening) else mv.opening
                pa = mv.plan_at(model, data, g) if callable(mv.plan_at) else mv.plan_at
                out[lb] = (None if op is None else round(float(op), 6),
                           None if pa is None else round(float(pa), 6))
        return out
    finally:
        C.CUTLERY_DESCEND_OPENING, C.CUTLERY_PLAN_AT_OPEN = keep


def shipped_narrow() -> float:
    sys.path.insert(0, str(ROOT))
    from envs import controller as C
    return round(float(C.GRIPPER_NARROW), 6)


def commanded_lift_mm() -> float:
    """How far above its grasp site the emitted ``fork_lift`` move asks the
    fork to go.  The bar for "was it lifted" has to come from what the
    controller ASKED for, not from a threshold chosen after seeing the answer.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from envs.randomize import make_env
    from envs import controller as C, scene_source

    model, data, _ = make_env(0)
    scene_source.install(scene_source.make("privileged"))
    sid = C.mujoco.mj_name2id(model, C.mujoco.mjtObj.mjOBJ_SITE, "fork_grasp")
    for entry in C.dinner_table_script():
        if isinstance(entry, tuple) and entry and entry[0] == "if":
            continue
        for mv in entry[0].values():
            if getattr(mv, "label", "") == "fork_lift" and mv.where is not None:
                tgt = np.asarray(mv.where(model, data), float)
                return round(float(tgt[2] - data.site_xpos[sid][2]) * 1000, 2)
    raise SystemExit("no fork_lift move in the emitted script")


def main() -> int:
    d = json.loads(DOC.read_text())
    mid = json.loads(MID.read_text())
    ind = json.loads(INDEP.read_text())
    ish = next(v for v in ind["variants"] if v.get("shipped"))
    print(f"controls for {DOC.relative_to(ROOT)} and {MID.relative_to(ROOT)}")

    # --- defaults ---------------------------------------------------------
    w = descend_widths()
    narrow = shipped_narrow()
    # ADOPTED 2026-09-09T03:00Z.  Both of this probe's knobs moved: opening
    # GRIPPER_NARROW (0.45) -> 0.60 and plan_at_open False -> True, as two of
    # the four constants in measure_cutlery_square's best cell.  The pair below
    # is the pair every published figure is NOW measured at; the pre-adoption
    # pair is asserted separately under mismatch_is_real, so the finding this
    # file exists for is kept rather than deleted along with the defect.
    ok = (set(w) == set(DESC)
          and all(abs(o - 0.60) < 1e-9 for o, _ in w.values())
          and all(p is None for _, p in w.values()))
    check("defaults", ok,
          f"committed defaults emit {w}: opening is 0.60 and plan_at is ABSENT "
          "on both -- CUTLERY_PLAN_AT_OPEN=True solves the descent at the same "
          "opening it executes at, which is the pair every published figure "
          "is now measured at")
    w_pa = descend_widths(plan_knob=False)
    w_op = descend_widths(opening_knob=0.15)
    check("defaults/reject",
          not (all(p is None for _, p in w_pa.values())
               and all(abs(o - 0.60) < 1e-9 for o, _ in w_op.values())),
          "the same reader with either knob moved back does NOT see the "
          "committed pair, so a default that had silently drifted would be "
          "caught")

    # --- knob_live --------------------------------------------------------
    moved_op = all(abs(o - 0.15) < 1e-9 for o, _ in w_op.values())
    # w_pa is now built with plan_knob=False, i.e. the knob moved AWAY from the
    # adopted default, so liveness is plan_at coming BACK rather than dropping.
    moved_pa = all(p is not None for _, p in w_pa.values())
    check("knob_live", moved_op and moved_pa,
          f"CUTLERY_DESCEND_OPENING=0.15 moves the emitted opening to "
          f"{sorted({o for o, _ in w_op.values()})} and CUTLERY_PLAN_AT_OPEN=False "
          "puts plan_at back on both moves -- neither knob is inert")
    check("knob_live/reject", not w_op == w,
          "a reader that ignored the opening knob would return the committed "
          "widths unchanged and is caught")

    # --- reader_reads_the_move -------------------------------------------
    # Build the script at 0.15, then move the constant to 0.99 before reading.
    # Only a reader that reads the EMITTED move can still say 0.15.
    w_split = descend_widths(opening_knob=0.15, rebind_after=0.99)
    const = LAST_CONSTANT_AT_READ
    armed = const is not None and abs(const - 0.99) < 1e-9
    reads_move = all(abs(o - 0.15) < 1e-9 for o, _ in w_split.values())
    check("reader_reads_the_move", armed and reads_move,
          f"the constant read {const} at read time -- the rebind landed, so the "
          f"test is armed -- and the reader returned "
          f"{sorted({o for o, _ in w_split.values()})}, the value the script "
          "was BUILT at: it is reading the emitted move, not the constant")
    check("reader_reads_the_move/reject",
          not (armed and all(abs(o - const) < 1e-9 for o, _ in w_split.values())),
          f"a reader that had fallen back to the constant would return {const} "
          "here and is caught; and if the rebind were removed the two values "
          "would coincide and this control would report unarmed")

    # --- mismatch_is_real -------------------------------------------------
    # The mismatch this probe was written to expose is REAL and is now CLOSED.
    # Both halves are asserted, because a fix that deleted the measurement of
    # the defect along with the defect would leave nothing to show the knob was
    # ever worth moving.
    w_pre = descend_widths(opening_knob=narrow, plan_knob=False)
    gaps_pre = {k: (o, p) for k, (o, p) in w_pre.items()
                if p is not None and abs(o - p) > 1e-9}
    check("mismatch_is_real", len(gaps_pre) == 2,
          f"at the PRE-ADOPTION pair (opening=GRIPPER_NARROW {narrow}, "
          f"plan_at_open=False) both descend moves still carry two DIFFERENT "
          f"widths {gaps_pre} -- the pose solved at one and executed at the "
          "other. The finding stands; it is reproduced here, not recited.")
    check("mismatch_is_real/reject",
          not len({k for k, (o, p) in w_pre.items()
                   if p is not None and abs(o - p) > 1.0}) == 2,
          "a reader whose tolerance (1.0 rad) is too wide to see this gap "
          "reports no mismatch and is caught")
    gaps_now = {k: (o, p) for k, (o, p) in w.items()
                if p is not None and abs(o - p) > 1e-9}
    check("mismatch_is_closed_by_the_adopted_defaults", not gaps_now,
          f"at the committed defaults there is no solved/executed gap left "
          f"({gaps_now or 'none'}): plan_at is dropped, so the jaws that "
          "descend are the jaws the solver placed")

    # --- identity ---------------------------------------------------------
    shipped_cells = [c for c in d["cells"] if c["shipped"]]
    s = cell(d, None, False)
    ok = (len(shipped_cells) == 1 and shipped_cells[0] is s
          and all(c["n"] == d["seeds"] for c in d["cells"])
          and not any("error" in r for r in d["sweep_runs"])
          and not any("error" in r for r in d["offset_runs"]))
    check("identity", ok,
          f"exactly one of {len(d['cells'])} cells is flagged shipped, it is "
          f"(opening=shipped, plan_at_open=False), all cells ran "
          f"{d['seeds']}/{d['seeds']} seeds and no run errored")
    forged = copy.deepcopy(d)
    forged["cells"][1]["shipped"] = True
    check("identity/reject",
          not len([c for c in forged["cells"] if c["shipped"]]) == 1,
          "a copy with the shipped flag on two cells is caught")

    # --- reproduces -------------------------------------------------------
    ok = (s["fork_placed"] == ish["fork_placed"] == 0
          and s["spoon_placed"] == ish["spoon_placed"] == 0
          and s["subgoals_met_total"] == ish["subgoals_met_total"] == 15
          and abs(s["fork_stall_mm_median"] - ish["fork_stall_mm_median"]) <= 2.0
          and abs(s["spoon_stall_mm_median"] - ish["spoon_stall_mm_median"]) <= 2.0)
    check("reproduces", ok,
          f"shipped cell fork {s['fork_placed']}/{s['n']}, spoon "
          f"{s['spoon_placed']}/{s['n']}, {s['subgoals_met_total']}/50 sub-goals, "
          f"stalls {s['fork_stall_mm_median']}/{s['spoon_stall_mm_median']} mm "
          f"against cutlery_approach.json's independent "
          f"{ish['fork_stall_mm_median']}/{ish['spoon_stall_mm_median']} mm")
    fs = copy.deepcopy(s)
    fs["fork_stall_mm_median"] = s["fork_stall_mm_median"] + 9.0
    check("reproduces/reject",
          not abs(fs["fork_stall_mm_median"] - ish["fork_stall_mm_median"]) <= 2.0,
          "a 9 mm shifted copy of the shipped cell no longer agrees with the "
          "independently published stall and is caught")

    # --- offset_is_half ---------------------------------------------------
    def half(o):
        return abs(o["executed_sep_mm_median"] - o["solved_sep_mm_median"]) / 2.0

    desc = {k: d["offset_by_waypoint"][k] for k in DESC}
    above = {k: d["offset_by_waypoint"][k] for k in ABOVE}
    resids = {k: round(abs(o["offset_mm_median"] - half(o)), 3)
              for k, o in desc.items()}
    ok = all(v < 0.5 for v in resids.values())
    check("offset_is_half", ok,
          "the displacement equals half the change in jaw separation -- "
          + ", ".join(f"{k}: {desc[k]['offset_mm_median']} mm measured against "
                      f"{round(half(desc[k]),2)} mm predicted from "
                      f"{desc[k]['solved_sep_mm_median']} -> "
                      f"{desc[k]['executed_sep_mm_median']} mm, residual {v}"
                      for k, v in resids.items())
          + " -- which is what a gripper with one fixed jaw must do")
    fo = copy.deepcopy(desc)
    for o in fo.values():
        o["offset_mm_median"] *= 2
    check("offset_is_half/reject",
          not all(abs(o["offset_mm_median"] - half(o)) < 0.5 for o in fo.values()),
          "a copy with the displacement doubled no longer matches the width "
          "change it is derived from and is caught")

    # --- offset_control ---------------------------------------------------
    ok = (all(not o["has_plan_at"] for o in above.values())
          and all(o["offset_mm_max"] < 0.01 for o in above.values())
          and all(o["has_plan_at"] for o in desc.values()))
    check("offset_control", ok,
          "the approach waypoints carry no plan_at and their offset is "
          + ", ".join(f"{k} max {o['offset_mm_max']} mm" for k, o in above.items())
          + " -- zero, so the descend number is the width mismatch and not "
            "this probe's algebra")
    fa = copy.deepcopy(above)
    for o in fa.values():
        o["offset_mm_max"] = 31.0
    check("offset_control/reject",
          not all(o["offset_mm_max"] < 0.01 for o in fa.values()),
          "a copy in which the no-plan_at control ALSO shows a 31 mm offset -- "
          "which would mean the arithmetic produced the number -- is caught")

    # --- offset_sideways --------------------------------------------------
    mid_narrow = mid["at_gripper_narrow"]
    ok = (all(abs(o["along_jaw_mm_median"]) > 20 * abs(o["along_approach_mm_median"])
              for o in desc.values())
          and abs(mid_narrow["along_jaw_mm"]) > 20 * abs(mid_narrow["along_approach_mm"])
          and all(abs(o["along_jaw_mm_median"] - mid_narrow["along_jaw_mm"]) < 2.0
                  for o in desc.values()))
    check("offset_sideways", ok,
          "the displacement is sideways along the jaw axis, not fore-aft along "
          "the wrist axis: "
          + ", ".join(f"{k} jaw {o['along_jaw_mm_median']} mm vs approach "
                      f"{o['along_approach_mm_median']} mm" for k, o in desc.items())
          + f"; the independent full-range sweep reads jaw "
            f"{mid_narrow['along_jaw_mm']} mm vs approach "
            f"{mid_narrow['along_approach_mm']} mm at the same opening")
    fsw = copy.deepcopy(desc)
    for o in fsw.values():
        o["along_jaw_mm_median"], o["along_approach_mm_median"] = (
            o["along_approach_mm_median"], o["along_jaw_mm_median"])
    check("offset_sideways/reject",
          not all(abs(o["along_jaw_mm_median"])
                  > 20 * abs(o["along_approach_mm_median"]) for o in fsw.values()),
          "a copy with the two components swapped -- the world the shipped "
          "docstring describes -- is caught")

    # --- contact_named ----------------------------------------------------
    firsts = [r["bodies"][b]["first_contact"]
              for r in d["sweep_runs"] if "bodies" in r for b in ("fork", "spoon")]
    got = [f for f in firsts if f]

    def two_sided(pair: str) -> bool:
        parts = pair.split(" | ")
        return len(parts) == 2 and all(p.strip() for p in parts)

    ok = bool(got) and all(two_sided(f["pair"]) for f in got)
    check("contact_named", ok,
          f"{len(got)} of {len(firsts)} descend holds recorded a first contact "
          "and every one names both sides, hand geom | world geom")
    # The corruption is derived from a real entry, not written as a literal.
    one_sided = got[0]["pair"].split(" | ")[1]
    check("contact_named/reject", not two_sided(one_sided),
          f"the same predicate applied to {one_sided!r} -- a real entry with "
          "its hand side dropped, which is the shape every earlier probe in "
          "this tree recorded -- is caught")

    # --- verdict ----------------------------------------------------------
    want = verdict_of(d)
    check("verdict", d.get("verdict") == want,
          f"the stored verdict is {d.get('verdict')!r} and the cell table -- "
          f"cutlery placements {sorted(c['fork_placed'] + c['spoon_placed'] for c in d['cells'])} "
          f"-- forces {want!r}")
    fv = copy.deepcopy(d)
    fv["cells"][2]["fork_placed"] = 1
    check("verdict/reject", verdict_of(fv) != want,
          f"a copy with one fork placement planted forces {verdict_of(fv)!r} "
          "instead, so the verdict is not free of its own table")

    # --- placement_lift ---------------------------------------------------
    def backed(cells):
        return all(not (c["fork_placed"] > 0 and c["fork_lifted_mm_median"] <= 0.0)
                   and not (c["spoon_placed"] > 0 and c["spoon_lifted_mm_median"] <= 0.0)
                   for c in cells)

    check("placement_lift", backed(d["cells"]),
          "no cell reports a placement it has no lift behind; lifts are "
          + ", ".join(f"{('shipped' if c['opening'] is None else c['opening'])}/"
                      f"{c['plan_at_open']}:{c['fork_lifted_mm_median']}"
                      for c in d["cells"]))
    fl = copy.deepcopy(d)
    fl["cells"][0]["fork_placed"] = 2
    fl["cells"][0]["fork_lifted_mm_median"] = 0.0
    check("placement_lift/reject", not backed(fl["cells"]),
          "a planted cell that places the fork twice without ever lifting it "
          "is caught")

    # ``placement_lift``'s accept side is vacuous while nothing is placed, so
    # the substantive half is stated separately: nothing is LIFTED either, in
    # any cell, which is what stops "the stall halved" being narrated as a
    # near miss.
    lifts = [max(c["fork_lifted_mm_median"], c["spoon_lifted_mm_median"])
             for c in d["cells"]]
    asked = commanded_lift_mm()
    bar = 0.1 * asked
    worst = max(d["cells"], key=lambda c: max(c["fork_lifted_mm_median"],
                                              c["spoon_lifted_mm_median"]))
    check("not_near_miss", max(lifts) < bar,
          f"the largest median lift over all {len(lifts)} cells is "
          f"{max(lifts)} mm -- at "
          f"{'shipped' if worst['opening'] is None else worst['opening']}/"
          f"{worst['plan_at_open']} -- against the {asked} mm the emitted "
          f"fork_lift move ASKS for, so under a tenth of the commanded lift "
          "(bar {:.1f} mm, taken from the controller not from this data); "
          "neither piece is carried anywhere".format(bar))
    check("not_near_miss/reject", not max(lifts + [0.5 * asked]) < bar,
          f"a median lift of half the commanded {asked} mm would be a "
          "different finding and is caught")

    # --- gain_not_cutlery -------------------------------------------------
    best = max(d["cells"], key=lambda c: c["subgoals_met_total"])
    cutlery_zero = all(c["fork_placed"] == 0 and c["spoon_placed"] == 0
                       for c in d["cells"])
    ok = cutlery_zero and best["subgoals_met_total"] >= s["subgoals_met_total"]
    check("gain_not_cutlery", ok,
          f"best cell scores {best['subgoals_met_total']}/50 against the "
          f"shipped {s['subgoals_met_total']}/50, and every cell places "
          "0 forks and 0 spoons -- so whatever moved, it is not the thing "
          "this hypothesis is about, and nothing is adopted on it")
    fg = copy.deepcopy(d)
    fg["cells"][2]["fork_placed"] = 3
    check("gain_not_cutlery/reject",
          not all(c["fork_placed"] == 0 and c["spoon_placed"] == 0
                  for c in fg["cells"]),
          "a copy where a cell really does place three forks would be a "
          "different finding entirely and is caught")

    # --- probe_controls ---------------------------------------------------
    check("probe_controls_envelope", all(d["controls"].values()),
          f"{len(d['controls'])} controls in the sweep probe: "
          + ", ".join(f"{k}={v}" for k, v in d["controls"].items()))
    check("probe_controls_midpoint", all(mid["controls"].values()),
          f"{len(mid['controls'])} controls in the midpoint probe: "
          + ", ".join(f"{k}={v}" for k, v in mid["controls"].items()))

    # --- no tautological checks -------------------------------------------
    # Pinning the COUNT catches a control that vanishes.  It does not catch one
    # neutered in place -- `check("x", ok, ...)` rewritten to `check("x", True,
    # ...)` keeps the count and goes quiet, which is how a mutant survived a
    # campaign on 2026-09-08.  This reads this file's own syntax tree, not its
    # text, so the comment you are reading cannot satisfy it.
    taut = tautological_checks(pathlib.Path(__file__).read_text())
    check("no_tautological_checks", not taut,
          f"none of the {len(PASSED) + len(FAILED)} check() calls in this file "
          "passes a literal as its verdict"
          if not taut else f"literal verdicts at: {taut}")
    planted = tautological_checks(
        'check("planted", True, "x")\ncheck("planted2", not False, "y")\n')
    check("no_tautological_checks/reject", len(planted) == 2,
          f"the same reader finds {planted} in a planted source, so it is "
          "looking at verdicts rather than at nothing")

    # --- pinned counts ----------------------------------------------------
    n_checks = len(PASSED) + len(FAILED)
    n_rejects = sum(1 for n in PASSED + FAILED if n.endswith("/reject"))
    check("control_count_pinned", n_checks == EXPECT_CHECKS,
          f"{n_checks} checks ran against a pinned {EXPECT_CHECKS} -- a control "
          "that vanishes fails here rather than going quiet")
    check("reject_count_pinned", n_rejects == EXPECT_REJECTS,
          f"{n_rejects} of them are negative controls against a pinned "
          f"{EXPECT_REJECTS}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
