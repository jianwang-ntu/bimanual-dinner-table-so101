#!/usr/bin/env python3
"""Controls for the fork-descent sweep, and for the fact it changed nothing.

``scripts/measure_fork_descent.py`` tested the intervention
``measure_grasp_feasibility.py`` named as next: a straddle-preserving descent.
It is REFUTED for placement -- ``fork_placed`` is 0/10 in all sixteen variants
-- and this suite exists to make that refutation, and the two mechanisms it
did separate, checkable rather than readable.

The control that matters most is ``defaults``: the sweep needed two new knobs
on ``_pick``, and committing a knob whose default silently moved the shipped
behaviour would invalidate every published figure in the repository.  So the
suite drives ``dinner_table_script()`` itself and asserts the move list it
emits at the committed defaults, with the negative control being the knob
turned on.

  defaults    ACCEPT the committed defaults emit exactly one, unsquared
              ``fork_descend`` move -- the shipped behaviour; REJECT the same
              reader with CUTLERY_DESCEND_STEPS=2 or _SQUARE=True
  identity    ACCEPT the sweep's baseline IS the shipped setting and ran the
              full seed count with no errors; REJECT a baseline relabelled
  refutation  ACCEPT no variant places the fork, and none places the spoon;
              REJECT a copy in which one variant placed it
  not_inert   ACCEPT the intervention moved the physics it claims to move --
              squaring the descent collapses the shove and improves arrival;
              REJECT a copy where the squared and unsquared shoves agree
  bow         ACCEPT more solved poses made the path BOW MORE, not less, which
              is what refutes the joint-space-chord story; REJECT the two
              medians swapped
  blocker     ACCEPT the blocking geom changes identity under squaring --
              ``fork_handle`` for the shipped descent, ``drawer_back`` for the
              squared one; REJECT a forged contact tally
  no_claim    ACCEPT the +4 sub-goals at the 6 mm end height are NOT claimed:
              the verdict is NO PLACEMENT GAINED, every one of those sub-goals
              is an object the intervention does not touch, and the paired
              sign test on them is not significant; REJECT a copy in which the
              gain reaches the fork

Standard library only for everything except ``defaults``, which must import
the controller because the claim is about what the controller emits.
Run:  python3 scripts/test_fork_descent.py
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
SWEEP = EV / "fork_descent.json"

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


# --------------------------------------------------------------- the reader
_DOC: dict = {}


def variant(steps, z, sq, doc=None):
    for v in (doc or _DOC)["variants"]:
        if v["steps"] == steps and v["descend_z_m"] == z and v["square"] == sq:
            return v
    raise KeyError((steps, z, sq))


def fork_descend_moves(steps=None, square=None):
    """The ``fork_*_descend*`` moves ``dinner_table_script()`` emits, as
    (label, squared) pairs.  Driven through the real script rather than by
    reading the source, because the claim is about behaviour."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    from envs import controller as C
    keep = (C.CUTLERY_DESCEND_STEPS, C.CUTLERY_DESCEND_SQUARE)
    try:
        if steps is not None:
            C.CUTLERY_DESCEND_STEPS = steps
        if square is not None:
            C.CUTLERY_DESCEND_SQUARE = square
        script = C.dinner_table_script()
    finally:
        C.CUTLERY_DESCEND_STEPS, C.CUTLERY_DESCEND_SQUARE = keep
    out = []
    for e in script:
        if isinstance(e, tuple) and e and e[0] == "if":
            continue
        for mv in e[0].values():
            lb = getattr(mv, "label", "")
            if lb.startswith("fork_"):
                out.append((lb, bool(mv.square)))
    return out


def descents(moves):
    return [m for m in moves if m[0].startswith("fork_descend")]


def sign_test_p(gains: int, losses: int) -> float:
    """Two-sided exact binomial on the discordant pairs (McNemar)."""
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def top_contact(d, steps, z, sq, body="fork"):
    tally: dict[str, int] = {}
    for r in d["runs"]:
        if (r["steps"], r["descend_z_m"], r["square"]) != (steps, z, sq):
            continue
        for c in r["objects"][body]["blocking_contacts"]:
            tally[c["geom"]] = tally.get(c["geom"], 0) + c["steps"]
    return (max(tally, key=tally.get) if tally else None), tally


def main() -> int:
    ok = True
    d = json.loads(SWEEP.read_text(encoding="utf-8"))
    _DOC.update(d)
    base = variant(1, 0.003, False)

    print("defaults")
    got = descents(fork_descend_moves())
    ok &= check("committed_defaults_emit_the_shipped_descent",
                got == [("fork_descend", False)],
                f"dinner_table_script() at the committed defaults emits {got} "
                "-- one solved pose, unsquared, exactly as every published "
                "figure was measured")
    stepped = descents(fork_descend_moves(steps=2))
    ok &= check("reject_a_default_that_had_stepped_the_descent",
                stepped != [("fork_descend", False)],
                f"with CUTLERY_DESCEND_STEPS=2 the same reader sees {stepped}, "
                "so the control is reading the script, not a constant")
    squared = descents(fork_descend_moves(square=True))
    ok &= check("reject_a_default_that_had_squared_the_descent",
                squared != [("fork_descend", False)],
                f"with CUTLERY_DESCEND_SQUARE=True the same reader sees "
                f"{squared}")
    # Over EVERY fork_* move, not just the descents: a reader that filtered
    # to descents first could only ever report descents, and the control
    # would have been unable to fail.
    on = [lb for lb, sq in fork_descend_moves(steps=4, square=True) if sq]
    off = [lb for lb, sq in fork_descend_moves(steps=4, square=True) if not sq]
    ok &= check("squaring_the_descent_never_squares_the_carry",
                on and all(lb.startswith("fork_descend") for lb in on)
                and "fork_lift" in off and "fork_above" in off,
                "with the knob full on, the squared fork moves are "
                f"{on} and the unsquared ones include fork_above and "
                "fork_lift -- _handoff already measured that squaring a "
                "loaded carry asks for a wrist the arm cannot hold")
    off_default = [lb for lb, sq in fork_descend_moves() if not sq]
    ok &= check("reject_a_reader_that_cannot_see_the_carry",
                "fork_lift" in off_default and "fork_above" in off_default
                and len(off_default) > 2,
                f"the same reader sees all {len(off_default)} unsquared fork "
                "moves at the committed defaults, so the control above is "
                "looking at the carry rather than at a filtered-out label")

    print("\nidentity")
    ok &= check("the_baseline_is_the_shipped_setting",
                base["shipped"] and d["shipped_variant"] == {
                    "steps": 1, "descend_z_m": 0.003, "square": False},
                f"baseline = {base['steps']} solved pose, "
                f"{base['descend_z_m']*1000:.0f} mm end height, unsquared")
    bent = json.loads(json.dumps(d))
    for v in bent["variants"]:
        v["shipped"] = (v["steps"] == 2 and v["descend_z_m"] == 0.003
                        and not v["square"])
    ok &= check("reject_a_sweep_whose_baseline_was_relabelled",
                not next(v for v in bent["variants"] if v["shipped"])["steps"] == 1,
                "moving the shipped flag to the 2-pose variant is caught")
    ok &= check("every_variant_ran_every_seed_with_no_errors",
                len(d["variants"]) == 16
                and all(v["seeds"] == d["seeds"] == 10 and not v["errors"]
                        for v in d["variants"]),
                f"{len(d['variants'])} variants x {d['seeds']} seeds, "
                "0 errors")

    print("\nrefutation")
    ok &= check("no_variant_places_the_fork",
                all(v["fork_placed"] == 0 for v in d["variants"])
                and d["verdict"] == "NO PLACEMENT GAINED",
                "fork_placed is 0/10 in all 16 variants; verdict "
                f"{d['verdict']!r}")
    lifted = json.loads(json.dumps(d))
    variant(1, 0.006, True)  # exists
    for v in lifted["variants"]:
        if v["steps"] == 4 and v["square"]:
            v["fork_placed"] = 1
    ok &= check("reject_a_sweep_in_which_one_variant_placed_it",
                not all(v["fork_placed"] == 0 for v in lifted["variants"]),
                "a single placement anywhere in the sweep is caught, so "
                "'0 everywhere' is a reading and not a constant")
    ok &= check("the_spoon_negative_control_holds",
                all(v["spoon_placed"] == 0 for v in d["variants"]),
                "measure_grasp_feasibility.py found no collision-free spoon "
                "grasp pose exists (jaws 10.4-48.4 mm ABOVE the handle top at "
                "first clearance); a trajectory fix must not place it, and "
                "none of the 16 variants does")

    print("\nnot_inert")
    pairs = [(k, z) for k in (1, 2, 4, 8) for z in (0.003, 0.006)]
    shoves = [(variant(k, z, False)["fork_shove_mm_median"],
               variant(k, z, True)["fork_shove_mm_median"]) for k, z in pairs]
    ok &= check("squaring_the_descent_collapses_the_shove",
                all(sq < 2.0 < un or (sq < un and un - sq > 2.0)
                    for un, sq in shoves)
                and max(un for un, _ in shoves) > 40.0,
                "median planar shove of the fork during its own descent, "
                "unsquared -> squared: "
                + ", ".join(f"{u}->{s}" for u, s in shoves)
                + " mm -- the intervention is not inert")
    flat = json.loads(json.dumps(d))
    for v in flat["variants"]:
        if v["square"]:
            v["fork_shove_mm_median"] = variant(
                v["steps"], v["descend_z_m"], False)["fork_shove_mm_median"]
    def shoves_of(doc):
        return [(next(v for v in doc["variants"] if (v["steps"], v["descend_z_m"], v["square"]) == (k, z, False))["fork_shove_mm_median"],
                 next(v for v in doc["variants"] if (v["steps"], v["descend_z_m"], v["square"]) == (k, z, True))["fork_shove_mm_median"])
                for k, z in pairs]
    ok &= check("reject_a_copy_where_squaring_changed_nothing",
                not all(u - s > 2.0 for u, s in shoves_of(flat)),
                "with the squared shoves overwritten by the unsquared ones the "
                "same test fails, so it is measuring a difference")
    b_arr = variant(1, 0.006, True)["fork_stalled_above_target_mm_median"]
    ok &= check("squaring_improves_arrival_and_still_buys_nothing",
                b_arr < base["fork_stalled_above_target_mm_median"]
                and variant(1, 0.006, True)["fork_placed"] == 0,
                f"best arrival {b_arr} mm against the shipped "
                f"{base['fork_stalled_above_target_mm_median']} mm "
                "(and against measure_cutlery_block.py's recorded 43.0 mm "
                "median) -- and the fork is still placed 0/10")

    print("\nbow")
    b1 = variant(1, 0.003, False)["fork_bow_mm_median"]
    b8 = variant(8, 0.003, False)["fork_bow_mm_median"]
    ok &= check("more_solved_poses_bows_the_path_more_not_less",
                b8 > b1,
                f"median lateral bow of the jaw meeting point off the straight "
                f"Cartesian line: {b1} mm at one solved pose, {b8} mm at eight "
                "-- the joint-space-chord story predicted the opposite, which "
                "is what refutes it")
    swapped = json.loads(json.dumps(d))
    for v in swapped["variants"]:
        if not v["square"] and v["descend_z_m"] == 0.003 and v["steps"] in (1, 8):
            v["fork_bow_mm_median"] = b1 if v["steps"] == 8 else b8
    def bow_of(doc, k):
        return next(v for v in doc["variants"]
                    if (v["steps"], v["descend_z_m"], v["square"])
                    == (k, 0.003, False))["fork_bow_mm_median"]
    ok &= check("reject_the_two_bow_medians_swapped",
                not bow_of(swapped, 8) > bow_of(swapped, 1),
                "with the medians exchanged the same test fails")

    print("\nblocker")
    g_ship, t_ship = top_contact(d, 1, 0.003, False)
    g_sq, t_sq = top_contact(d, 1, 0.006, True)
    ok &= check("the_blocking_geom_changes_identity_under_squaring",
                g_ship == "fork_handle" and g_sq == "drawer_back",
                f"shipped descent is blocked by {g_ship!r} "
                f"({t_ship.get('fork_handle')} steps of contact, the object it "
                f"is trying to grasp); the squared descent by {g_sq!r} "
                f"({t_sq.get('drawer_back')} steps) -- two different walls, so "
                "squaring trades the object for the back of the drawer")
    forged = json.loads(json.dumps(d))
    for r in forged["runs"]:
        if (r["steps"], r["descend_z_m"], r["square"]) == (1, 0.006, True):
            r["objects"]["fork"]["blocking_contacts"] = [
                {"geom": "fork_handle", "steps": 99999}]
    ok &= check("reject_a_forged_contact_tally",
                top_contact(forged, 1, 0.006, True)[0] == "fork_handle",
                "rewriting the squared variant's contacts to name the fork "
                "flips the reader, so it is reading the tally")

    print("\nno_claim")
    alt = variant(1, 0.006, False)
    delta = alt["subgoals_total"] - base["subgoals_total"]
    per = {g: [0, 0] for g in ("drawer_open", "fork_placed", "spoon_placed",
                               "plate_placed", "mug_placed")}
    for r in d["runs"]:
        key = (r["steps"], r["descend_z_m"], r["square"])
        if key == (1, 0.003, False):
            i = 0
        elif key == (1, 0.006, False):
            i = 1
        else:
            continue
        for g in per:
            per[g][i] += int(r["subgoals"][g])
    touched = per["fork_placed"][1] - per["fork_placed"][0] + \
        per["spoon_placed"][1] - per["spoon_placed"][0]
    ok &= check("the_subgoal_gain_touches_nothing_the_intervention_touches",
                delta == 4 and touched == 0,
                f"the 6 mm end height scores {alt['subgoals_total']}/50 against "
                f"the shipped {base['subgoals_total']}/50, and all {delta} of "
                "the gain is plate and mug -- "
                + ", ".join(f"{g} {a}->{b}" for g, (a, b) in per.items())
                + f"; the fork and spoon contribution to the delta is {touched}")
    gains = losses = 0
    by_seed = {}
    for r in d["runs"]:
        key = (r["steps"], r["descend_z_m"], r["square"])
        if key in ((1, 0.003, False), (1, 0.006, False)):
            by_seed.setdefault(r["seed"], {})[key[1]] = r["subgoals"]
    for s, two in by_seed.items():
        for g in ("plate_placed", "mug_placed"):
            a, b = int(two[0.003][g]), int(two[0.006][g])
            gains += int(b > a)
            losses += int(a > b)
    p = sign_test_p(gains, losses)
    ok &= check("and_it_is_not_significant_so_it_is_not_shipped",
                p > 0.05 and d["verdict"] == "NO PLACEMENT GAINED",
                f"paired sign test over the 20 plate/mug seed-slots: {gains} "
                f"gained, {losses} lost, p={p:.3f} -- indistinguishable from "
                "chaos sensitivity to a 3 mm change in a waypoint the plate "
                "and the mug never touch, so the committed default stays at "
                "3 mm and no published figure moves")
    reach = json.loads(json.dumps(d))
    for r in reach["runs"]:
        if (r["steps"], r["descend_z_m"], r["square"]) == (1, 0.006, False):
            r["subgoals"]["fork_placed"] = True
    t2 = sum(int(r["subgoals"]["fork_placed"]) for r in reach["runs"]
             if (r["steps"], r["descend_z_m"], r["square"]) == (1, 0.006, False)) \
        - sum(int(r["subgoals"]["fork_placed"]) for r in reach["runs"]
              if (r["steps"], r["descend_z_m"], r["square"]) == (1, 0.003, False))
    ok &= check("reject_a_copy_in_which_the_gain_reached_the_fork",
                t2 != 0,
                f"if the 6 mm variant had placed the fork the same reader sees "
                f"a fork contribution of {t2}, so 'the gain touches nothing we "
                "touched' is a measurement")

    EV.mkdir(parents=True, exist_ok=True)
    (EV / "fork_descent_controls.json").write_text(
        json.dumps({"all_pass": bool(ok),
                    "sweep": str(SWEEP.relative_to(ROOT)),
                    "variants": len(d["variants"]),
                    "seeds": d["seeds"],
                    "verdict": d["verdict"],
                    "controls": results}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
