#!/usr/bin/env python3
"""Controls for the cutlery placement randomizer and the seat sweep.

Two things were added on 2026-09-08 and this file is what stops either of them
being believed on its own say-so:

  envs/randomize.py::randomize_cutlery   closes the half of T4's first named
      axis that was missing -- the fork and the spoon had the same x, y and yaw
      on every seed.  It ships OFF (`CUTLERY_PLACE_SPREAD = 0.0`) and
      evidence/cutlery_placement.json is the measurement of what switching it
      on costs, so the switch is a decision with a price on it.
  scripts/measure_cutlery_seat.py        tests route (c) of F-CUTLERY-LIP-001,
      the SCENE route, which TECHNICAL_SUMMARY section 8 named as one of the
      two things never tried.  Its result is negative and this file's job is to
      make sure the negative is a real one and not a sweep that never moved
      anything.

Every accept is paired with a reject built by bending the artifact it reads,
because a check that can only pass proves nothing.  Two of the rejects exist
for a specific failure this tree has hit before: a sweep whose "variants" are
all the same run reads exactly like a mechanism that does not matter.

The control count is PINNED.  A control that quietly disappears is itself a
failure, so a short run fails as loudly as a red one.

Standard library only.  Run:  python3 scripts/test_cutlery_placement.py
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
RANDOMIZER = ROOT / "envs" / "randomize.py"
PLACEMENT = EV / "cutlery_placement.json"
SEAT = EV / "cutlery_seat.json"
TRACKING = EV / "descent_tracking.json"

# Pinned: 35 controls, of which 12 are named `reject_` and are built by
# bending the artifact their partner accepts.
EXPECTED_CONTROLS = 35
EXPECTED_REJECTS = 12

# The per-sub-goal tally every artifact in the repository quotes, and the
# configuration it was measured at: cutlery seated where dinner_table.build
# authors it, no jitter, no splay, unsquared descent.
PUBLISHED = {"drawer_open": 10, "fork_placed": 0, "spoon_placed": 0,
             "plate_placed": 4, "mug_placed": 1}

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def bend(obj):
    return json.loads(json.dumps(obj))


# --------------------------------------------------------------- the code
def constants(source: str) -> dict:
    """Read the switches out of the randomizer with ast, not by importing it,
    so this file runs on a clone with no mujoco and cannot be fooled by a
    monkey-patch at runtime."""
    got = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.startswith("CUTLERY"):
                    got[t.id] = ast.literal_eval(node.value)
    return got


def has_rejection_pass(source: str) -> bool:
    """The draw must be rejected by MuJoCo's own contact pass.

    Read as a SHAPE, not as a substring.  A text scan for "_penetrates" passes
    on a function whose guard has been replaced by `if True:` as long as the
    name survives anywhere else in the body -- measured, 2026-09-08, when the
    `no_penetration_test` mutant walked through exactly that hole.  What has to
    be true is that the accept/break is guarded by a comparison that CALLS the
    penetration test against the tolerance."""
    fn = next((n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.FunctionDef)
               and n.name == "randomize_cutlery"), None)
    if fn is None:
        return False
    calls_forward = any(isinstance(n, ast.Attribute) and n.attr == "mj_forward"
                        for n in ast.walk(fn))
    # (a) the penetration of the drawn pose is actually computed, inside the
    #     draw loop rather than once outside it, and
    # (b) the accept -- the `break` that stops redrawing -- is guarded by a
    #     comparison against the tolerance.  `if True:` satisfies neither.
    # The DRAW loop is the innermost `for` that contains the `break`.  Taking
    # any enclosing loop would also see the fallback branch's call and pass on
    # a body whose draw loop no longer measures anything.
    breaking = [n for n in ast.walk(fn) if isinstance(n, ast.For)
                and any(isinstance(c, ast.Break) for c in ast.walk(n))]
    if not breaking:
        return False
    draw = min(breaking, key=lambda lp: len(list(ast.walk(lp))))
    calls_pen = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "_penetrates" for c in ast.walk(draw))
    guarded = any(
        isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and any(isinstance(c, ast.Name) and c.id == "PENETRATION_TOL_M"
                for c in ast.walk(node.test))
        and any(isinstance(c, ast.Break) for c in ast.walk(node))
        for node in ast.walk(draw))
    return calls_forward and calls_pen and guarded


def variant_key(v: dict) -> tuple:
    return tuple(round(float(v[k]), 6) for k in
                 ("seat_dy", "splay") if k in v) or (round(v["spread"], 6),)


def main() -> int:
    ok = True
    src = RANDOMIZER.read_text(encoding="utf-8")
    consts = constants(src)

    print("code")
    ok &= check("switches_ship_off",
                consts.get("CUTLERY_PLACE_SPREAD") == 0.0
                and tuple(consts.get("CUTLERY_SEAT_XY", ())) == (0.0, 0.0)
                and consts.get("CUTLERY_SPLAY_X") == 0.0,
                f"{ {k: v for k, v in consts.items() if k != 'CUTLERY'} } -- so "
                "the shipped scene is the one every figure was measured in")
    flipped = re.sub(r"^CUTLERY_SPLAY_X = 0\.0$", "CUTLERY_SPLAY_X = 0.02",
                     src, count=1, flags=re.M)
    ok &= check("reject_a_switch_left_on",
                constants(flipped).get("CUTLERY_SPLAY_X") == 0.02,
                "a 20 mm splay left in the source is read back, so the check "
                "above is reading the file and not asserting a literal")
    ok &= check("draws_are_rejected_by_the_contact_pass",
                has_rejection_pass(src),
                "randomize_cutlery calls mj_forward and tests _penetrates "
                "against PENETRATION_TOL_M")
    ok &= check("reject_a_randomizer_that_never_checks",
                not has_rejection_pass(src.replace(
                    "if pen >= PENETRATION_TOL_M:", "if True:", 1))
                and not has_rejection_pass(src.replace(
                    "_penetrates(model, data, bid)\n"
                    "            if pen", "0.0\n            if pen", 1)),
                "the check fails BOTH when the guard is replaced by `if True:` "
                "and when the penetration is never computed -- neither edit "
                "removes the identifier from the file, which is why this reads "
                "the syntax tree and not the text")

    # ------------------------------------------------------------ placement
    print("\nplacement randomizer")
    pl = json.loads(PLACEMENT.read_text(encoding="utf-8"))
    c = pl["controls"]
    base = next(v for v in pl["variants"] if v["shipped_before"])
    moved = [v for v in pl["variants"] if not v["shipped_before"]]

    ok &= check("default_reproduces_the_published_tally",
                base["per_goal"] == PUBLISHED and base["subgoals_met_total"] == 15,
                f"at spread 0.0: {base['per_goal']} = "
                f"{base['subgoals_met_total']}/50, which is what every artifact "
                "in the repository quotes")
    bent = bend(pl)
    next(v for v in bent["variants"] if v["shipped_before"])["per_goal"]["plate_placed"] = 6
    bent_base = next(v for v in bent["variants"] if v["shipped_before"])
    ok &= check("reject_a_baseline_that_drifted",
                bent_base["per_goal"] != PUBLISHED,
                "moving the baseline's plate to 6/10 -- the figure T1's prose "
                "still carried -- fails the same comparison")

    ok &= check("spread_zero_really_is_zero",
                base["fork_start_spread_mm"] == {"x": 0.0, "y": 0.0}
                and base["spoon_start_spread_mm"] == {"x": 0.0, "y": 0.0},
                "fork and spoon start at one pose on all "
                f"{base['n']} seeds")
    ok &= check("spread_one_really_moves",
                all(v["fork_start_spread_mm"]["y"] > 5.0
                    and v["spoon_start_spread_mm"]["y"] > 5.0 for v in moved),
                "; ".join(f"spread {v['spread']}: fork y "
                          f"{v['fork_start_spread_mm']['y']} mm, spoon y "
                          f"{v['spoon_start_spread_mm']['y']} mm" for v in moved)
                + " -- a randomizer that rejected every draw would read as a "
                  "null result instead of a broken one")
    bentz = bend(pl)
    for v in bentz["variants"]:
        v["fork_start_spread_mm"] = {"x": 0.0, "y": 0.0}
        v["spoon_start_spread_mm"] = {"x": 0.0, "y": 0.0}
    ok &= check("reject_a_sweep_that_never_moved_anything",
                not all(v["fork_start_spread_mm"]["y"] > 5.0
                        for v in bentz["variants"] if not v["shipped_before"]),
                "flattening every variant's spread to zero fails the check above")

    # upstream_untouched, recomputed here rather than read off the probe's own
    # boolean -- the probe is the thing under test.
    ref = {r["seed"]: r["upstream"] for r in pl["runs"]
           if r.get("shipped_before") and "upstream" in r}
    mism = [r["seed"] for r in pl["runs"]
            if "upstream" in r and r["upstream"] != ref.get(r["seed"])]
    ok &= check("plate_mug_bottle_drawer_are_untouched",
                not mism and len(ref) > 0,
                f"all {len(pl['runs'])} runs across {len(pl['spreads'])} spreads "
                f"agree with the baseline seed-for-seed on every non-cutlery "
                "draw -- so a change in the score is attributable to the cutlery")
    bentu = bend(pl)
    victim = next(r for r in bentu["runs"] if not r["shipped_before"])
    victim["upstream"]["drawer_slide"] = 0.0123
    ref2 = {r["seed"]: r["upstream"] for r in bentu["runs"] if r.get("shipped_before")}
    ok &= check("reject_a_reshuffled_random_stream",
                any(r["upstream"] != ref2.get(r["seed"])
                    for r in bentu["runs"] if "upstream" in r),
                "moving one seed's drawer_slide in one arm is caught, so the "
                "check above can fail")

    cost = base["subgoals_met_total"] - moved[-1]["subgoals_met_total"]
    ok &= check("the_cost_of_switching_on_is_measured",
                moved[-1]["spread"] == 1.0 and cost == 2,
                f"full spread scores {moved[-1]['subgoals_met_total']}/50 against "
                f"{base['subgoals_met_total']}/50 -- switching the randomizer on "
                f"costs {cost} sub-goals, and that is the number the record quotes")
    ok &= check("cutlery_is_still_never_placed",
                all(v["per_goal"]["fork_placed"] == 0
                    and v["per_goal"]["spoon_placed"] == 0
                    for v in pl["variants"]),
                "fork 0/10 and spoon 0/10 at every spread -- randomizing where "
                "they start does not make them graspable")
    ok &= check("probe_controls_all_pass",
                all(c[k] for k in ("spread_0_is_fixed", "spread_1_moves",
                                   "upstream_untouched", "baseline_reproduces",
                                   "all_runs_ok")),
                {k: c[k] for k in ("spread_0_is_fixed", "spread_1_moves",
                                   "upstream_untouched", "baseline_reproduces",
                                   "all_runs_ok")})

    # ----------------------------------------------------------- seat sweep
    print("\nseat sweep")
    st = json.loads(SEAT.read_text(encoding="utf-8"))
    cells = st["cells"]
    sbase = next(x for x in cells if x["shipped"])

    ok &= check("seat_baseline_reproduces_the_published_tally",
                sbase["per_goal"] == PUBLISHED,
                f"the shipped seat scores {sbase['per_goal']} -- the sweep is "
                "measuring this scene and not another one")
    feasible = [x for x in cells if x["worst_penetration_mm"] >= -2.0]
    infeasible = [x for x in cells if x["worst_penetration_mm"] < -2.0]
    placed_any = [x for x in cells if x["per_goal"]["fork_placed"]
                  or x["per_goal"]["spoon_placed"]]

    ok &= check("every_feasible_cell_places_no_cutlery",
                all(x["per_goal"]["fork_placed"] == 0
                    and x["per_goal"]["spoon_placed"] == 0 for x in feasible),
                f"{len(feasible)} feasible cells x {sbase['n']} seeds = "
                f"{sum(x['n'] for x in feasible)} rollouts, fork 0 and spoon 0 "
                "in all of them")
    # The grid DOES contain one placement, and it is not a win: read against
    # the cell it came from rather than left in the headline.
    ok &= check("the_only_placement_in_the_grid_is_in_an_infeasible_cell",
                all(x["worst_penetration_mm"] < -2.0 for x in placed_any),
                "; ".join(
                    f"seat_dy {x['seat_dy']} splay {x['splay']} sq "
                    f"{int(x['square'])}: fork {x['per_goal']['fork_placed']} "
                    f"spoon {x['per_goal']['spoon_placed']} at "
                    f"{x['worst_penetration_mm']} mm penetration"
                    for x in placed_any) or "no cell placed anything"
                + " -- a spoon that starts inside the drawer front is not a "
                  "grasp, and this control is what stops it being reported as "
                  "one")
    bents = bend(st)
    victim_cell = next(x for x in bents["cells"]
                       if x["worst_penetration_mm"] >= -2.0)
    victim_cell["per_goal"]["fork_placed"] = 1
    ok &= check("reject_a_success_in_a_feasible_cell",
                not all(x["per_goal"]["fork_placed"] == 0
                        for x in bents["cells"]
                        if x["worst_penetration_mm"] >= -2.0),
                "one placed fork in a cell whose scene is legal fails the "
                "check above, so the negative result is a measured one")

    reaches = [x["fork_reach_mm"] for x in feasible]
    span = max(reaches) - min(reaches)
    ok &= check("the_sweep_actually_moved_the_reach",
                span >= 35.0,
                f"over the LEGAL cells the fork's reach swept "
                f"{min(reaches)}-{max(reaches)} mm, a span of {span:.1f} mm "
                "toward a measured saturation line near 300 mm")
    all_reaches = [x["fork_reach_mm"] for x in cells]
    worst_inf = (min(x["worst_penetration_mm"] for x in infeasible)
                 if infeasible else None)
    ok &= check("wider_seats_were_tried_and_are_infeasible",
                len(infeasible) >= 3
                and min(all_reaches) < min(reaches) - 15.0,
                f"{len(infeasible)} cells reached down to {min(all_reaches)} mm "
                f"-- {min(reaches) - min(all_reaches):.1f} mm closer than any "
                "legal cell -- and every one of them starts the cutlery inside "
                f"a wall (worst {worst_inf} mm), so the legal span is bounded "
                "by the drawer and not by the sweep")
    flat = bend(st)
    for x in flat["cells"]:
        x["fork_reach_mm"] = 386.7
    ff = [x for x in flat["cells"] if x["worst_penetration_mm"] >= -2.0]
    ok &= check("reject_a_sweep_that_held_the_reach_fixed",
                (max(x["fork_reach_mm"] for x in ff)
                 - min(x["fork_reach_mm"] for x in ff)) < 35.0,
                "flattening every cell's reach fails the check above")

    unsq = [x for x in feasible if not x["square"]]
    nearest = min(unsq, key=lambda x: x["fork_reach_mm"])
    lowest_stall = min(unsq, key=lambda x: x["fork_stall_mm_median"])
    ok &= check("the_stall_does_not_track_the_reach",
                nearest is not lowest_stall,
                f"the closest unsquared cell ({nearest['fork_reach_mm']} mm) "
                f"stalls {nearest['fork_stall_mm_median']} mm, while the lowest "
                f"stall ({lowest_stall['fork_stall_mm_median']} mm) is at "
                f"{lowest_stall['fork_reach_mm']} mm -- so the residual is not "
                "the reach-driven servo saturation the record attributed it to")
    tracked = bend(st)
    for x in tracked["cells"]:
        x["fork_stall_mm_median"] = round(x["fork_reach_mm"] / 10.0, 1)
    tu = [x for x in tracked["cells"] if not x["square"]]
    ok &= check("reject_a_stall_that_did_track_the_reach",
                min(tu, key=lambda x: x["fork_reach_mm"])
                is min(tu, key=lambda x: x["fork_stall_mm_median"]),
                "a synthetic stall made proportional to reach puts the same "
                "cell at both minima, and the check above would not fire")

    sq = [x for x in feasible if x["square"]]
    med = lambda xs: sorted(xs)[len(xs) // 2]
    sq_stall = med([x["fork_stall_mm_median"] for x in sq])
    un_stall = med([x["fork_stall_mm_median"] for x in unsq])
    ok &= check("squaring_lowers_the_stall_and_still_places_nothing",
                sq_stall < un_stall
                and all(x["per_goal"]["fork_placed"] == 0 for x in sq),
                f"median fork stall {un_stall} mm unsquared against {sq_stall} mm "
                "squared -- the best arrival this tree has recorded, and still "
                "0/10")
    ok &= check("the_refutation_rests_only_on_legal_scenes",
                all(x["worst_penetration_mm"] >= -2.0 for x in feasible)
                and len(feasible) >= 12,
                f"{len(feasible)} of {len(cells)} cells clear the -2 mm "
                "tolerance eval_seeds.py uses for no_initial_penetration, and "
                "only those carry the result")
    burrowed = bend(st)
    for x in burrowed["cells"]:
        x["worst_penetration_mm"] = -10.86
    ok &= check("reject_a_grid_with_no_legal_cell_left",
                not [x for x in burrowed["cells"]
                     if x["worst_penetration_mm"] >= -2.0],
                "burying every cell 10.86 mm into the woodwork -- the depth "
                "seat_dy -0.038 really produces -- leaves nothing to conclude "
                "from")
    ok &= check("seat_probe_controls_all_pass",
                all(st["controls"][k] for k in
                    ("baseline_reproduces", "seat_is_applied",
                     "control_direction_did_not_help", "all_runs_ok")),
                {k: st["controls"][k] for k in
                 ("baseline_reproduces", "seat_is_applied",
                  "control_direction_did_not_help", "all_runs_ok")})

    print("\ndescent tracking")
    tr = json.loads(TRACKING.read_text(encoding="utf-8"))
    wp = tr["waypoints"]
    grasp = [wp["fork_descend"], wp["spoon_descend"]]
    free = [wp["fork_above"], wp["spoon_above"]]
    ok &= check("the_solver_is_not_the_failure",
                all(w["ik_err_mm_median"] < 10.0 and abs(w["dz_mm_median"]) > 20.0
                    for w in grasp),
                "; ".join(f"{k}: solver {wp[k]['ik_err_mm_median']} mm, arm "
                          f"{wp[k]['dz_mm_median']} mm"
                          for k in ("fork_descend", "spoon_descend"))
                + " -- the pose is found and then not reached, which is why "
                  "seven routes that moved the pose changed nothing")
    bentt = bend(tr)
    for k in ("fork_descend", "spoon_descend"):
        bentt["waypoints"][k]["ik_err_mm_median"] = 30.0
    ok &= check("reject_a_solver_that_never_found_the_pose",
                not all(bentt["waypoints"][k]["ik_err_mm_median"] < 10.0
                        for k in ("fork_descend", "spoon_descend")),
                "a 30 mm IK residual would make this a solver problem and the "
                "check above would not fire")
    ok &= check("only_the_waypoints_in_contact_saturate",
                all(w["saturated_joints_per_seed_median"] == 0.0
                    and abs(w["dz_mm_median"]) < 5.0 for w in free)
                and all(w["saturated_joints_per_seed_median"] >= 1.0
                        and w["seeds_with_arm_contact"] == w["n"]
                        for w in grasp),
                f"free via-points: {[w['dz_mm_median'] for w in free]} mm off, "
                f"{[w['saturated_joints_per_seed_median'] for w in free]} joints "
                f"saturated, {[w['seeds_with_arm_contact'] for w in free]}/10 in "
                f"contact -- against {[w['dz_mm_median'] for w in grasp]} mm and "
                f"{[w['saturated_joints_per_seed_median'] for w in grasp]} joints "
                "at the grasps, in contact on every seed. The same arm at a "
                "comparable reach tracks to about a millimetre when nothing is "
                "under the jaws, so the residual is the contact and not the reach")
    bentf = bend(tr)
    bentf["waypoints"]["fork_above"]["saturated_joints_per_seed_median"] = 2.0
    ok &= check("reject_a_probe_whose_control_saturates_too",
                not all(bentf["waypoints"][k]["saturated_joints_per_seed_median"] == 0.0
                        for k in ("fork_above", "spoon_above")),
                "if the free via-points saturated as well, this probe would be "
                "measuring the arm's posture rather than the wall, and the "
                "check above would not fire")
    ok &= check("tracking_probe_controls_all_pass",
                all(tr["controls"].values()),
                {k: v for k, v in tr["controls"].items() if not v} or
                f"all {len(tr['controls'])} pass, including "
                "fewer_saturated_joints_than_the_record_claims, whose bar is "
                "the record's own 'three joints'")

    print("\ncompleteness")
    # Neither probe's headline can distinguish "every cell scored zero" from
    # "some runs never finished", because a dropped run contributes nothing to
    # either.  Recomputed here from the run lists rather than read off the
    # probes' own all_runs_ok flags.
    want_pl = len(pl["spreads"]) * pl["seeds"]
    want_st = len(st["cells"]) * sbase["n"]
    got_pl = sum(1 for r in pl["runs"] if "objects" in r)
    got_st = sum(1 for r in st["runs"] if "objects" in r)
    ok &= check("no_run_was_silently_dropped",
                got_pl == want_pl == len(pl["runs"])
                and got_st == want_st == len(st["runs"]),
                f"placement {got_pl}/{want_pl} runs complete, seat "
                f"{got_st}/{want_st} -- a sweep that lost rollouts reads 0/10 "
                "in exactly the same way as one that ran them all")
    dropped = bend(pl)
    dropped["runs"][0] = {"seed": 0, "spread": 0.0, "error": "RuntimeError: x"}
    ok &= check("reject_a_sweep_that_lost_a_rollout",
                sum(1 for r in dropped["runs"] if "objects" in r)
                != len(pl["spreads"]) * pl["seeds"],
                "replacing one run with an error row is caught")

    print("\ncount")
    rejects = sum(1 for r in results if r["control"].startswith("reject_"))
    ok &= check("control_count_is_pinned",
                len(results) + 1 == EXPECTED_CONTROLS and rejects == EXPECTED_REJECTS,
                f"{len(results) + 1} controls, {rejects} of them rejects, "
                f"against a pinned {EXPECTED_CONTROLS}/{EXPECTED_REJECTS} -- a "
                "control that vanishes fails as loudly as one that goes red")

    EV.mkdir(parents=True, exist_ok=True)
    (EV / "cutlery_placement_controls.json").write_text(
        json.dumps({"all_pass": bool(ok),
                    "controls": results,
                    "n_controls": len(results),
                    "n_rejects": rejects,
                    "pinned": {"controls": EXPECTED_CONTROLS,
                               "rejects": EXPECTED_REJECTS},
                    "published_per_goal": PUBLISHED,
                    "switches": {k: v for k, v in consts.items()
                                 if k != "CUTLERY"}},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
