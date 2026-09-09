#!/usr/bin/env python3
"""Controls for F-AIS-TAKEPLAN-001 -- and for its REFUTED remedy.

Two claims, and they point opposite ways.  Both are pinned here so neither can
be quietly dropped by a later reader who prefers the other.

CLAIM 1, in our favour, MEASURED:
    ``_handoff``'s two TAKE waypoints are solved at the pinch width and then
    executed at the OPEN width, because ``Rollout._plan`` overwrites the solved
    gripper value with ``opening`` after the solve.  ``Gripper.tip_mid`` is the
    midpoint between the jaw faces, so it moves by half the opening change.  The
    commanded joint vector therefore puts the taker's jaws 43.9-46.5 mm from the
    point it was solved onto -- on 10 of 10 seeds, before the arm has moved, and
    against a 45 mm fork placement tolerance.  The giver's own descend, which
    already has the fix (``CUTLERY_PLAN_AT_OPEN`` drops its ``plan_at``), lands
    at 1.5-1.9 mm through the same solver onto the same site.

CLAIM 2, AGAINST us, MEASURED:
    removing that error does not improve the score.  Four arms, ten seeds each:
    shipped 18/50, plan_at_open 15/50, plan_at_open+30 mm 15/50, 30 mm 16/50,
    task_success 0/10 in all four.  Neither knob is adopted.  Nor is harm
    claimed -- 15-18/50 is inside the scatter the CUTLERY_SQUEEZE sweep already
    recorded for this quantity at n=10.

The falsifiable form:

  * if the knobs are not inert at their defaults, the shipped robot moved and
    nothing below is about the shipped robot;
  * if the knobs are inert when turned ON, they are not connected to anything
    and CLAIM 1's arm measured nothing;
  * if the forward kinematics behind plan_err cannot reproduce the simulator's
    own jaws, plan_err is a number about the probe;
  * if the giver's positive control is not small, plan_err is about the
    instrument rather than the arms;
  * if any arm beats shipped, CLAIM 2 is false and something should have been
    adopted.

Each of the four evidence-backed checks is re-run against a corruption DERIVED
FROM THE ARTIFACT (never a pinned literal), and each corruption must flip it.

Run:  python3 scripts/test_take_waypoint.py
"""
from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

SERVO = EVID / "taker_servo.json"
SERVO_ON = EVID / "taker_servo_planatopen.json"
AB = EVID / "take_waypoint_ab.json"
REACH = EVID / "taker_reach.json"

PLAN_ERR_LO, PLAN_ERR_HI = 40.0, 50.0      # the taker band the claim asserts
GIVER_MAX_MM = 5.0                          # the giver's band, same solver
SEEDS = 10

CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def taker_band(servo):
    v = [x for x in servo["summary"]["plan_err_mm_fork_take"].values() if x is not None]
    return v


def giver_band(servo):
    v = [x for x in servo["summary"]["plan_err_mm_fork_descend"].values() if x is not None]
    return v


def finding_holds(servo) -> bool:
    t, g = taker_band(servo), giver_band(servo)
    return (len(t) == SEEDS and len(g) == SEEDS
            and all(PLAN_ERR_LO <= x <= PLAN_ERR_HI for x in t)
            and sum(1 for x in g if x <= GIVER_MAX_MM) >= SEEDS - 1
            and min(t) > max(x for x in g if x <= GIVER_MAX_MM))


def refutation_holds(ab) -> bool:
    arms = ab["arms"]
    ship = arms["A_shipped"]["subgoals_total"]
    return (all(v["subgoals_total"] <= ship for v in arms.values())
            and all(v["task_success_count"] == 0 for v in arms.values()))


def main() -> int:
    for p in (SERVO, SERVO_ON, AB, REACH):
        if not p.exists():
            print(f"MISSING EVIDENCE: {p}")
            return 2
    servo = json.loads(SERVO.read_text())
    servo_on = json.loads(SERVO_ON.read_text())
    ab = json.loads(AB.read_text())
    reach = json.loads(REACH.read_text())
    src = (ROOT / "envs" / "controller.py").read_text(encoding="utf-8")

    # ---- the knobs exist, default to the shipped values, and are inert ------
    check("CUTLERY_TAKE_PLAN_AT_OPEN defaults to the shipped False",
          re.search(r'CUTLERY_TAKE_PLAN_AT_OPEN = os\.environ\.get\(\s*'
                    r'"CUTLERY_TAKE_PLAN_AT_OPEN", "0"\)', src) is not None)
    check("CUTLERY_TAKE_OFFSET defaults to the shipped 0.0",
          'os.environ.get("CUTLERY_TAKE_OFFSET", "0.0")' in src)
    check("both knobs are threaded to BOTH _handoff call sites",
          src.count("plan_at_open=CUTLERY_TAKE_PLAN_AT_OPEN") == 2
          and src.count("take_offset=CUTLERY_TAKE_OFFSET") == 2,
          f'{src.count("plan_at_open=CUTLERY_TAKE_PLAN_AT_OPEN")} / '
          f'{src.count("take_offset=CUTLERY_TAKE_OFFSET")}')
    check("at the defaults the shipped ten-number column still reproduces "
          "to 0.0 mm -- the edits did not move the shipped robot",
          servo["controls"]["reproduces_published"]["pass"]
          and all(d == 0.0 for d in
                  servo["controls"]["reproduces_published"]["abs_delta_mm"]),
          str(servo["controls"]["reproduces_published"]["abs_delta_mm"]))

    # ---- the instrument can be trusted -------------------------------------
    for k in ("fk_matches_sim", "positive_control_giver", "clip_detector_fires",
              "contact_detector_fires", "labels_all_present"):
        check(f"servo control {k} PASSES", servo["controls"][k]["pass"],
              json.dumps(servo["controls"][k].get("max_err_mm_by_seed")
                         or servo["controls"][k].get("seeds_over_tol") or "")[:120])
    check("the reach probe's own four controls PASS",
          all(v["pass"] for v in reach["controls"].values()),
          str({k: v["pass"] for k, v in reach["controls"].items()}))

    # ---- CLAIM 1 -----------------------------------------------------------
    t, g = taker_band(servo), giver_band(servo)
    check(f"taker plan_err at fork_take is {PLAN_ERR_LO}-{PLAN_ERR_HI} mm on "
          f"all {SEEDS} seeds", finding_holds(servo),
          f"taker {min(t)}-{max(t)}  giver {min(g)}-{max(g)}")
    check("the taker's WORST-CASE giver comparison still separates them: "
          "min(taker) > max(giver, grasping seeds)",
          min(t) > max(x for x in g if x <= GIVER_MAX_MM),
          f"{min(t)} > {max(x for x in g if x <= GIVER_MAX_MM)}")
    check("the ik residual is NOT the defect -- the solver finds the pose",
          all(v is not None and v < 10.0
              for v in reach["summary"]["ik_err_mm_fork_take"].values()),
          str(sorted(reach["summary"]["ik_err_mm_fork_take"].values()))[:90])
    check("nothing is clipped, so the command is not truncated by the range",
          all((v or 0.0) == 0.0
              for v in servo["summary"]["max_clip_rad_fork_take"].values()))

    # ---- the knob is LIVE when on ------------------------------------------
    on = [x for x in servo_on["summary"]["plan_err_mm_fork_take"].values()
          if x is not None]
    check("with the knob ON plan_err collapses below 10 mm on all seeds",
          len(on) == SEEDS and all(x < 10.0 for x in on),
          f"{min(on)}-{max(on)}")
    check("with the knob ON the shipped column DELIBERATELY stops reproducing "
          "-- recorded, not hidden; an ON arm that still reproduced would "
          "prove the knob inert",
          servo_on["controls"]["reproduces_published"]["pass"] is False)

    # ---- CLAIM 2, against us ------------------------------------------------
    arms = ab["arms"]
    ship = arms["A_shipped"]["subgoals_total"]
    check("AGAINST US: no arm beats shipped, so nothing is adopted",
          refutation_holds(ab),
          str({k: v["subgoals_total"] for k, v in arms.items()}))
    check("AGAINST US: task_success is 0/10 in every arm -- the row's own bar "
          "is untouched by any of this",
          ab["task_success_in_every_arm"] == [0])
    check("the four arms are four different experiments",
          ab["controls"]["arms_differ"]["pass"],
          str(ab["controls"]["arms_differ"]["distinct_per_seed_columns"]))
    check("HARM IS NOT CLAIMED -- the record says NO MEASURED IMPROVEMENT and "
          "cites the prior n=10 noise measurement",
          "not_claimed" in ab and "noise" in ab["not_claimed"]
          and "NO MEASURED" in ab["not_claimed"].upper())
    check("the source records the refutation next to the knob, so the axis is "
          "not re-proposed by the next reader",
          "MEASURED AND NOT ADOPTED" in src
          and src.count("MEASURED AND NOT ADOPTED") == 2,
          str(src.count("MEASURED AND NOT ADOPTED")))

    # ---- NEGATIVE CONTROLS, each derived from the artifact ------------------
    m = copy.deepcopy(servo)
    band = m["summary"]["plan_err_mm_fork_take"]
    gv = m["summary"]["plan_err_mm_fork_descend"]
    for k in band:                       # make the taker look like the giver
        band[k] = gv.get(k, 1.7)
    check("NEGATIVE CONTROL: flattening the taker band onto the giver's own "
          "measured values breaks CLAIM 1", not finding_holds(m))

    m2 = copy.deepcopy(servo)
    keys = list(m2["summary"]["plan_err_mm_fork_take"])
    m2["summary"]["plan_err_mm_fork_take"].pop(keys[0])
    check("NEGATIVE CONTROL: dropping one seed from the band breaks CLAIM 1 -- "
          "'on all ten seeds' is checked, not assumed", not finding_holds(m2))

    m3 = copy.deepcopy(ab)
    worst = min(m3["arms"], key=lambda k: m3["arms"][k]["subgoals_total"])
    m3["arms"][worst]["subgoals_total"] = ship + 1
    check("NEGATIVE CONTROL: raising the weakest arm one subgoal above shipped "
          "breaks CLAIM 2 -- the refutation would have to be withdrawn",
          not refutation_holds(m3))

    m4 = copy.deepcopy(ab)
    m4["arms"]["A_shipped"]["task_success_count"] = 1
    check("NEGATIVE CONTROL: a single task_success anywhere breaks the "
          "'0/10 in every arm' check",
          not all(v["task_success_count"] == 0 for v in m4["arms"].values()))

    print(f"\n  taker plan_err fork_take, mm : {sorted(t)}")
    print(f"  giver plan_err fork_descend  : {sorted(g)}")
    print(f"  arms                         : "
          f"{ {k: v['subgoals_total'] for k, v in arms.items()} }")

    bad = [c for c in CHECKS if not c[1]]
    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"  -- {det}" if det else ""))
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
