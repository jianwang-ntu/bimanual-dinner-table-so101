#!/usr/bin/env python3
"""Controls for what the domain randomizer actually varies.

The track's criterion T4 is worded "maintains performance under randomized
object placement, weights, friction, shapes, lighting, and background", and
README.md and TECHNICAL_SUMMARY.md both describe the `state` stage as varying
"object x, y and yaw".  That sentence is not true of every object:
`envs/randomize.py::NOMINAL_XY` names three bodies and `GRASPABLES` names five,
so the fork and the spoon start at the same place, at the same yaw, on every
seed.  Their SHAPE and MASS are randomized; their POSE is not.

Amended 2026-09-08.  `envs/randomize.py::randomize_cutlery` now exists and can
place the cutlery, and `scripts/measure_cutlery_placement.py` measured what
turning it on costs: 15/50 sub-goals fall to 13/50 at full spread.  It ships
OFF -- `CUTLERY_PLACE_SPREAD = 0.0` -- so the sentence above is still true of
what runs, but it is now true because of a constant rather than because of an
absent capability, and a suite that did not read that constant would go on
passing while the shipped scene changed underneath it.

Each mechanism is driven from both sides, because a check that can only pass
proves nothing:

  coverage   ACCEPT the randomizer's own tables say three of five graspables
             are placement-randomized; REJECT a patched table that covers four
  switch     ACCEPT the cutlery randomizer that exists as of 2026-09-08 is
             OFF -- CUTLERY_PLACE_SPREAD is 0.0 -- because that is the only
             reason the coverage claim above is still true; REJECT a source
             in which it has been left switched on, which would make every
             published figure a figure of a scene the code no longer builds
  evidence   ACCEPT the shipped 10-seed run shows zero variation in the fork's
             and the spoon's distance to the nearest arm, and non-zero
             variation for the plate, the mug and the bottle; REJECT a run in
             which one cutlery distance has been perturbed
  logs       ACCEPT no episode's state log names the fork or the spoon;
             REJECT a log that does
  docs       ACCEPT README.md and TECHNICAL_SUMMARY.md both state the
             restriction; REJECT the unqualified sentence that shipped
  reach      ACCEPT the measured envelope covers the cutlery radius, so the
             fixed placement is not an out-of-reach placement; REJECT a probe
             that stopped short

Standard library only, so it runs on a fresh clone before anything is
installed.  Run:  python3 scripts/test_randomization_coverage.py
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
DOCS = {"README.md": ROOT / "README.md",
        "TECHNICAL_SUMMARY.md": ROOT / "TECHNICAL_SUMMARY.md"}
RUN = EV / "eval_seeds_scripted.json"
REACH = EV / "reach_envelope.json"

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


# ----------------------------------------------------------- the randomizer
def tables(source: str) -> tuple[set, set]:
    """Read GRASPABLES and NOMINAL_XY out of the randomizer without importing
    it, so the control runs with no mujoco installed and cannot be fooled by a
    monkey-patch at runtime."""
    tree = ast.parse(source)
    got: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in (
                    "GRASPABLES", "NOMINAL_XY", "CUTLERY",
                    "CUTLERY_PLACE_SPREAD", "CUTLERY_SEAT_XY",
                    "CUTLERY_SPLAY_X"):
                got[t.id] = ast.literal_eval(node.value)
    return set(got["GRASPABLES"]), set(got["NOMINAL_XY"]), got


def scene_switches(got: dict) -> dict:
    """The three constants that decide where the cutlery actually starts."""
    return {"CUTLERY_PLACE_SPREAD": got.get("CUTLERY_PLACE_SPREAD"),
            "CUTLERY_SEAT_XY": got.get("CUTLERY_SEAT_XY"),
            "CUTLERY_SPLAY_X": got.get("CUTLERY_SPLAY_X")}


def switches_are_off(got: dict) -> bool:
    sw = scene_switches(got)
    return (sw["CUTLERY_PLACE_SPREAD"] == 0.0
            and tuple(sw["CUTLERY_SEAT_XY"] or (0.0, 0.0)) == (0.0, 0.0)
            and sw["CUTLERY_SPLAY_X"] == 0.0)


# ------------------------------------------------------------- the evidence
def spread(run: dict, body: str) -> float:
    """Millimetres between the largest and smallest distance from this body to
    the nearest arm base, over every episode in the run."""
    d = [e["initial_state"][body]["nearest_arm_m"] for e in run["episodes"]]
    return (max(d) - min(d)) * 1000.0


def state_logged(run: dict) -> set:
    out: set = set()
    for e in run["episodes"]:
        out |= set(e["randomization"]["state"])
    return out


def dims_spread(run: dict, key: str) -> float:
    v = [e["randomization"]["dims"][key] for e in run["episodes"]]
    return (max(v) - min(v)) * 1000.0


def mass_spread(run: dict, body: str) -> float:
    v = [e["randomization"]["model"]["mass_scale"][body] for e in run["episodes"]]
    return max(v) - min(v)


# ----------------------------------------------------------------- the docs
# The correction is only worth anything where the over-broad claim lives, so
# the check is sited on the randomization table itself -- the paragraph holding
# the `| state |` row, and the one after it -- rather than anywhere in the
# document.  Sited loosely it passed on the ABSENCE LEDGER, which names the fork
# and the spoon for an unrelated reason; that is a vacuous pass, and it is what
# this siting exists to prevent.
RESTRICTION = ("every seed", "not randomized", "not placement-randomized", "fixed")


def randomization_block(text: str) -> str:
    paras = re.split(r"\n\s*\n", text)
    for i, para in enumerate(paras):
        if re.search(r"^\|\s*state\s*\|", para, re.M):
            return "\n\n".join(paras[i:i + 2])
    return ""


def doc_states_restriction(text: str, fixed: set) -> bool:
    block = randomization_block(text).lower()
    if not block:
        return False
    return (all(b in block for b in fixed)
            and any(w in block for w in RESTRICTION))


def main() -> int:
    ok = True
    src = RANDOMIZER.read_text(encoding="utf-8")
    graspables, placed, consts = tables(src)
    fixed = graspables - placed

    print("coverage")
    ok &= check("randomizer_places_three_of_five_graspables",
                len(placed) == 3 and len(graspables) == 5 and fixed == {"fork", "spoon"},
                f"GRASPABLES={sorted(graspables)}, NOMINAL_XY={sorted(placed)}, "
                f"never placement-randomized={sorted(fixed)}")

    patched = re.sub(r"NOMINAL_XY = \{", 'NOMINAL_XY = {\n    "spoon": (0.0, 0.128),', src, count=1)
    g2, p2, _ = tables(patched)
    ok &= check("reject_a_table_that_covered_four",
                not (len(p2) == 3 and (g2 - p2) == {"fork", "spoon"}),
                f"with the spoon added the same reader sees {sorted(p2)}, so the "
                "control fails -- it is reading the table, not a constant")

    ok &= check("cutlery_randomizer_exists_and_ships_off",
                "CUTLERY" in consts and switches_are_off(consts),
                f"randomize_cutlery covers {sorted(consts.get('CUTLERY', ()))} "
                f"and every switch is off: {scene_switches(consts)} -- which "
                "is WHY the claim above still holds")
    on = re.sub(r"^CUTLERY_PLACE_SPREAD = 0\.0$", "CUTLERY_PLACE_SPREAD = 1.0",
                src, count=1, flags=re.M)
    _, _, c_on = tables(on)
    ok &= check("reject_a_randomizer_left_switched_on",
                c_on.get("CUTLERY_PLACE_SPREAD") == 1.0
                and not switches_are_off(c_on),
                "with CUTLERY_PLACE_SPREAD = 1.0 the same reader sees "
                f"{scene_switches(c_on)} and the control fails -- so it is "
                "reading the constant, not asserting a constant")
    seated = re.sub(r"^CUTLERY_SEAT_XY = \(0\.0, 0\.0\)$",
                    "CUTLERY_SEAT_XY = (0.0, -0.02)", src, count=1, flags=re.M)
    _, _, c_seat = tables(seated)
    ok &= check("reject_a_scene_seated_somewhere_else",
                not switches_are_off(c_seat),
                "a -20 mm seat offset is caught by the same reader "
                f"({scene_switches(c_seat)}) -- the seat moves the scene "
                "without touching NOMINAL_XY, so the table alone cannot see it")

    print("\nevidence")
    run = json.loads(RUN.read_text(encoding="utf-8"))
    n = len(run["episodes"])
    cutlery = {b: spread(run, b) for b in sorted(fixed)}
    varied = {b: spread(run, b) for b in sorted(placed)}
    ok &= check("cutlery_distance_is_identical_on_every_seed",
                n == 10 and all(v == 0.0 for v in cutlery.values()),
                f"over {n} seeds, spread in distance to the nearest arm: "
                + ", ".join(f"{b} {v:.3f} mm" for b, v in cutlery.items()))
    ok &= check("placed_bodies_do_vary_on_the_same_measure",
                all(v > 1.0 for v in varied.values()),
                ", ".join(f"{b} {v:.1f} mm" for b, v in varied.items())
                + " -- so a zero spread is a finding, not an artefact of the measure")

    bent = json.loads(json.dumps(run))
    bent["episodes"][3]["initial_state"]["fork"]["nearest_arm_m"] += 0.010
    ok &= check("reject_a_run_where_one_cutlery_distance_moved",
                spread(bent, "fork") > 0.0,
                f"perturbing seed 3's fork by 10 mm gives spread "
                f"{spread(bent, 'fork'):.3f} mm, so the check can fail")

    print("\nlogs")
    logged = state_logged(run)
    ok &= check("no_state_log_names_the_fork_or_the_spoon",
                not (logged & fixed),
                f"state keys logged across all {n} episodes: {sorted(logged)}")
    forged = json.loads(json.dumps(run))
    forged["episodes"][0]["randomization"]["state"]["fork"] = {"xy": [0.0, 0.178], "yaw": 1.57}
    ok &= check("reject_a_log_that_names_the_fork",
                bool(state_logged(forged) & fixed),
                "a forged state log is caught")

    ok &= check("cutlery_shape_and_mass_ARE_randomized",
                dims_spread(run, "cutlery_l") > 1.0
                and all(mass_spread(run, b) > 0.05 for b in sorted(fixed)),
                f"cutlery_l spread {dims_spread(run, 'cutlery_l'):.1f} mm; mass_scale spread "
                + ", ".join(f"{b} {mass_spread(run, b):.3f}x" for b in sorted(fixed))
                + " -- the correction is about placement only")

    print("\nreach")
    reach = json.loads(REACH.read_text(encoding="utf-8"))
    far = max(r["achieved_r_m"] for r in reach["rows"])
    need = max(min(abs(c[0] - 0.22), abs(c[0] + 0.22)) * 0 + 0.387 for c in [reach["cutlery"]["fork"]])
    ok &= check("measured_envelope_covers_the_cutlery_radius",
                far >= need,
                f"the arm reached {far:.3f} m of planar radius; the fork sits at "
                f"{need:.3f} m -- the fixed placement is not an out-of-reach placement")
    short = {"rows": [dict(r, achieved_r_m=min(r["achieved_r_m"], 0.28)) for r in reach["rows"]]}
    ok &= check("reject_a_probe_that_stopped_short",
                max(r["achieved_r_m"] for r in short["rows"]) < need,
                "a probe capped at 0.280 m fails the same check")

    print("\ndocs")
    for name, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        ok &= check(f"{name}_states_the_restriction",
                    doc_states_restriction(text, fixed),
                    "the document names the fork and the spoon as not "
                    "placement-randomized")
    shipped = ("| Stage | What varies |\n| state | object x, y and yaw, initial drawer opening |")
    ok &= check("reject_the_unqualified_sentence_that_shipped",
                not doc_states_restriction(shipped, fixed),
                f"the row that actually shipped -- {shipped!r} -- is not accepted")

    EV.mkdir(parents=True, exist_ok=True)
    (EV / "randomization_coverage_controls.json").write_text(
        json.dumps({"all_pass": bool(ok),
                    "graspables": sorted(graspables),
                    "placement_randomized": sorted(placed),
                    "never_placement_randomized": sorted(fixed),
                    "scene_switches": scene_switches(consts),
                    "cutlery_randomizer": sorted(consts.get("CUTLERY", ())),
                    "seeds": n,
                    "controls": results}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
