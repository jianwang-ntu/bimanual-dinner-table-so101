#!/usr/bin/env python3
"""Controls for F-FORK-EJECT-001, and the test that can refute it.

The claim is about a MECHANISM, and it corrects a claim this repository already
ships on three surfaces.

    The giver DOES carry the fork -- on 7 of 10 seeds it lifts it out of the
    drawer and holds it over the hand-off site at 84-92 N.  It then loses it in
    3-7 simulator steps (6-14 ms) while the jaw separation does not move at all,
    sitting at 6.8-8.2 mm across a 12 mm handle.  Force that collapses while the
    jaws are still at their tightest is an object being squeezed OUT from
    between two faces, not a grip that opened and not a link that swept it away.
    ``pinch``'s ``squeeze`` is an absolute 5 mm set on a 60 mm mug; on cutlery
    it is 42 percent of the object.

Two things this must NOT be allowed to do, and both are asserted:

  * it must not quietly repair the previous tick's refutation.  F-FORK-HANDOFF-
    001 recorded "the taker does not knock the fork out -- its jaws never come
    within a jaw half-span of it", on a closest approach of 62.3 mm.  That
    number was taken at WAYPOINT ENDS.  At full step resolution the taker's jaw
    meeting point comes within 18.6 mm of the fork and taker bodies are in
    contact with it on 7 of 10 seeds.  The refutation is WRONG AS MEASURED and
    the test says so.
  * it must not replace it with a knock story either.  Seeds 4 and 8 carry the
    fork and lose it with NO taker contact anywhere in the episode, so a knock
    cannot be the mechanism.  Both halves are asserted.

And the published figures must be untouched: this probe adopts nothing.

Run:  python3 scripts/test_fork_knockout.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
# scripts/mutate_fork_knockout.py points this at a mutated COPY so the campaign
# never writes into evidence/.  The path read is printed, so a run against a
# copy can never be mistaken for a run against the tree.
KNOCKOUT = pathlib.Path(
    os.environ.get("FORK_KNOCKOUT_EVIDENCE", str(EVID / "fork_knockout.json")))
RELEASE = pathlib.Path(
    os.environ.get("FORK_RELEASE_EVIDENCE", str(EVID / "fork_release.json")))

CHECKS: list[tuple[str, bool, str]] = []

# envs/dinner_table.xml: fork_handle is a box with half-sizes
# [0.006, 0.043, 0.0025] -- 12 mm across the jaw axis.
HANDLE_MM = 12.0
LOADED_N = 50.0          # "carrying it" rather than brushing it
COLLAPSE_STEPS = 25      # 0.05 s at the model's 0.002 s timestep
# F-FORK-HANDOFF-001's published closest approach, taken at waypoint ends.
PUBLISHED_CLOSEST_MM = 62.3


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def main() -> int:
    print(f"  reading {KNOCKOUT}")
    ko = json.loads(KNOCKOUT.read_text())
    ctl, summ, rows = ko["controls"], ko["summary"], ko["per_seed"]
    ok_rows = [r for r in rows if "error" not in r]
    carried = [r for r in ok_rows if r["carry_run_steps"]]

    # ---- the probe's own controls, read back rather than restated ----------
    for key, why in (
        ("reproduces_published", "the probe is on the shipped configuration"),
        ("link_geoms_are_new", "the arm LINKS are instrumented for the first time"),
        ("detector_can_say_yes", "the contact detector can return True"),
        ("unrelated_body_absent", "the body attribution names nothing spurious"),
        ("loss_is_after_lift", "the loss found is the hand-off, not the pick"),
    ):
        check(f"probe control: {key} -- {why}", ctl[key]["pass"],
              json.dumps({k: v for k, v in ctl[key].items()
                          if k not in ("pass", "why_it_matters")})[:220])

    # ---- the claim ---------------------------------------------------------
    check("the giver really carries the fork on a majority of seeds",
          len(carried) >= 6, f"carried on seeds {[r['seed'] for r in carried]}")

    forces = {r["seed"]: r["carry_grip_force_N"]["max"] for r in carried}
    check("the carry is LOADED -- peak jaw-on-fork force is far above a brush "
          "on every carried seed",
          all(v > LOADED_N for v in forces.values()), str(forces))

    at_loss = {r["seed"]: r["carry_grip_force_N"]["at_loss"] for r in carried}
    check("and the force is GONE at the loss step, on every carried seed",
          all(v < 1.0 for v in at_loss.values()), str(at_loss))

    # the collapse is fast: from loaded to nothing inside the window
    def collapse(r) -> int | None:
        v = r["carry_grip_force_N"]["last_20_steps"]
        hi = max((i for i, x in enumerate(v) if x > LOADED_N), default=None)
        if hi is None:
            return None
        return next((i - hi for i, x in enumerate(v) if i > hi and x < 1.0), None)

    cols = {r["seed"]: collapse(r) for r in carried}
    check("the collapse is a release, not a slide: loaded to nothing in under "
          "a twentieth of a second, on every carried seed",
          all(c is not None and c <= COLLAPSE_STEPS for c in cols.values()),
          f"steps from >{LOADED_N} N to <1 N: {cols}")

    seps = {r["seed"]: r["carry_jaw_sep_mm"] for r in carried}
    check("and the jaws DO NOT MOVE while it happens -- separation at the loss "
          "is the tightest of the whole carry, on every carried seed",
          all(abs(v["at_loss"] - v["min"]) < 0.05 for v in seps.values()),
          str({k: (v["min"], v["at_loss"]) for k, v in seps.items()}))

    check(f"those jaws are INSIDE the {HANDLE_MM} mm handle by 3-6 mm, which is "
          f"what the absolute 5 mm squeeze buys on cutlery",
          all(HANDLE_MM - v["at_loss"] > 3.0 for v in seps.values()),
          str({k: round(HANDLE_MM - v["at_loss"], 1) for k, v in seps.items()}))

    # ---- AGAINST US, 1: the previous tick's refutation was wrong -----------
    closest = min(r["min_taker_tip_mm"] for r in ok_rows)
    check("CORRECTION, kept rather than dropped: F-FORK-HANDOFF-001's "
          f"'the taker never comes within a jaw half-span' ({PUBLISHED_CLOSEST_MM} mm) "
          "was measured at waypoint ends and is WRONG at step resolution",
          closest < 45.0,
          f"closest taker meeting point to the fork, every step: {closest} mm "
          f"against the published {PUBLISHED_CLOSEST_MM} mm")

    touched = [r["seed"] for r in ok_rows
               if r["first_taker_contact_step"] is not None]
    check("and taker bodies DO touch the fork, on most seeds",
          len(touched) >= 5, f"seeds {touched}")

    # ---- AGAINST US, 2: but a knock is still not the mechanism -------------
    clean = [r["seed"] for r in carried if r["first_taker_contact_step"] is None]
    check("REFUTED IN THE OTHER DIRECTION TOO: a taker knock cannot be the "
          "mechanism -- there are carried seeds that lose the fork with no "
          "taker contact anywhere in the episode",
          len(clean) >= 1, f"seeds {clean}")

    # ---- AGAINST US, 3: nothing was shipped, no figure moved ---------------
    check("the probe touched no controller knob",
          ko["knobs_touched"] == [] and ko["shipped_controller"] is True)
    if RELEASE.exists():
        fr = json.loads(RELEASE.read_text())
        check("fork_placed is UNCHANGED at 3/10 -- this is a mechanism "
              "correction, not a rewrite of an inconvenient figure",
              sum(1 for v in fr["summary"]["placed"].values() if v) == 3,
              str(fr["summary"]["placed"]))

    # ---- the code fact the finding turns on --------------------------------
    src = (ROOT / "envs" / "controller.py").read_text()
    check("pinch's squeeze is an absolute distance, not a fraction of the "
          "object", "def pinch(width_fn, squeeze: float = 0.005" in src)
    check("both cutlery pinches now read one named constant, so the quantity "
          "can be swept",
          src.count("squeeze=CUTLERY_SQUEEZE") == 2
          and "CUTLERY_SQUEEZE = " in src)

    print(f"\n  carried seeds: {[r['seed'] for r in carried]}")
    print(f"  peak carry force N: {forces}")
    print(f"  jaw separation at loss (mm, handle is {HANDLE_MM}): "
          f"{ {k: v['at_loss'] for k, v in seps.items()} }")
    print(f"  verdicts: {summ['verdicts']}")

    bad = [c for c in CHECKS if not c[1]]
    print()
    for n, okc, det in CHECKS:
        print(f"  [{'PASS' if okc else 'FAIL'}] {n}" + (f"  -- {det}" if det else ""))
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
