#!/usr/bin/env python3
"""Controls for the cutlery-approach sweep, and for the fact it changed nothing.

``scripts/measure_cutlery_approach.py`` tested route (b) of
``F-CUTLERY-LIP-001`` -- approach along the drawer's own axis -- which was the
last of that defect's three named routes still standing.  It is REFUTED:
``fork_placed`` is 0/10 in all eight variants, the fork is never lifted, and
the mechanism the hypothesis rested on is refuted in the OPPOSITE direction to
the prediction.  This suite exists to make that refutation checkable rather
than readable.

The control that matters most is ``defaults``.  The sweep needed a new
module-level knob, ``CUTLERY_APPROACH``, and a knob whose default silently
moved the shipped behaviour would invalidate every published figure in this
repository at once.  So the suite drives ``dinner_table_script()`` itself and
reads the offset the emitted cutlery approach move actually asks for, rather
than reading the constant or the source.

  defaults      ACCEPT the committed default emits cutlery approach moves at
                the pure +z 0.075 m every published figure was measured at,
                for BOTH the fork and the spoon; REJECT the same reader with
                CUTLERY_APPROACH carrying a non-zero y
  knob_live     ACCEPT setting the knob actually moves the emitted target by
                exactly what was set -- an inert knob would produce this
                sweep's null result for a reason that has nothing to do with
                the drawer; REJECT a reader that ignores the knob
  identity      ACCEPT exactly one variant is flagged shipped, it is the
                (dy=0, unsquared) setting, and it ran the full seed count
                with no errors; REJECT the flag moved to another variant
  reproduces    ACCEPT the shipped variant reproduces what the repository
                already publishes from an INDEPENDENT probe -- 15/50
                sub-goals, 24.5 mm median stall, 18.2 mm median bow, 4.1 mm
                median shove, fork_handle then drawer_front -- so the harness
                is shown to agree with the published numbers before it is
                used to argue anything; REJECT a shifted copy
  refutation    ACCEPT no variant places the fork; REJECT a copy in which one
                variant placed it
  neg_control   ACCEPT the spoon is placed by no variant either, as
                F-SPOON-POSE-INFEASIBLE-001 requires of any approach-side
                change; REJECT a copy in which the spoon places
  not_inert     ACCEPT the knob moved the physics -- the blocking geom changes
                identity across the swept offsets and the shove spans more
                than an order of magnitude; REJECT a copy in which every
                variant wears the same wall
  wall_is_not_y ACCEPT the hypothesis' own mechanism is refuted: the offset
                that leans INTO drawer_back is the one with the LOWEST stall,
                and leaning out trades the back wall for the taller front one;
                REJECT a copy ordered the way the hypothesis predicted
  not_near_miss ACCEPT the fork is never lifted off the drawer floor in any
                variant, so none of this is a near miss to be narrated as
                progress; REJECT a copy with a real lift
  no_claim      ACCEPT no variant beats the shipped total, so nothing here is
                a gain and nothing is claimed; REJECT a copy where one does
                and the verdict still reads REFUTED

Standard library only, except ``defaults``/``knob_live``, which must import
the controller because the claim is about what the controller emits.
Run:  python3 scripts/test_cutlery_approach.py
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DOC = EVID / "cutlery_approach.json"

FAILED: list[str] = []
PASSED: list[str] = []


def check(name: str, ok: bool, detail) -> bool:
    (PASSED if ok else FAILED).append(name)
    print(("  ok   " if ok else "  FAIL ") + name + " -- " + str(detail))
    return ok


def load() -> dict:
    return json.loads(DOC.read_text())


def variant(d: dict, dy: float, sq: bool) -> dict:
    for v in d["variants"]:
        if abs(v["dy_m"] - dy) < 1e-12 and bool(v["square"]) is bool(sq):
            return v
    raise SystemExit(f"no variant dy={dy} square={sq} in {DOC}")


# --- controls that must import the controller -------------------------------

def approach_offsets(knob=None) -> dict[str, tuple]:
    """The offset each cutlery approach move actually asks for, from the site.

    Read out of the emitted script, not out of the constant: the claim is
    about what ``dinner_table_script()`` produces.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from envs.randomize import make_env
    from envs import controller as C, scene_source

    model, data, _ = make_env(0)
    scene_source.install(scene_source.make("privileged"))
    keep = C.CUTLERY_APPROACH
    try:
        if knob is not None:
            C.CUTLERY_APPROACH = knob
        script = C.dinner_table_script()
        out = {}
        for entry in script:
            if isinstance(entry, tuple) and entry and entry[0] == "if":
                continue
            for mv in entry[0].values():
                lb = getattr(mv, "label", "")
                if lb in ("fork_above", "spoon_above") and mv.where is not None:
                    site = lb.split("_")[0] + "_grasp"
                    sid = C.mujoco.mj_name2id(
                        model, C.mujoco.mjtObj.mjOBJ_SITE, site)
                    base = data.site_xpos[sid]
                    off = np.asarray(mv.where(model, data), float) - base
                    out[lb] = tuple(round(float(x), 6) for x in off)
        return out
    finally:
        C.CUTLERY_APPROACH = keep


def main() -> int:
    d = load()
    print(f"controls for {DOC.relative_to(ROOT)}")

    # --- defaults ---------------------------------------------------------
    off = approach_offsets()
    want = (0.0, 0.0, 0.075)
    ok = (set(off) == {"fork_above", "spoon_above"}
          and all(o == want for o in off.values()))
    check("defaults", ok, f"emitted {off}, shipped pure +z {want}")
    bad = approach_offsets(knob=(0.0, -0.040, 0.075))
    check("defaults/reject", not all(o == want for o in bad.values()),
          f"knob dy=-0.040 emits {bad} -- the reject the ACCEPT is paired with")

    # --- knob_live --------------------------------------------------------
    moved = approach_offsets(knob=(0.0, -0.040, 0.075))
    ok = all(abs(o[1] - (-0.040)) < 1e-9 and abs(o[2] - 0.075) < 1e-9
             for o in moved.values())
    check("knob_live", ok,
          f"dy=-0.040 moves both approach targets to {moved} -- the knob is "
          "not inert, so the null result is about the drawer and not about a "
          "sweep that swept nothing")
    inert = {k: want for k in moved}
    check("knob_live/reject",
          not all(abs(o[1] - (-0.040)) < 1e-9 for o in inert.values()),
          "a reader that ignored the knob would report the shipped offset")

    # --- identity ---------------------------------------------------------
    ships = [v for v in d["variants"] if v["shipped"]]
    ok = (len(ships) == 1 and abs(ships[0]["dy_m"]) < 1e-12
          and ships[0]["square"] is False
          and ships[0]["n"] == d["seeds"]
          and sum(1 for r in d["runs"] if "error" in r) == 0)
    check("identity", ok,
          f"1 shipped variant dy={ships[0]['dy_m']} square={ships[0]['square']}"
          f" n={ships[0]['n']}/{d['seeds']}, "
          f"{len(d['runs'])} runs, 0 errors")
    forged = copy.deepcopy(d)
    for v in forged["variants"]:
        v["shipped"] = abs(v["dy_m"] - (-0.040)) < 1e-12 and v["square"] is False
    fs = [v for v in forged["variants"] if v["shipped"]]
    check("identity/reject",
          not (len(fs) == 1 and abs(fs[0]["dy_m"]) < 1e-12),
          "flag moved to dy=-0.040 is caught")

    # --- reproduces -------------------------------------------------------
    s = variant(d, 0.0, False)
    pub = {"subgoals_met_total": 15, "fork_stall_mm_median": 24.5,
           "fork_bow_mm_median": 18.2, "fork_shove_mm_median": 4.1}
    got = {k: s[k] for k in pub}
    walls = [c["geom"] for c in s["fork_blocking_contacts"][:2]]
    ok = got == pub and walls == ["fork_handle", "drawer_front"]
    check("reproduces", ok,
          f"shipped variant {got} and walls {walls}; measure_fork_descent.py "
          "recorded 15/50, 24.5 mm, 18.2 mm bow, fork_handle 5288 then "
          "drawer_front 2362 from an independent sweep")
    check("reproduces/reject", {**got, "fork_stall_mm_median": 24.6} != pub,
          "a 0.1 mm shift in the published stall is caught")

    # --- refutation -------------------------------------------------------
    placed = {(v["dy_m"], v["square"]): v["fork_placed"] for v in d["variants"]}
    check("refutation", all(p == 0 for p in placed.values()),
          f"fork_placed 0 in all {len(placed)} variants "
          f"({d['seeds']} seeds each)")
    forged = copy.deepcopy(d)
    forged["variants"][0]["fork_placed"] = 3
    check("refutation/reject",
          not all(v["fork_placed"] == 0 for v in forged["variants"]),
          "a forged 3/10 placement is caught")

    # --- neg_control ------------------------------------------------------
    check("neg_control", all(v["spoon_placed"] == 0 for v in d["variants"]),
          "spoon_placed 0 in all variants -- an approach change must not "
          "place an object F-SPOON-POSE-INFEASIBLE-001 says has no "
          "collision-free grasp pose at all")
    forged = copy.deepcopy(d)
    forged["variants"][0]["spoon_placed"] = 1
    check("neg_control/reject",
          not all(v["spoon_placed"] == 0 for v in forged["variants"]),
          "a spoon placement would mean the probe measures something other "
          "than what it claims, and is caught")

    # --- not_inert --------------------------------------------------------
    top = {v["fork_blocking_contacts"][0]["geom"] for v in d["variants"]}
    shoves = [v["fork_shove_mm_median"] for v in d["variants"]]
    ok = len(top) >= 3 and max(shoves) / max(min(shoves), 1e-9) > 10
    check("not_inert", ok,
          f"blocking geom takes {len(top)} identities {sorted(top)} and the "
          f"median shove spans {min(shoves)}-{max(shoves)} mm across the sweep")
    forged = copy.deepcopy(d)
    for v in forged["variants"]:
        v["fork_blocking_contacts"] = [{"geom": "fork_handle", "steps": 5288}]
        v["fork_shove_mm_median"] = 4.1
    ft = {v["fork_blocking_contacts"][0]["geom"] for v in forged["variants"]}
    check("not_inert/reject", not len(ft) >= 3,
          "a sweep whose knob did nothing would also read 0/10 everywhere, "
          "and is caught")

    # --- wall_is_not_y ----------------------------------------------------
    lowest = min(d["variants"], key=lambda v: v["fork_stall_mm_median"])
    out70 = variant(d, -0.070, True)
    ok = (lowest["dy_m"] > 0
          and out70["fork_blocking_contacts"][0]["geom"] == "drawer_front")
    check("wall_is_not_y", ok,
          f"lowest stall {lowest['fork_stall_mm_median']} mm is at "
          f"dy={lowest['dy_m']:+}, the offset leaning INTO drawer_back, while "
          f"leaning out to dy=-0.070 trades it for {out70['fork_blocking_contacts'][0]['geom']} "
          f"({out70['fork_blocking_contacts'][0]['steps']} steps). The "
          "hypothesis predicted the opposite ordering.")
    # The hypothesis predicted stall RISES with dy: leaning out toward the
    # open front (dy<0) clears drawer_back and arrives lowest, leaning in
    # (dy>0) arrives highest.  The forgery is that ordering, so the reject
    # fires on the world the hypothesis expected rather than on noise.
    forged = copy.deepcopy(d)
    for v in forged["variants"]:
        v["fork_stall_mm_median"] = 40.0 + 100.0 * v["dy_m"]
        v["fork_blocking_contacts"] = [
            {"geom": "drawer_back" if v["dy_m"] >= 0 else "drawer_floor",
             "steps": 4000}]
    lo = min(forged["variants"], key=lambda v: v["fork_stall_mm_median"])
    f70 = variant(forged, -0.070, True)
    check("wall_is_not_y/reject",
          not (lo["dy_m"] > 0
               and f70["fork_blocking_contacts"][0]["geom"] == "drawer_front"),
          f"a copy ordered as predicted -- lowest stall at dy={lo['dy_m']:+} "
          "and no front wall traded for -- is caught")

    # --- not_near_miss ----------------------------------------------------
    lifts = [v["fork_lifted_mm_median"] for v in d["variants"]]
    check("not_near_miss", max(lifts) < 1.0,
          f"median peak fork lift is {min(lifts)}-{max(lifts)} mm across all "
          "variants -- the fork never leaves the drawer floor, so nothing "
          "here is a near miss")
    check("not_near_miss/reject", not max(lifts + [12.0]) < 1.0,
          "a 12 mm lift would be a different finding and is caught")

    # --- no_claim ---------------------------------------------------------
    best = max(v["subgoals_met_total"] for v in d["variants"])
    check("no_claim", best == s["subgoals_met_total"] == 15,
          f"best total over the sweep is {best}/50 and it IS the shipped "
          "setting -- no variant is even neutral, so nothing is shipped and "
          "nothing is claimed")
    forged = copy.deepcopy(d)
    forged["variants"][0]["subgoals_met_total"] = 19
    fb = max(v["subgoals_met_total"] for v in forged["variants"])
    check("no_claim/reject", not fb == 15,
          "a variant beating the shipped total would have to be reported as "
          "a gain, not as a refutation, and is caught")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
