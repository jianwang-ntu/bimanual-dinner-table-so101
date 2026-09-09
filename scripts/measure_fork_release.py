#!/usr/bin/env python3
"""Where does the fork's placement error come from -- the carry, the grip, or the drop?

H-CUTLERY-SQUARE-001 moved this project's blocker.  Squaring the cutlery pick
took ``fork_placed`` from 0/10 to 3/10, and the residual looked like a release
failure: on the adopted cell the fork peaks 118-123 mm above its start on 7 of
10 seeds and then scores 14.0-143.6 mm from ``target_fork`` against a 45 mm
tolerance.  The manifest's next_step therefore said "THE RELEASE, NOT THE
GRASP" and asked for a direct measurement of where the fork sits in the jaws at
release rather than another sweep.

This is that measurement, and it does not answer the question it was asked.
It walks the WHOLE fork phase -- the right arm's pick, the hand-off, and the
left arm's place -- and at every waypoint records three things about the fork:
which arm's jaws are touching it, how far it is from that arm's jaw meeting
point, and its height.  The three together say whether the fork is being
carried at all.

Quantities, all in the horizontal plane the scorer uses:

  carry_mm   |tip_xy - target_xy| at the end of ``target_fork_down``.
             How close the jaw MEETING POINT got to the site ``_place`` aims.
  grip_mm    |fork_xy - tip_xy| at the same instant.  Where the fork is
             relative to those jaws.  It is only a GRIP offset if the fork is
             in the jaws, which is what ``held`` decides and why it is
             recorded next to it rather than assumed.
  drift_mm   |fork_xy(final) - fork_xy(release)| -- how far the fork travels
             after the jaws open.

At the instant of release (fork - target) = (tip - target) + (fork - tip)
exactly, and the probe asserts that identity closes to 0.5 mm per seed before
reporting anything.  A decomposition that does not close is arithmetic.

Controls.  Each can fail, each is reported as it comes out:

  reproduces_published   the per-seed ``final_to_target_mm`` column must
                         reproduce the 10 numbers the adopted cell published in
                         evidence/cutlery_square.json.per_seed_best_cell, and
                         the placed column with it.  Pinned to the literal, so
                         a drift in the controller or in that file fails this
                         control instead of hiding inside it.  Without it every
                         offset below could belong to a neighbouring
                         configuration.
  decomposition_closes   the identity above, per seed.
  labels_present         every seed must reach every watched waypoint.
  detector_can_say_yes   THE POSITIVE CONTROL FOR ``held``.  A contact test
                         that never returns True proves nothing by returning
                         False, so the same detector, on the same rollout, must
                         report the RIGHT arm holding the fork somewhere in its
                         own pick -- which is independently corroborated by the
                         fork rising out of the drawer.  If this fails, every
                         "not held" below is void and nothing may be concluded
                         from it.
  height_agrees_with_held  an INDEPENDENT cross-check of the same claim that
                         uses no contact data at all: a fork that is not held
                         at the end of a waypoint should be resting on
                         something -- table top (~752 mm) or drawer (~783 mm) --
                         not hovering.  Contact and geometry have to agree.
  offsets_are_measured   ``grip_mm`` must not be identically zero everywhere.

Writes evidence/fork_release.json.
Run:  python3 scripts/measure_fork_release.py --seeds 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

BODY = "fork"
TARGET = "target_fork"
PICKER = "right"        # _pick(("right", "fork", ...)) in dinner_table_script
PLACER = "left"         # _handoff("right", "left", "fork", "target_fork", ...)

# Every waypoint of the fork phase, in script order.  Labels come from
# _pick / _handoff / _place; they are asserted present rather than assumed.
PICK_LABELS = ("fork_above", "fork_descend", "fork_close", "fork_lift")
HANDOFF_LABELS = ("fork_present", "fork_meet", "fork_take_above", "fork_take",
                  "fork_taker_close", "fork_giver_release", "fork_giver_clear",
                  "fork_taker_lift")
PLACE_LABELS = ("target_fork_over", "target_fork_down", "target_fork_release",
                "target_fork_retreat")
WATCH = PICK_LABELS + HANDOFF_LABELS + PLACE_LABELS

# evidence/cutlery_square.json -> per_seed_best_cell, the adopted cell.
PUBLISHED_FINAL_MM = [158.9, 149.4, 14.0, 54.6, 70.2, 83.0, 159.8, 35.9,
                      143.6, 20.3]
PUBLISHED_PLACED = [False, False, True, False, False, False, False, True,
                    False, True]
REPRO_TOL_MM = 1.0

# envs/task.py: TABLE_TOP_Z = 0.75.  The fork's handle half-thickness is
# 0.0025 m, so a fork lying on the table reads ~752.5 mm; its drawer pose in
# envs/dinner_table.xml is z=0.784.  Both are RESTING heights.
RESTING_MM = (752.5, 783.5)
RESTING_TOL_MM = 6.0


def _mm(v) -> float:
    return round(float(v) * 1000.0, 1)


def _entry_groups(trace: list[dict]) -> list[tuple[int, int]]:
    """(start, end_exclusive) index ranges of trace rows dispatched together.

    ``Rollout.run`` appends one row per arm BEFORE stepping, so every row of one
    script entry carries the same ``t``; time always advances afterwards because
    each entry runs at least one step.  Grouping on equal consecutive ``t`` is
    therefore exact, and it is what lets a two-arm entry report BOTH labels
    instead of only the last one.
    """
    groups, i = [], 0
    while i < len(trace):
        j = i + 1
        while j < len(trace) and trace[j]["t"] == trace[i]["t"]:
            j += 1
        groups.append((i, j))
        i = j
    return groups


def run(seed: int) -> dict:
    from envs.task import TaskMonitor, PLACEMENTS, gripper_geoms
    from envs import randomize as R
    from envs import controller as C
    from envs import scene_source

    model, data, _log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET)
    fork_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
    jaws = {a: gripper_geoms(model, f"{a}_") for a in (PICKER, PLACER)}

    roll = C.Rollout(model, data)
    grips = {a: C._grip(model, a) for a in (PICKER, PLACER)}

    def held_by(d) -> list[str]:
        out = set()
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if g1 in fork_geoms:
                other = g2
            elif g2 in fork_geoms:
                other = g1
            else:
                continue
            for a, js in jaws.items():
                if other in js:
                    out.add(a)
        return sorted(out)

    # samples[n] is overwritten every step, so what survives for each n is the
    # LAST step executed while that many rows had been dispatched -- the end of
    # that entry, after its ramp and its settle.
    samples: dict[int, dict] = {}

    def on_step(d):
        n = len(roll.trace)
        if n == 0:
            return
        samples[n] = {
            "t": round(float(d.time), 3),
            "held": held_by(d),
            "fork": d.xpos[bid].copy(),
            "target": d.site_xpos[sid].copy(),
            "tip": {a: grips[a].tip_mid(model, d).copy() for a in grips},
        }

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    # ---- map each sample to the labels of the entry it belongs to -----------
    groups = _entry_groups(roll.trace)
    end_to_labels = {j: [roll.trace[k]["label"] for k in range(i, j)]
                     for i, j in groups}
    end_to_arms = {j: [roll.trace[k]["arm"] for k in range(i, j)]
                   for i, j in groups}
    # The solver residual Rollout already records for each dispatched move.
    # It separates "the pose could not be solved" from "the pose was solved and
    # the arm did not get there", which the tip distance alone cannot.
    end_to_ikerr = {j: [roll.trace[k]["ik_err_mm"] for k in range(i, j)]
                    for i, j in groups}

    waypoints: dict[str, dict] = {}
    for n, s in samples.items():
        labels = end_to_labels.get(n)
        if not labels:
            continue
        for lab, arm, ikerr in zip(labels, end_to_arms[n], end_to_ikerr[n]):
            if lab not in WATCH:
                continue
            tip = s["tip"][arm] if arm in s["tip"] else None
            row = {
                "t": s["t"],
                "arm": arm,
                "ik_err_mm": ikerr,
                "held_by": s["held"],
                "fork_z_mm": _mm(s["fork"][2]),
                "resting": any(abs(_mm(s["fork"][2]) - r) <= RESTING_TOL_MM
                               for r in RESTING_MM),
            }
            if tip is not None:
                row["tip_to_fork_mm"] = _mm(
                    np.linalg.norm(s["fork"][:2] - tip[:2]))
                row["tip_z_mm"] = _mm(tip[2])
            # Both arms' jaws at every fork-phase waypoint, so the question
            # "did the taker arrive on top of the giver" is answered by a
            # distance and not by the order of the labels.
            gt, tt = s["tip"][PICKER], s["tip"][PLACER]
            row["giver_tip_to_fork_mm"] = _mm(np.linalg.norm(s["fork"] - gt))
            row["taker_tip_to_fork_mm"] = _mm(np.linalg.norm(s["fork"] - tt))
            row["tip_separation_mm"] = _mm(np.linalg.norm(gt - tt))
            if lab.startswith("target_fork"):
                carry_v = tip[:2] - s["target"][:2]
                grip_v = s["fork"][:2] - tip[:2]
                err_v = s["fork"][:2] - s["target"][:2]
                row.update({
                    "carry_mm": _mm(np.linalg.norm(carry_v)),
                    "grip_mm": _mm(np.linalg.norm(grip_v)),
                    "err_mm": _mm(np.linalg.norm(err_v)),
                    "carry_vec_mm": [_mm(v) for v in carry_v],
                    "grip_vec_mm": [_mm(v) for v in grip_v],
                    "closes_mm": _mm(np.linalg.norm(err_v - (carry_v + grip_v))),
                    "fork_above_target_mm": _mm(s["fork"][2] - s["target"][2]),
                    "fork_xy_m": [float(s["fork"][0]), float(s["fork"][1])],
                })
            waypoints[lab] = row

    final_xy = data.xpos[bid][:2].copy()
    tgt_xy = data.site_xpos[sid][:2].copy()
    rel = waypoints.get("target_fork_release")

    return {
        "seed": seed,
        "final_to_target_mm": _mm(np.linalg.norm(final_xy - tgt_xy)),
        "final_z_mm": _mm(data.xpos[bid][2]),
        "placed": bool(rep["subgoals"][f"{BODY}_placed"]),
        "subgoals_met": int(sum(rep["subgoals"].values())),
        "monitor_handoffs": [h for h in rep["handoffs"] if h["object"] == BODY],
        "drift_after_release_mm": (
            None if rel is None or "fork_xy_m" not in rel else
            _mm(np.linalg.norm(final_xy - np.asarray(rel["fork_xy_m"])))),
        "waypoints": waypoints,
        "labels_missing": [l for l in WATCH if l not in waypoints],
        "ever_held_by": sorted({a for w in waypoints.values()
                                for a in w["held_by"]}),
    }


def _job(seed: int) -> dict:
    try:
        return run(seed)
    except Exception as exc:                       # keep the sweep going
        import traceback
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(EVID / "fork_release.json"))
    a = ap.parse_args()

    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, range(a.seeds)):
            rows.append(r)
            if "error" in r:
                print(f"  seed={r['seed']:<2} ERROR {r['error']}")
            else:
                d = r["waypoints"].get("target_fork_down", {})
                print(f"  seed={r['seed']:<2} final={r['final_to_target_mm']:>6} "
                      f"carry={d.get('carry_mm', '-'):>6} grip={d.get('grip_mm', '-'):>6} "
                      f"held_ever={r['ever_held_by']} placed={r['placed']}")
    rows.sort(key=lambda r: r["seed"])
    ok = [r for r in rows if "error" not in r]

    def wp(r, lab, key, default=None):
        return (r["waypoints"].get(lab) or {}).get(key, default)

    # ---------------------------------------------------------------- controls
    got_final = [r["final_to_target_mm"] for r in ok]
    deltas = [round(abs(g - p), 2)
              for g, p in zip(got_final, PUBLISHED_FINAL_MM[:len(got_final)])]
    closes = [wp(r, "target_fork_release", "closes_mm") for r in ok
              if wp(r, "target_fork_release", "closes_mm") is not None]

    # POSITIVE control: the same detector must be able to say True.
    picker_holds = {r["seed"]: [l for l in PICK_LABELS + HANDOFF_LABELS
                                if PICKER in (wp(r, l, "held_by") or [])]
                    for r in ok}
    placer_holds = {r["seed"]: [l for l in HANDOFF_LABELS + PLACE_LABELS
                                if PLACER in (wp(r, l, "held_by") or [])]
                    for r in ok}

    # INDEPENDENT cross-check: unheld at a waypoint => resting on something.
    disagreements = []
    for r in ok:
        for lab, w in r["waypoints"].items():
            if not w["held_by"] and not w["resting"]:
                disagreements.append({"seed": r["seed"], "label": lab,
                                      "fork_z_mm": w["fork_z_mm"]})

    controls = {
        "reproduces_published": {
            "pass": bool(len(got_final) == len(PUBLISHED_FINAL_MM)
                         and all(d <= REPRO_TOL_MM for d in deltas)
                         and [r["placed"] for r in ok] == PUBLISHED_PLACED[:len(ok)]),
            "pinned_literal": PUBLISHED_FINAL_MM,
            "measured": got_final,
            "abs_delta_mm": deltas,
            "tol_mm": REPRO_TOL_MM,
            "placed_pinned": PUBLISHED_PLACED,
            "placed_measured": [r["placed"] for r in ok],
        },
        "decomposition_closes": {
            "pass": bool(closes and max(closes) <= 0.5),
            "max_residual_mm": max(closes) if closes else None,
        },
        "labels_present": {
            "pass": all(not r["labels_missing"] for r in ok) and len(ok) == len(rows),
            "missing": {r["seed"]: r["labels_missing"] for r in ok
                        if r["labels_missing"]},
            "errors": [{k: v for k, v in r.items() if k != "traceback"}
                       for r in rows if "error" in r],
        },
        "detector_can_say_yes": {
            "pass": any(v for v in picker_holds.values()),
            "picker_arm": PICKER,
            "labels_where_picker_holds_the_fork": picker_holds,
            "why_it_matters": (
                "a contact test that never returns True cannot support a "
                "'never held' conclusion; the picker's own grasp is the "
                "positive case, corroborated independently by the fork "
                "leaving the drawer"),
        },
        "height_agrees_with_held": {
            "pass": not disagreements,
            "disagreements": disagreements,
            "resting_heights_mm": RESTING_MM,
            "tol_mm": RESTING_TOL_MM,
        },
        "offsets_are_measured": {
            "pass": any((wp(r, "target_fork_down", "grip_mm") or 0) > 0
                        for r in ok),
        },
    }

    # --------------------------------------------------------------- the answer
    summary = {
        "placer_arm": PLACER,
        "seeds_where_placer_ever_holds_the_fork":
            [s for s, v in placer_holds.items() if v],
        "placer_hold_labels": placer_holds,
        "picker_hold_labels": picker_holds,
        "fork_z_at_taker_lift_mm": {r["seed"]: wp(r, "fork_taker_lift", "fork_z_mm")
                                    for r in ok},
        "fork_resting_at_taker_lift": {r["seed"]: wp(r, "fork_taker_lift", "resting")
                                       for r in ok},
        "at_down": {r["seed"]: {"carry_mm": wp(r, "target_fork_down", "carry_mm"),
                                "grip_mm": wp(r, "target_fork_down", "grip_mm"),
                                "err_mm": wp(r, "target_fork_down", "err_mm"),
                                "held_by": wp(r, "target_fork_down", "held_by"),
                                "resting": wp(r, "target_fork_down", "resting")}
                    for r in ok},
        "monitor_handoffs_recorded": {r["seed"]: len(r["monitor_handoffs"])
                                      for r in ok},
        # Where the fork is lost: the last waypoint at which EITHER arm's jaws
        # are touching it, and how close the taker's jaws were at that moment.
        # The closest the TAKER's jaws ever come to the fork, over every
        # hand-off waypoint and every seed.  The taker is commanded to
        # fork_grasp + 40 mm and then + 4 mm; how near it actually gets, and
        # what the solver said about that pose, separate a solver failure from
        # a tracking failure.
        "taker_closest_approach_mm": min(
            [v for r in ok for l in HANDOFF_LABELS
             for v in [wp(r, l, "taker_tip_to_fork_mm")] if v is not None]
            or [None]),
        "taker_ik_err_at_take_mm": {r["seed"]: wp(r, "fork_take", "ik_err_mm")
                                    for r in ok},
        "taker_ik_err_at_take_above_mm": {
            r["seed"]: wp(r, "fork_take_above", "ik_err_mm") for r in ok},
        "last_contact_waypoint": {
            r["seed"]: next((l for l in reversed(WATCH)
                             if (wp(r, l, "held_by") or [])), None)
            for r in ok},
        "geometry_at_handoff": {
            r["seed"]: {l: {"giver_mm": wp(r, l, "giver_tip_to_fork_mm"),
                            "taker_mm": wp(r, l, "taker_tip_to_fork_mm"),
                            "tip_sep_mm": wp(r, l, "tip_separation_mm"),
                            "fork_z_mm": wp(r, l, "fork_z_mm"),
                            "ik_err_mm": wp(r, l, "ik_err_mm"),
                            "held_by": wp(r, l, "held_by")}
                        for l in ("fork_meet", "fork_take_above", "fork_take",
                                  "fork_taker_close")}
            for r in ok},
        "final_to_target_mm": {r["seed"]: r["final_to_target_mm"] for r in ok},
        "placed": {r["seed"]: r["placed"] for r in ok},
    }

    doc = {
        "probe": "measure_fork_release.py",
        "question": ("Of the fork's placement error, how much is the carry, "
                     "how much is the grip offset, and how much is post-release "
                     "drift -- and is the fork in the placing arm's jaws at all?"),
        "shipped_controller": True,
        "knobs_touched": [],
        "seeds": len(rows),
        "tolerance_mm": 45.0,
        "controls": controls,
        "summary": summary,
        "per_seed": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print()
    for k, v in controls.items():
        print(f"  control {k:<26} {'PASS' if v.get('pass') else 'FAIL'}")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
