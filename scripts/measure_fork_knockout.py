#!/usr/bin/env python3
"""What takes the fork out of the giver's jaws -- and is it the taker's ARM?

F-FORK-HANDOFF-001 established that the placing arm never holds the fork: on
every seed the fork is resting on the table or in the drawer by the time the
taker lifts.  It also recorded, against itself, that its refutation of a
taker/giver collision was PARTIAL:

    "the taker's LINKS are not instrumented, so a link collision is not
     excluded."

That is this probe.  ``envs.task.gripper_geoms`` matches only bodies whose name
contains "gripper" or "jaw" -- 23 of each arm's 48 geoms.  The base, shoulder,
upper arm, lower arm, wrist and camera mount (25 geoms per arm) have never been
watched touching anything.  Here EVERY geom of EVERY body is watched, attributed
by body name, at EVERY simulator step, and the question asked is narrow:

    at the step the fork leaves the giver's jaws, what else is touching it?

Reported per seed:

  loss_step        the last step at which a GIVER jaw geom touched the fork,
                   after the fork has been lifted clear of its start height.
  touching_at_loss the body names in contact with the fork at that step and in
                   the WINDOW_STEPS steps that follow it -- the candidates for
                   what knocked it out.
  first_taker_contact  the first step at which any geom of the TAKER arm
                   touches the fork, and which body it was.
  verdict          KNOCKED_OUT_BY_TAKER when a taker body is in contact inside
                   the window; SLIPPED when nothing but the giver is;
                   NEVER_LIFTED when the giver never carried it at all.

Controls, each able to fail and each reported as it comes out:

  reproduces_published    the per-seed final_to_target_mm column must reproduce
                          the ten numbers evidence/fork_release.json publishes
                          for the shipped controller, pinned as a literal.  A
                          probe that has drifted off the shipped configuration
                          is measuring a different robot.
  link_geoms_are_new      the all-geom watch set must be a STRICT superset of
                          the jaw-only set used by every previous probe, and the
                          difference must be non-empty -- otherwise "the links
                          are now instrumented" is a false claim.
  detector_can_say_yes    the same all-geom detector must report the GIVER
                          holding the fork during its own pick.  A contact test
                          that never fires proves nothing by not firing.
  unrelated_body_absent   NEGATIVE CONTROL.  ``bottle`` is never handled by the
                          script and is randomized away from the hand-off site;
                          it must NOT appear in any fork contact.  If it does,
                          the attribution is matching bodies it should not and
                          every name below is suspect.
  loss_is_after_lift      the loss step must come AFTER the fork first rises
                          clear of its start height, on every seed where the
                          giver lifts it.  A "loss" found before the lift is the
                          pick, not the hand-off.

Writes evidence/fork_knockout.json.
Run:  python3 scripts/measure_fork_knockout.py --seeds 10
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
GIVER = "right"          # _pick(("right", "fork", ...)) in dinner_table_script
TAKER = "left"           # _handoff("right", "left", "fork", ...)

# How far EITHER SIDE of the loss step to keep listing contacts.  0.05 s at the
# model's 0.002 s timestep; long enough for a sweeping link to register, short
# enough that the table the fork lands on is not swept up as a cause.
#
# It is symmetric because the first version of this probe looked FORWARD only,
# and a link that pushes the fork out does so while the giver is still holding
# it -- i.e. BEFORE the loss step, where a forward-only window cannot see it.
WINDOW_STEPS = 25

# A lift of more than this above the fork's own start height means the giver
# really carried it, as opposed to nudging it inside the drawer.
LIFT_MM = 15.0

# evidence/fork_release.json -> summary.final_to_target_mm, shipped controller.
PUBLISHED_FINAL_MM = [158.9, 149.4, 14.0, 54.6, 70.2, 83.0, 159.8, 35.9,
                      143.6, 20.3]
REPRO_TOL_MM = 1.0


def _mm(v) -> float:
    return round(float(v) * 1000.0, 1)


def run(seed: int) -> dict:
    from envs.task import TaskMonitor, gripper_geoms
    from envs import randomize as R
    from envs import controller as C
    from envs import scene_source

    model, data, _log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET)
    fork_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}

    # EVERY geom, attributed by the body it belongs to.  This is the whole
    # point of the probe: the previous instrument saw 23 geoms per arm.
    geom_body = {g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(model.geom_bodyid[g])) or f"body{g}")
                 for g in range(model.ngeom)}
    jaw_only = {a: gripper_geoms(model, f"{a}_") for a in (GIVER, TAKER)}
    all_arm = {a: {g for g, nm in geom_body.items() if nm.startswith(f"{a}_")}
               for a in (GIVER, TAKER)}

    roll = C.Rollout(model, data)
    grips = {a: C._grip(model, a) for a in (GIVER, TAKER)}

    def contacts(d) -> set[str]:
        """Body names currently touching the fork.  No name filter at all."""
        out = set()
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if g1 in fork_geoms:
                other = g2
            elif g2 in fork_geoms:
                other = g1
            else:
                continue
            if other in fork_geoms:
                continue
            out.add(geom_body[other])
        return out

    _f6 = np.zeros(6)

    def grip_force(d, want: set[int]) -> float:
        """Summed contact normal force between ``want`` geoms and the fork."""
        tot = 0.0
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if not ((g1 in fork_geoms and g2 in want)
                    or (g2 in fork_geoms and g1 in want)):
                continue
            mujoco.mj_contactForce(model, d, c, _f6)
            tot += abs(float(_f6[0]))
        return tot

    def touching_geoms(d, want: set[int]) -> set[int]:
        out = set()
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if g1 in fork_geoms and g2 in want:
                out.add(g2)
            elif g2 in fork_geoms and g1 in want:
                out.add(g1)
        return out

    steps: list[dict] = []
    state = {"z0": None}

    def on_step(d):
        n = len(roll.trace)
        if state["z0"] is None:
            state["z0"] = float(d.xpos[bid][2])
        steps.append({
            "i": len(steps),
            "t": round(float(d.time), 3),
            "trace_n": n,
            "bodies": sorted(contacts(d)),
            "giver_jaw": bool(touching_geoms(d, jaw_only[GIVER])),
            "taker_jaw": bool(touching_geoms(d, jaw_only[TAKER])),
            "giver_any": sorted({geom_body[g]
                                 for g in touching_geoms(d, all_arm[GIVER])}),
            "taker_any": sorted({geom_body[g]
                                 for g in touching_geoms(d, all_arm[TAKER])}),
            "fork_z_mm": _mm(d.xpos[bid][2]),
            "giver_force_N": grip_force(d, jaw_only[GIVER]),
            "giver_sep_mm": _mm(np.linalg.norm(
                np.mean([d.geom_xpos[g] for g in grips[GIVER].moving], axis=0)
                - np.mean([d.geom_xpos[g] for g in grips[GIVER].fixed], axis=0))),
            "giver_tip_mm": _mm(np.linalg.norm(
                d.xpos[bid] - grips[GIVER].tip_mid(model, d))),
            "taker_tip_mm": _mm(np.linalg.norm(
                d.xpos[bid] - grips[TAKER].tip_mid(model, d))),
        })

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    z0_mm = _mm(state["z0"])
    # index -> label(s) of the script entry in flight at that step
    labels_at = {}
    for k, tr in enumerate(roll.trace):
        labels_at.setdefault(tr["t"], []).append(f"{tr['label']}[{tr['arm']}]")

    lifted = [s for s in steps if s["fork_z_mm"] - z0_mm > LIFT_MM]
    first_lift = lifted[0]["i"] if lifted else None

    held = [s for s in steps if s["giver_jaw"]]
    # The loss step ends the CARRY: the last step of the CONTIGUOUS run of
    # giver-jaw contact that is in progress at the first lift.  The first
    # version of this probe took the last giver-jaw contact anywhere in the
    # episode, which on seeds 3 and 7 returned t=83.7 s and t=220.4 s -- the
    # giver brushing a fork that had been lying on the table for a minute,
    # during the plate and mug phases.  That is not the hand-off.
    loss = None
    if first_lift is not None:
        idx = {s["i"] for s in held}
        j = first_lift
        # walk back to the start of the run in progress at the lift, then
        # forward to its end, tolerating single-step contact dropout
        while j - 1 in idx or j - 2 in idx:
            j -= 1 if j - 1 in idx else 2
        run_start = j
        j = first_lift if first_lift in idx else run_start
        while j + 1 in idx or j + 2 in idx:
            j += 1 if j + 1 in idx else 2
        loss = steps[j] if j in idx else None
        carry_run = (run_start, j)
    else:
        carry_run = None

    win: list[dict] = []
    if loss is not None:
        win = [s for s in steps
               if loss["i"] - WINDOW_STEPS <= s["i"] <= loss["i"] + WINDOW_STEPS]

    taker_first = next((s for s in steps if s["taker_any"]), None)

    touching_at_loss = sorted({b for s in win for b in s["bodies"]})
    taker_in_window = sorted({b for s in win for b in s["taker_any"]})
    if first_lift is None:
        verdict = "NEVER_LIFTED"
    elif taker_in_window:
        verdict = "KNOCKED_OUT_BY_TAKER"
    else:
        verdict = "SLIPPED"

    # Over the carry: was the fork ever actually LOADED by the giver's jaws?
    # A position-controlled gripper commanded to the object's own width holds
    # it with ~0 N and drops it on any acceleration; that is a different defect
    # from a link sweeping it out, and only force separates them.
    carry_force, carry_sep = None, None
    if carry_run is not None:
        lo, hi = carry_run
        fs = [s["giver_force_N"] for s in steps[lo:hi + 1]]
        ss = [s["giver_sep_mm"] for s in steps[lo:hi + 1]]
        carry_force = {"min": round(min(fs), 3), "max": round(max(fs), 3),
                       "at_loss": round(steps[hi]["giver_force_N"], 3),
                       "last_20_steps": [round(v, 3) for v in fs[-20:]]}
        carry_sep = {"min": round(min(ss), 2), "max": round(max(ss), 2),
                     "at_loss": round(steps[hi]["giver_sep_mm"], 2)}

    fork = data.xpos[bid]
    tgt = data.site_xpos[sid]
    return {
        "seed": seed,
        "final_to_target_mm": _mm(np.linalg.norm(fork[:2] - tgt[:2])),
        "subgoals_met": int(rep["subgoals_met"]),
        "fork_start_z_mm": z0_mm,
        "first_lift_step": first_lift,
        "first_lift_t": lifted[0]["t"] if lifted else None,
        "max_fork_z_mm": max(s["fork_z_mm"] for s in steps),
        "carry_run_steps": carry_run,
        "loss_step": None if loss is None else loss["i"],
        "loss_t": None if loss is None else loss["t"],
        "loss_fork_z_mm": None if loss is None else loss["fork_z_mm"],
        "loss_labels": [] if loss is None else labels_at.get(loss["t"], []),
        "touching_at_loss": touching_at_loss,
        "taker_bodies_in_window": taker_in_window,
        "taker_jaw_in_window": any(s["taker_jaw"] for s in win),
        "first_taker_contact_step": None if taker_first is None else taker_first["i"],
        "first_taker_contact_t": None if taker_first is None else taker_first["t"],
        "first_taker_contact_bodies": [] if taker_first is None else taker_first["taker_any"],
        "verdict": verdict,
        "giver_jaw_steps": len(held),
        "min_giver_tip_mm": min(s["giver_tip_mm"] for s in steps),
        "min_taker_tip_mm": min(s["taker_tip_mm"] for s in steps),
        "carry_grip_force_N": carry_force,
        "carry_jaw_sep_mm": carry_sep,
        "all_bodies_ever_touching_fork": sorted({b for s in steps for b in s["bodies"]}),
        "window": [{k: s[k] for k in ("i", "t", "bodies", "fork_z_mm",
                                      "giver_force_N", "giver_sep_mm",
                                      "giver_tip_mm", "taker_tip_mm")}
                   for s in win],
        "n_jaw_geoms": {a: len(jaw_only[a]) for a in jaw_only},
        "n_arm_geoms": {a: len(all_arm[a]) for a in all_arm},
        "link_geoms": {a: sorted({geom_body[g] for g in all_arm[a] - jaw_only[a]})
                       for a in all_arm},
    }


def _job(seed: int) -> dict:
    try:
        return run(seed)
    except Exception as exc:
        import traceback
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(EVID / "fork_knockout.json"))
    a = ap.parse_args()

    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, range(a.seeds)):
            rows.append(r)
            if "error" in r:
                print(f"  seed={r['seed']:<2} ERROR {r['error']}")
            else:
                print(f"  seed={r['seed']:<2} {r['verdict']:<22} "
                      f"loss_t={r['loss_t']} z={r['loss_fork_z_mm']} "
                      f"taker_in_window={r['taker_bodies_in_window']}")
    rows.sort(key=lambda r: r["seed"])
    ok = [r for r in rows if "error" not in r]

    got = [r["final_to_target_mm"] for r in ok]
    deltas = [round(abs(g - p), 2)
              for g, p in zip(got, PUBLISHED_FINAL_MM[:len(got)])]
    repro = {
        "pass": len(ok) == len(PUBLISHED_FINAL_MM) and all(
            d <= REPRO_TOL_MM for d in deltas),
        "pinned_literal": PUBLISHED_FINAL_MM,
        "measured": got,
        "abs_delta_mm": deltas,
        "tol_mm": REPRO_TOL_MM,
    }

    link = ok[0]["link_geoms"] if ok else {}
    njaw = ok[0]["n_jaw_geoms"] if ok else {}
    narm = ok[0]["n_arm_geoms"] if ok else {}
    link_new = {
        "pass": bool(ok) and all(narm[a] > njaw[a] and link[a] for a in narm),
        "jaw_geoms_per_arm": njaw,
        "all_arm_geoms_per_arm": narm,
        "bodies_never_watched_before": link,
        "why_it_matters": "if this fails, the probe watches exactly what the "
                          "previous instrument watched and adds nothing",
    }

    # The first version of this control demanded a giver-jaw contact on EVERY
    # seed and failed on seed 6, where the giver's jaws never come within
    # 77 mm of the fork -- a seed with no contact to detect.  Weakening it to
    # "fires somewhere" would be resolving the failure in our own favour, so
    # the requirement is instead SPLIT and the second half is new:
    #   (a) the detector must fire, on real steps, in the shipped rollout; and
    #   (b) every seed where it does NOT fire must be corroborated, by geometry
    #       that uses no contact data at all, as a seed where the giver never
    #       got near enough to hold anything.
    NEAR_MM = 20.0
    silent = [r for r in ok if r["giver_jaw_steps"] == 0]
    uncorroborated = [r["seed"] for r in silent
                      if r["min_giver_tip_mm"] <= NEAR_MM]
    yes = {
        "pass": any(r["giver_jaw_steps"] > 0 for r in ok) and not uncorroborated,
        "detector_fires_on_seeds": [r["seed"] for r in ok
                                    if r["giver_jaw_steps"] > 0],
        "giver_jaw_contact_steps": {r["seed"]: r["giver_jaw_steps"] for r in ok},
        "silent_seeds": [r["seed"] for r in silent],
        "silent_seed_min_giver_tip_mm": {r["seed"]: r["min_giver_tip_mm"]
                                         for r in silent},
        "corroboration_threshold_mm": NEAR_MM,
        "seeds_silent_AND_within_reach": uncorroborated,
        "why_it_matters": "a contact test that never returns True cannot "
                          "support any 'was not touching' conclusion; and a "
                          "seed where it stays silent must be shown, without "
                          "contact data, to be a seed with nothing to detect",
    }

    bottle = sorted({b for r in ok for b in r["all_bodies_ever_touching_fork"]
                     if b == "bottle"})
    neg = {
        "pass": not bottle,
        "unrelated_bodies_seen": bottle,
        "all_bodies_seen": sorted({b for r in ok
                                   for b in r["all_bodies_ever_touching_fork"]}),
        "why_it_matters": "the bottle is never handled by the script; if the "
                          "attribution names it, the names are not trustworthy",
    }

    bad = [r["seed"] for r in ok
           if r["loss_step"] is not None and r["first_lift_step"] is not None
           and r["loss_step"] < r["first_lift_step"]]
    order = {
        "pass": not bad,
        "seeds_with_loss_before_lift": bad,
        "why_it_matters": "a loss found before the lift is the pick failing, "
                          "not the hand-off",
    }

    controls = {"reproduces_published": repro, "link_geoms_are_new": link_new,
                "detector_can_say_yes": yes, "unrelated_body_absent": neg,
                "loss_is_after_lift": order}

    from collections import Counter
    verdicts = Counter(r["verdict"] for r in ok)
    out = {
        "probe": "measure_fork_knockout.py",
        "question": "at the step the fork leaves the giver's jaws, what else "
                    "is touching it -- and is it the taker's arm?",
        "shipped_controller": True,
        "knobs_touched": [],
        "seeds": a.seeds,
        "window_steps": WINDOW_STEPS,
        "controls": controls,
        "summary": {
            "verdicts": dict(verdicts),
            "seeds_where_giver_lifted": [r["seed"] for r in ok
                                         if r["first_lift_step"] is not None],
            "taker_bodies_in_window": {r["seed"]: r["taker_bodies_in_window"]
                                       for r in ok},
            "touching_at_loss": {r["seed"]: r["touching_at_loss"] for r in ok},
            "loss_labels": {r["seed"]: r["loss_labels"] for r in ok},
            "loss_fork_z_mm": {r["seed"]: r["loss_fork_z_mm"] for r in ok},
            "first_taker_contact_bodies": {r["seed"]: r["first_taker_contact_bodies"]
                                           for r in ok},
            "carry_grip_force_N": {r["seed"]: r["carry_grip_force_N"] for r in ok},
            "carry_jaw_sep_mm": {r["seed"]: r["carry_jaw_sep_mm"] for r in ok},
            "min_taker_tip_mm": {r["seed"]: r["min_taker_tip_mm"] for r in ok},
        },
        "per_seed": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print("\ncontrols:")
    for k, v in controls.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    print("verdicts:", dict(verdicts))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
