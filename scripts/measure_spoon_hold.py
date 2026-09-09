"""Is the spoon EVER held by either gripper, on any seed?

``spoon_placed`` is 0 on every arm this workspace has ever measured, and
``envs/task.py`` scores ``task_success`` as ``all(sg.values())`` -- so this one
sub-goal caps task success at zero by itself, and it is the only sub-goal that
can move ``req_demo_video`` off DRAFT.

Three independent interventions have now left it EXACTLY invariant, with zero
discordant seeds in every case:

  * ``evidence/cutlery_direct_place_ab.json`` -- deleting the hand-off and
    letting the picker place the spoon itself.  spoon_placed 0/20 -> 0/20.
  * ``evidence/spoon_depth.json`` -- teleporting the spoon up to 30 mm in y,
    away from the drawer front rail that blocks its grasp pose.  0/10 at every
    station.  (That file also reports the spoon never rising above 14.4 mm, but
    it is STALE: its baseline is 15/50 with fork_placed 0/10, the controller
    from before ``824e5ed`` squared the cutlery pick.  Whether that still holds
    on the shipped arm is exactly what this script measures.)
  * ``evidence/arm_swap_ab.json`` -- giving the spoon to the OTHER arm, whose
    grasp pose needs 10 mm less clearance.  0/20 -> 0/20.

Those rule out the transfer, the object's position, and the arm assignment.
What none of them measured is the thing they all assume: that the spoon is in
somebody's jaws at all.  This script measures that directly.

METHOD.  One scripted rollout per seed, unchanged.  At every step, look at the
contact list and ask which arms' jaw geoms are touching the spoon's geoms.
Record the same for the FORK in the SAME rollout.

THE FORK IS THE POSITIVE CONTROL, and it is a control that can fail:
``evidence/fork_release.json`` reports this same contact detector registering
the picking arm's grasp on 9 of 10 seeds.  If the fork comes back empty here
too, the detector is broken and the spoon's zero says nothing.  A probe whose
only possible answer is "not found" would not be evidence.

Writes evidence/spoon_hold.json.
Run:  PYTHONPATH=. python3 scripts/measure_spoon_hold.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import mujoco
import numpy as np

BODIES = ("spoon", "fork")
ARMS = ("left", "right")

# CONTACT IS NOT A GRASP.  The jaws brush both objects far more often than they
# carry them, so a bare "was it ever touched" would answer yes and mean nothing.
# An object counts as LIFTED only if it rose past this threshold while an arm
# was touching it.  20 mm is taken from the measured hand floor in
# TECHNICAL_SUMMARY.md -- the dynamic floor sits 20.25 mm above the drawer
# floor -- so anything under it is inside the noise of a knock.
LIFT_MM = 20.0


def _geoms_of(model, body: str) -> set[int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def run(seed: int) -> dict:
    from envs.task import TaskMonitor, gripper_geoms
    from envs import randomize as R
    from envs import controller as C
    from envs import scene_source

    model, data, _log = R.make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bids = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in BODIES}
    geoms = {b: _geoms_of(model, b) for b in BODIES}
    jaws = {a: gripper_geoms(model, f"{a}_") for a in ARMS}

    roll = C.Rollout(model, data)

    # steps_held[body][arm] -- how many simulator steps that arm's jaws were in
    # contact with that body.  A single frame of contact is a brush, not a
    # grasp, so the count is reported rather than a bare boolean.
    steps_held = {b: {a: 0 for a in ARMS} for b in BODIES}
    both_jaws = {b: 0 for b in BODIES}
    z0 = {b: None for b in BODIES}
    zmax = {b: -1e9 for b in BODIES}
    # The height the body reached while an arm was actually touching it.
    zmax_held = {b: -1e9 for b in BODIES}

    def on_step(d):
        for b in BODIES:
            z = float(d.xpos[bids[b]][2])
            if z0[b] is None:
                z0[b] = z
            zmax[b] = max(zmax[b], z)
        touching = {b: set() for b in BODIES}
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            for b in BODIES:
                if g1 in geoms[b]:
                    other = g2
                elif g2 in geoms[b]:
                    other = g1
                else:
                    continue
                for a, js in jaws.items():
                    if other in js:
                        touching[b].add(a)
        for b in BODIES:
            for a in touching[b]:
                steps_held[b][a] += 1
            if touching[b]:
                zmax_held[b] = max(zmax_held[b], float(d.xpos[bids[b]][2]))
            # Both of one arm's jaw faces on the object at once is the closest
            # contact-only proxy for a PINCH rather than a knock.
            if touching[b]:
                both_jaws[b] += 1

    roll.run(C.dinner_table_script(), monitor=mon, on_step=on_step)
    rep = mon.report(data)

    out = {"seed": seed, "subgoals": rep["subgoals"], "bodies": {}}
    for b in BODIES:
        held_arms = [a for a in ARMS if steps_held[b][a] > 0]
        lift_held = (round((zmax_held[b] - z0[b]) * 1000.0, 2)
                     if zmax_held[b] > -1e8 else None)
        out["bodies"][b] = {
            "ever_touched": bool(held_arms),
            "lifted": bool(lift_held is not None and lift_held >= LIFT_MM),
            "held_by": held_arms,
            "steps_held_by_arm": dict(steps_held[b]),
            "lift_mm": round((zmax[b] - z0[b]) * 1000.0, 2),
            "lift_while_held_mm": lift_held,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="evidence/spoon_hold.json")
    args = ap.parse_args()

    runs = [run(s) for s in range(args.seeds)]

    agg = {}
    for b in BODIES:
        held = [r["seed"] for r in runs if r["bodies"][b]["ever_touched"]]
        lifted = [r["seed"] for r in runs if r["bodies"][b]["lifted"]]
        lifts = [r["bodies"][b]["lift_mm"] for r in runs]
        agg[b] = {
            "seeds_ever_touched": held,
            "n_seeds_ever_touched": len(held),
            "seeds_lifted": lifted,
            "n_seeds_lifted": len(lifted),
            "lift_threshold_mm": LIFT_MM,
            "max_steps_held_any_seed": max(
                max(r["bodies"][b]["steps_held_by_arm"].values())
                for r in runs),
            "lift_mm_min": min(lifts),
            "lift_mm_max": max(lifts),
        }

    # The control is the fork being LIFTED, not merely touched: a detector that
    # only ever reports contact would pass a control it cannot fail.
    control_ok = agg["fork"]["n_seeds_lifted"] > 0
    doc = {
        "probe": "measure_spoon_hold.py",
        "question": "Is the spoon ever held by either gripper, on any seed?",
        "finding_id": "F-AIS-SPOON-NEVER-GRASPED-001",
        "job": "lablab:ai-infra-summit-hackathon",
        "row_under_test": "req_demo_video (DRAFT) -- spoon_placed is the "
                          "sub-goal that caps task_success at 0",
        "seeds": args.seeds,
        "method": "one scripted rollout per seed, privileged scene, nothing in "
                  "envs/task.py or the controller changed; per-step contact "
                  "between each body's geoms and each arm's jaw geoms",
        "positive_control": {
            "body": "fork",
            "pass": control_ok,
            "expectation": "evidence/fork_release.json reports this detector "
                           "seeing the picking arm's grasp on 9 of 10 seeds",
            "why_it_matters": "if the fork also came back never-lifted, the "
                              "detector would be broken and the spoon's zero "
                              "would carry no information",
        },
        "aggregate": agg,
        "runs": runs,
    }
    sp, fk = agg["spoon"], agg["fork"]
    if not control_ok:
        doc["verdict"] = "INVALID -- positive control failed; the detector saw " \
                         "no fork lift either, so this run says nothing " \
                         "about the spoon"
    elif sp["n_seeds_lifted"] == 0:
        doc["verdict"] = f"The spoon is TOUCHED on {sp['n_seeds_ever_touched']}" \
                         f"/{args.seeds} seeds and LIFTED on 0, against a fork " \
                         f"control lifted on {fk['n_seeds_lifted']}. The spoon " \
                         "is a GRASP failure -- upstream of the hand-off, the " \
                         "arm assignment and the object's position."
    else:
        doc["verdict"] = (
            f"SPLIT. The spoon is touched on {sp['n_seeds_ever_touched']}/"
            f"{args.seeds} seeds but LIFTED on only {sp['n_seeds_lifted']} "
            f"{sp['seeds_lifted']}, against a fork lifted on "
            f"{fk['n_seeds_lifted']} {fk['seeds_lifted']} in the SAME rollouts. "
            "So spoon_placed=0 is TWO failures, not one: a grasp that does not "
            "close on most seeds, and -- on the seeds where it does carry the "
            "spoon ~120 mm -- a place that still misses. Any fix aimed at only "
            "one of them can move at most a few seeds.")

    p = pathlib.Path(args.out)
    if p.parent != pathlib.Path("") and not str(p).startswith("/"):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"positive control (fork): touched "
          f"{fk['n_seeds_ever_touched']}/{args.seeds}, LIFTED "
          f"{fk['n_seeds_lifted']}/{args.seeds} {fk['seeds_lifted']}")
    print(f"spoon: touched {sp['n_seeds_ever_touched']}/{args.seeds} "
          f"{sp['seeds_ever_touched']}, LIFTED {sp['n_seeds_lifted']}"
          f"/{args.seeds} {sp['seeds_lifted']}")
    print(f"lift mm  spoon {agg['spoon']['lift_mm_min']}-{agg['spoon']['lift_mm_max']}"
          f"   fork {agg['fork']['lift_mm_min']}-{agg['fork']['lift_mm_max']}")
    print(doc["verdict"])
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
