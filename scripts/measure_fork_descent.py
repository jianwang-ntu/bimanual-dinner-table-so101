"""Does a straddle-preserving descent get the fork out of the drawer?

``scripts/measure_grasp_feasibility.py`` closed the pose question for the fork
and left one open: the grasp pose it asks for is reachable and it grips --
6.0 mm above the fork on 3/3 seeds, jaws 3.3-3.9 mm below the 5.0 mm handle
top -- while the rollout stalls 21-24 mm above the target in sustained contact
with ``fork_handle`` itself and shoves the fork 73 mm across the drawer.  A
pose that is known good and an arm that ends up wearing the object means the
fault is on the way there, not at the end of it.

``Rollout.run`` ramps in JOINT space between two solved poses.  ``_pick``
solves exactly two for the descent -- 75 mm above the fork and 3 mm above it --
so the tip traces whatever curve joint-space interpolation produces over a
72 mm drop.  Nothing asks that curve to be vertical, and the jaws are open
around a 12 mm handle at the bottom of it.

This sweep crosses the number of solved poses in the descent with where the
descent ends, and measures the MECHANISM as well as the outcome:

* ``bow_mm`` -- the largest perpendicular distance of the jaw meeting point
  from the straight Cartesian line between where the descent starts and the
  point it is aiming at.  This is the quantity the hypothesis is about.  If
  more solved poses do not straighten the path, the story is wrong however the
  placements land.
* ``shove_mm`` -- how far the fork moves in the plane during the descent
  itself, isolated from the rest of the episode.
* ``stalled_above_target_mm`` -- as ``measure_cutlery_block.py`` measured it.

The spoon is swept by the same knob and is the built-in negative control:
``measure_grasp_feasibility`` measured that NO collision-free grasp pose for it
exists (jaws 10.4-48.4 mm ABOVE the handle top at first clearance), so a
trajectory fix must not place it.  A sweep that placed both would be measuring
something other than what it claims.

The shipped controller (1 solved pose, 3 mm) is a variant of the sweep rather
than a number quoted from an earlier file, and every variant runs the same
seeds and the same untouched scorer in one process pool.

Writes evidence/fork_descent.json.
Run:  python3 scripts/measure_fork_descent.py --seeds 10
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

BODIES = ("fork", "spoon")
# The shipped settings.  Named so the baseline is identified by what the
# repository does, not by its position in the sweep.
SHIPPED_STEPS, SHIPPED_Z, SHIPPED_SQ = 1, 0.003, False


def _is_descent(label: str, body: str) -> bool:
    return label == f"{body}_descend" or label.startswith(f"{body}_descend")


def run(seed: int, steps: int, descend_z: float, sq: bool) -> dict:
    from envs.randomize import make_env
    from envs.task import TaskMonitor, PLACEMENTS
    from envs import controller as C
    from envs import scene_source

    model, data, _ = make_env(seed)
    mon = TaskMonitor(model)
    scene_source.install(scene_source.make("privileged"))

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    start = {o: data.xpos[bid(o)].copy() for o in BODIES}

    C.CUTLERY_DESCEND_STEPS = int(steps)
    C.CUTLERY_DESCEND_Z = float(descend_z)
    C.CUTLERY_DESCEND_SQUARE = bool(sq)
    script = C.dinner_table_script()
    roll = C.Rollout(model, data)

    n_solved = {b: 0 for b in BODIES}
    for entry in script:
        if isinstance(entry, tuple) and entry and entry[0] == "if":
            continue
        for mv in entry[0].values():
            for b in BODIES:
                if _is_descent(getattr(mv, "label", ""), b):
                    n_solved[b] += 1

    live = {"label": "", "arm": "", "body": None}
    # Per body: where the descent began (tip xy), what the last descent
    # waypoint aims at, the tip samples along the way, and the fork's own
    # position when the descent opened.
    track: dict[str, dict] = {}
    asked_z: dict[str, float] = {}
    reached_z: dict[str, float] = {}
    contacts: dict[str, dict[str, int]] = {}
    peak = {o: 0.0 for o in BODIES}
    orig_plan = roll._plan

    def plan(mv):
        lb = getattr(mv, "label", "")
        live["label"], live["arm"] = lb, mv.arm
        live["body"] = next((b for b in BODIES if _is_descent(lb, b)), None)
        b = live["body"]
        if b is not None and mv.where is not None:
            tgt = np.asarray(mv.where(model, data), float)
            t = track.setdefault(b, {"samples": [], "obj0": None, "tip0": None})
            if t["tip0"] is None:
                t["tip0"] = roll.grips[mv.arm].tip_mid(model, data).copy()
                t["obj0"] = data.xpos[bid(b)].copy()
            t["aim"] = tgt                     # last one written wins
            if lb == f"{b}_descend":
                asked_z[b] = float(tgt[2])
        return orig_plan(mv)

    roll._plan = plan

    def on_step(d):
        for o in BODIES:
            peak[o] = max(peak[o], float(d.xpos[bid(o)][2] - start[o][2]))
        b = live["body"]
        if b is None:
            return
        g = roll.grips[live["arm"]]
        tip = g.tip_mid(model, d)
        track[b]["samples"].append(tip.copy())
        reached_z[b] = min(reached_z.get(b, 9.9), float(tip[2]))
        track[b]["obj_end"] = d.xpos[bid(b)].copy()
        mine = {gg for gg in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[gg])) or ""
                    ).startswith(f"{live['arm']}_")}
        tally = contacts.setdefault(b, {})
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if (g1 in mine) != (g2 in mine):
                other = g2 if g1 in mine else g1
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       other) or f"geom{other}"
                tally[nm] = tally.get(nm, 0) + 1

    roll.run(script, monitor=mon, on_step=on_step)
    rep = mon.report(data)

    out_bodies = {}
    for b in BODIES:
        t = track.get(b)
        end = data.xpos[bid(b)]
        tgt = data.site_xpos[sid(PLACEMENTS[f"{b}_placed"][1])]
        rec = {
            "solved_descent_poses": n_solved[b],
            "lifted_mm": round(peak[b] * 1000, 1),
            "planar_move_mm": round(
                float(np.linalg.norm(end[:2] - start[b][:2])) * 1000, 1),
            "final_to_target_mm": round(
                float(np.linalg.norm(end[:2] - tgt[:2])) * 1000, 1),
            "placed": bool(rep["subgoals"][f"{b}_placed"]),
        }
        if t and t["samples"] and t.get("aim") is not None:
            p0, p1 = np.asarray(t["tip0"], float), np.asarray(t["aim"], float)
            v = p1 - p0
            n = float(np.linalg.norm(v))
            bow = 0.0
            if n > 1e-9:
                u = v / n
                for s in t["samples"]:
                    w = np.asarray(s, float) - p0
                    bow = max(bow, float(np.linalg.norm(w - (w @ u) * u)))
            rec["bow_mm"] = round(bow * 1000, 1)
            rec["descent_span_mm"] = round(n * 1000, 1)
            rec["shove_mm"] = round(float(np.linalg.norm(
                np.asarray(t["obj_end"], float)[:2]
                - np.asarray(t["obj0"], float)[:2])) * 1000, 1)
        if b in asked_z and b in reached_z:
            rec["asked_tip_z_m"] = round(asked_z[b], 4)
            rec["lowest_tip_z_m"] = round(reached_z[b], 4)
            rec["stalled_above_target_mm"] = round(
                (reached_z[b] - asked_z[b]) * 1000, 1)
        top = sorted(contacts.get(b, {}).items(), key=lambda kv: -kv[1])[:3]
        rec["blocking_contacts"] = [{"geom": nm, "steps": s} for nm, s in top]
        out_bodies[b] = rec

    return {
        "seed": seed, "steps": int(steps), "descend_z_m": float(descend_z),
        "square": bool(sq),
        "shipped": (int(steps) == SHIPPED_STEPS
                    and abs(float(descend_z) - SHIPPED_Z) < 1e-12
                    and bool(sq) == SHIPPED_SQ),
        "subgoals_met": int(sum(rep["subgoals"].values())),
        "subgoals": rep["subgoals"],
        "sim_time_s": round(float(data.time), 1),
        "objects": out_bodies,
    }


def _job(a):
    seed, steps, dz, sq = a
    try:
        return run(seed, steps, dz, sq)
    except Exception as exc:                       # keep the sweep going
        return {"seed": seed, "steps": steps, "descend_z_m": dz, "square": sq,
                "error": f"{type(exc).__name__}: {exc}"}


def _agg(key, rows):
    vals = [r["objects"][key[0]].get(key[1]) for r in rows]
    vals = [v for v in vals if v is not None]
    return (round(float(np.median(vals)), 1) if vals else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", default="1,2,4,8")
    ap.add_argument("--descend-z", default="0.003,0.006")
    ap.add_argument("--square", default="0,1")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=str(EVID / "fork_descent.json"))
    a = ap.parse_args()

    steps = [int(x) for x in a.steps.split(",")]
    zs = [float(x) for x in a.descend_z.split(",")]
    sqs = [bool(int(x)) for x in a.square.split(",")]
    if (SHIPPED_STEPS, SHIPPED_Z, SHIPPED_SQ) not in [
            (k, z, q) for k in steps for z in zs for q in sqs]:
        raise SystemExit("the shipped setting (1 pose, 3 mm, unsquared) must "
                         "be in the sweep")
    jobs = [(s, k, z, q) for k in steps for z in zs for q in sqs
            for s in range(a.seeds)]

    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(_job, jobs))

    variants = []
    for k in steps:
      for z in zs:
        for q in sqs:
            rs = [r for r in runs if r["steps"] == k
                  and r["descend_z_m"] == z and r["square"] == q]
            ok = [r for r in rs if "error" not in r]
            v = {
                "steps": k, "descend_z_m": z, "square": q, "seeds": len(ok),
                "shipped": bool(ok and ok[0]["shipped"]),
                "subgoals_total": sum(r["subgoals_met"] for r in ok),
                "subgoals_of": 5 * len(ok),
                "errors": [r["error"] for r in rs if "error" in r],
            }
            for b in BODIES:
                v[f"{b}_placed"] = sum(int(r["objects"][b]["placed"])
                                       for r in ok)
                for m in ("bow_mm", "shove_mm", "stalled_above_target_mm",
                          "lifted_mm"):
                    v[f"{b}_{m}_median"] = _agg((b, m), ok)
            variants.append(v)

    base = next(v for v in variants if v["shipped"])
    best = max(variants, key=lambda v: (v["fork_placed"], v["subgoals_total"],
                                        -(v["fork_bow_mm_median"] or 1e9)))
    out = {
        "question": "does splitting the cutlery descent into solved poses "
                    "keep the jaws around the handle, and does that place "
                    "the fork?",
        "method": "sweep solved-pose count x descent end height on one seed "
                  "stream and one untouched scorer; the shipped controller "
                  "(1 pose, 3 mm) is a swept variant, not a quoted number",
        "negative_control": "the same knob drives the spoon, for which "
                            "measure_grasp_feasibility.py found no "
                            "collision-free grasp pose exists; a trajectory "
                            "fix must not place it",
        "seeds": a.seeds,
        "shipped_variant": {"steps": SHIPPED_STEPS, "descend_z_m": SHIPPED_Z,
                            "square": SHIPPED_SQ},
        "baseline": base, "best": best,
        "verdict": ("PLACES THE FORK" if best["fork_placed"] > base["fork_placed"]
                    else "NO PLACEMENT GAINED"),
        "variants": variants,
        "runs": runs,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    hdr = (f"{'steps':>5} {'end_mm':>7} {'sq':>3} {'fork':>5} {'spoon':>6} "
           f"{'bow_mm':>7} {'shove':>6} {'stall':>6} {'subg':>5}")
    print(hdr)
    for v in variants:
        print(f"{v['steps']:>5} {v['descend_z_m']*1000:>7.1f} "
              f"{int(v['square']):>3} "
              f"{v['fork_placed']:>5} {v['spoon_placed']:>6} "
              f"{str(v['fork_bow_mm_median']):>7} "
              f"{str(v['fork_shove_mm_median']):>6} "
              f"{str(v['fork_stalled_above_target_mm_median']):>6} "
              f"{v['subgoals_total']:>5}"
              + ("   <- shipped" if v["shipped"] else ""))
    print(f"\nverdict: {out['verdict']}   -> {a.out}")


if __name__ == "__main__":
    main()
