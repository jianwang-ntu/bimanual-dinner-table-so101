"""Does approaching along the drawer's own axis get the fork out of it?

This is route (b) of ``F-CUTLERY-LIP-001`` as that entry actually worded it,
and it is the last of its three named routes still standing.  Route (a), a
non-zero ``standoff``, is refuted (``H-CUTLERY-STANDOFF-REFUTED-001``: cutlery
0/10 in all 12 variants).  ``H-CUTLERY-DESCENT-REFUTED-001`` is often read as
having closed route (b) as well; it did not.  It swept how the descent is
INTERPOLATED -- 1/2/4/8 solved poses, squared and not -- and left the approach
VECTOR at the pure +z every published figure was measured at.  What it did
establish is the target for this probe: squaring kills the shove, and the wall
the arm then stalls on stops being the fork and becomes ``drawer_back``, 4,069
steps of contact, arriving 10.7 mm above a grasp pose that
``measure_grasp_feasibility.py`` has already teleported the arm onto and found
collision-free and gripping.

``drawer_back`` is a coordinate, not a mystery.  ``drawer_slide`` has
axis="0 -1 0", so the drawer opens toward -y; ``drawer_back`` sits at drawer-
local y=+0.068 and ``drawer_front`` at y=-0.073.  The authored cutlery offsets
are ('fork', 0.018) and ('spoon', -0.032), which puts the FORK 46 mm in front
of the back wall's inner face and 86 mm behind the front -- the fork is the
object at the BACK of the drawer.  Descending onto it on a pure +z line brings
the wrist down inside that 46 mm.  The back and side walls are also 14 mm
LOWER than the front (half-heights 0.018 against 0.028), so leaning the
approach out toward -y trades a shorter wall for a taller one and is not
free either way -- which is why this is swept and not assumed.

The sweep crosses the approach vector's y component with descent squaring:

* ``dy < 0`` leans the approach out over the open drawer front, so the descent
  runs diagonally in +y as it drops and the arm's wrist stays out of the back.
* ``dy > 0`` is the control that must NOT help: it leans the approach further
  into the wall the contact tally already names.

Reported per variant, on the same seeds and the same untouched scorer:

* ``fork_placed`` / ``spoon_placed`` -- the outcome, 20 of the 50 sub-goals
  that have never scored.
* ``stalled_above_target_mm`` -- the residual ``measure_cutlery_block.py`` and
  ``measure_fork_descent.py`` both report, so the three are comparable.
* ``blocking_contacts`` -- WHICH geom the arm is wearing.  If leaning out of
  the back wall works, ``drawer_back`` must fall out of this tally.  A variant
  that placed the fork while still jammed on ``drawer_back`` would mean the
  mechanism claimed here is not the one that acted.

The SPOON is the built-in negative control.  ``F-SPOON-POSE-INFEASIBLE-001``
measured that it has no collision-free grasp pose at all where the scene parks
it -- jaws 10.4-48.4 mm ABOVE the handle top at first clearance -- so no change
to how the arm ARRIVES may place it.  A sweep that placed both would be
measuring something other than what it claims.

The shipped controller (pure +z, unsquared) is a variant of the sweep rather
than a number quoted from an earlier file, and the run asserts it is present.

Writes evidence/cutlery_approach.json.
Run:  python3 scripts/measure_cutlery_approach.py --seeds 10
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
# The shipped settings, named so the baseline is identified by what the
# repository does rather than by its position in the sweep.
SHIPPED_DY, SHIPPED_SQ = 0.0, False
SHIPPED_Z = 0.075


def _is_descent(label: str, body: str) -> bool:
    return label == f"{body}_descend" or label.startswith(f"{body}_descend")


def run(seed: int, dy: float, sq: bool) -> dict:
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

    C.CUTLERY_APPROACH = (0.0, float(dy), SHIPPED_Z)
    C.CUTLERY_DESCEND_SQUARE = bool(sq)
    script = C.dinner_table_script()
    roll = C.Rollout(model, data)

    live = {"label": "", "arm": "", "body": None}
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
            t["aim"] = tgt
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
        "seed": seed, "dy_m": float(dy), "square": bool(sq),
        "shipped": (abs(float(dy) - SHIPPED_DY) < 1e-12
                    and bool(sq) == SHIPPED_SQ),
        "subgoals_met": int(sum(rep["subgoals"].values())),
        "subgoals": rep["subgoals"],
        "sim_time_s": round(float(data.time), 1),
        "objects": out_bodies,
    }


def _job(a):
    seed, dy, sq = a
    try:
        return run(seed, dy, sq)
    except Exception as exc:                       # keep the sweep going
        return {"seed": seed, "dy_m": dy, "square": sq,
                "error": f"{type(exc).__name__}: {exc}"}


def _med(rows, body, key):
    vals = [r["objects"][body].get(key) for r in rows if "objects" in r]
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 1) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dys", default="0.0,-0.040,-0.070")
    ap.add_argument("--square", default="0,1")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default=str(EVID / "cutlery_approach.json"))
    a = ap.parse_args()

    dys = [float(x) for x in a.dys.split(",")]
    sqs = [bool(int(x)) for x in a.square.split(",")]
    if not any(abs(d - SHIPPED_DY) < 1e-12 for d in dys) or SHIPPED_SQ not in sqs:
        raise SystemExit("the shipped setting (dy=0, unsquared) must be in "
                         "the sweep -- it is the baseline every published "
                         "figure was measured at")

    jobs = [(s, d, q) for d in dys for q in sqs for s in range(a.seeds)]
    rows = []
    with futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)

    variants = []
    for d in dys:
        for q in sqs:
            sel = [r for r in rows
                   if abs(r["dy_m"] - d) < 1e-12 and r["square"] == q
                   and "error" not in r]
            if not sel:
                continue
            tally: dict[str, int] = {}
            for r in sel:
                for c in r["objects"]["fork"]["blocking_contacts"]:
                    tally[c["geom"]] = tally.get(c["geom"], 0) + c["steps"]
            variants.append({
                "dy_m": d, "square": q,
                "shipped": abs(d - SHIPPED_DY) < 1e-12 and q == SHIPPED_SQ,
                "n": len(sel),
                "subgoals_met_total": sum(r["subgoals_met"] for r in sel),
                "subgoals_met_mean": round(
                    float(np.mean([r["subgoals_met"] for r in sel])), 2),
                "fork_placed": sum(r["objects"]["fork"]["placed"] for r in sel),
                "spoon_placed": sum(r["objects"]["spoon"]["placed"] for r in sel),
                "fork_stall_mm_median": _med(sel, "fork", "stalled_above_target_mm"),
                "spoon_stall_mm_median": _med(sel, "spoon", "stalled_above_target_mm"),
                "fork_shove_mm_median": _med(sel, "fork", "shove_mm"),
                "fork_bow_mm_median": _med(sel, "fork", "bow_mm"),
                "fork_lifted_mm_median": _med(sel, "fork", "lifted_mm"),
                "fork_blocking_contacts": [
                    {"geom": k, "steps": v} for k, v in
                    sorted(tally.items(), key=lambda kv: -kv[1])[:4]],
            })

    out = {
        "probe": "measure_cutlery_approach.py",
        "question": "Does leaning the cutlery approach out along the drawer's "
                    "own axis (-y, over the open front) clear the arm of "
                    "drawer_back and let the fork be grasped?",
        "hypothesis_id": "H-CUTLERY-APPROACH-001",
        "defect": "F-CUTLERY-LIP-001 route (b), the last unrefuted route",
        "negative_control": "spoon -- F-SPOON-POSE-INFEASIBLE-001 says no "
                            "collision-free grasp pose exists for it, so it "
                            "must not be placed by any approach change",
        "positive_control_direction": "dy>0 leans further into drawer_back and "
                                      "must not help",
        "seeds": a.seeds,
        "approach_z_m": SHIPPED_Z,
        "shipped_variant": {"dy_m": SHIPPED_DY, "square": SHIPPED_SQ},
        "variants": sorted(variants, key=lambda v: (-v["fork_placed"],
                                                    v["fork_stall_mm_median"]
                                                    if v["fork_stall_mm_median"]
                                                    is not None else 9e9)),
        "runs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    for v in out["variants"]:
        print("dy=%+7.3f sq=%d  fork=%2d/%d spoon=%2d/%d  subgoals=%3d  "
              "stall=%s  wall=%s%s"
              % (v["dy_m"], v["square"], v["fork_placed"], v["n"],
                 v["spoon_placed"], v["n"], v["subgoals_met_total"],
                 v["fork_stall_mm_median"],
                 v["fork_blocking_contacts"][0]["geom"]
                 if v["fork_blocking_contacts"] else "-",
                 "   <-- SHIPPED" if v["shipped"] else ""))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
