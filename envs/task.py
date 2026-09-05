#!/usr/bin/env python3
"""The dinner-table task: the instruction, its sub-goals, and how they score.

This is the evaluation configuration.  It is deliberately separate from any
policy: the same predicates score a scripted rollout, a learned policy, or a
human teleoperating the arms, so a number reported here means the same thing
whoever produced it.

Scoring follows the track brief's first criterion -- sequencing, object
hand-off, coordination, manipulation accuracy and overall task success -- so
each of those is a field in the report rather than a single opaque pass/fail.
"""
from __future__ import annotations

import numpy as np
import mujoco

INSTRUCTION = (
    "Open the top drawer, pick up the fork and the spoon and lay them either "
    "side of the place setting, put the plate on the mat, then set the mug "
    "down to the right of the plate."
)

# id -> (moving body, target site, planar tolerance in metres)
PLACEMENTS = {
    "fork_placed":  ("fork", "target_fork", 0.045),
    "spoon_placed": ("spoon", "target_spoon", 0.045),
    "plate_placed": ("plate", "target_plate", 0.050),
    "mug_placed":   ("mug", "target_mug", 0.050),
}
DRAWER_OPEN_M = 0.060          # how far the drawer must travel to count as open
UPRIGHT_COS = np.cos(np.deg2rad(25.0))
TABLE_TOP_Z = 0.75
DROP_Z = TABLE_TOP_Z - 0.05    # below this an object has left the table

SUBGOAL_ORDER = ["drawer_open", "fork_placed", "spoon_placed",
                 "plate_placed", "mug_placed"]


def _body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _site(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _upright(data, bid) -> bool:
    """True when the body's local +z still points roughly up."""
    return float(data.xmat[bid].reshape(3, 3)[2, 2]) >= UPRIGHT_COS


def gripper_geoms(model, prefix: str) -> set[int]:
    out = set()
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[g])) or ""
        if nm.startswith(prefix) and ("gripper" in nm or "jaw" in nm):
            out.add(g)
    return out


class TaskMonitor:
    """Steps alongside the simulation and records what the rubric asks about."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.jaws = {s: gripper_geoms(model, f"{s}_") for s in ("left", "right")}
        self.obj_geoms = {}
        for nm in ("plate", "mug", "bottle", "spoon", "fork", "drawer"):
            bid = _body(model, nm)
            self.obj_geoms[nm] = {g for g in range(model.ngeom)
                                  if model.geom_bodyid[g] == bid}
        self.drawer_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")]

        self.touched: dict[str, set[str]] = {n: set() for n in self.obj_geoms}
        self.held_by: dict[str, str | None] = {n: None for n in self.obj_geoms}
        self.handoffs: list[dict] = []
        self.first_done: dict[str, float] = {}
        self.dropped: set[str] = set()
        self.max_drawer = 0.0

    # ------------------------------------------------------------------ step
    def step(self, data: mujoco.MjData) -> None:
        self.max_drawer = max(self.max_drawer, float(data.qpos[self.drawer_adr]))

        contact_sides = {n: set() for n in self.obj_geoms}
        for c in range(data.ncon):
            g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
            for obj, geoms in self.obj_geoms.items():
                if g1 in geoms or g2 in geoms:
                    other = g2 if g1 in geoms else g1
                    for side, jaw in self.jaws.items():
                        if other in jaw:
                            contact_sides[obj].add(side)

        for obj, sides in contact_sides.items():
            self.touched[obj] |= sides
            # a hand-off is one arm releasing into the other on the same object
            if len(sides) == 1:
                side = next(iter(sides))
                prev = self.held_by[obj]
                if prev is not None and prev != side:
                    self.handoffs.append(
                        {"object": obj, "from": prev, "to": side,
                         "t": round(float(data.time), 3)})
                self.held_by[obj] = side

        for name in ("plate", "mug", "bottle", "spoon", "fork"):
            if float(data.xpos[_body(self.model, name)][2]) < DROP_Z:
                self.dropped.add(name)

        for gid, ok in self.subgoals(data).items():
            if ok and gid not in self.first_done:
                self.first_done[gid] = round(float(data.time), 3)

    # -------------------------------------------------------------- predicates
    def subgoals(self, data: mujoco.MjData) -> dict[str, bool]:
        out = {"drawer_open": float(data.qpos[self.drawer_adr]) >= DRAWER_OPEN_M}
        for gid, (body, site, tol) in PLACEMENTS.items():
            bid, sid = _body(self.model, body), _site(self.model, site)
            near = float(np.linalg.norm(data.xpos[bid][:2] - data.site_xpos[sid][:2]))
            resting = abs(float(data.xpos[bid][2]) - TABLE_TOP_Z) < 0.09
            upright = _upright(data, bid) if body in ("plate", "mug") else True
            out[gid] = bool(near <= tol and resting and upright and
                            body not in self.dropped)
        return out

    # ------------------------------------------------------------------ report
    def report(self, data: mujoco.MjData) -> dict:
        sg = self.subgoals(data)
        done = [g for g in SUBGOAL_ORDER if sg[g]]
        # sequencing credit: the longest prefix of the intended order achieved
        prefix = 0
        for g in SUBGOAL_ORDER:
            if sg[g]:
                prefix += 1
            else:
                break
        return {
            "instruction": INSTRUCTION,
            "subgoals": sg,
            "subgoals_met": len(done),
            "subgoals_total": len(SUBGOAL_ORDER),
            "in_order_prefix": prefix,
            "task_success": all(sg.values()),
            "handoffs": self.handoffs,
            "handoff_occurred": bool(self.handoffs),
            "arms_that_touched_an_object": sorted(
                {s for o, ss in self.touched.items() if o != "drawer" for s in ss}),
            "bimanual": len({s for o, ss in self.touched.items()
                             if o != "drawer" for s in ss}) == 2,
            "objects_dropped": sorted(self.dropped),
            "max_drawer_travel_m": round(self.max_drawer, 4),
            "first_completion_times_s": self.first_done,
            "sim_time_s": round(float(data.time), 3),
        }
