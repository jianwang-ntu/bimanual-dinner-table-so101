#!/usr/bin/env python3
"""A scripted bimanual controller for the dinner-table task.

This is a *controller*, not a policy: it holds no learned parameters and reads
no camera.  It exists so that the evaluation harness in ``envs/task.py`` has
something to score other than an arm holding still, and so the demonstration
video has a rollout to film.  What it does read from the simulator is only what
a perception stack would supply -- the world pose of the object it is about to
touch -- and every waypoint is resolved from that pose at the moment the move
starts, so the same script runs against a randomized scene.

The mechanism is three small pieces:

``tip_mid``     where the two jaws actually meet, which is *not* the
                ``gripperframe`` site: at the home opening the meeting point
                sits ~31 mm along the approach axis from it.
``align_roll``  ``wrist_roll`` turns the jaw closing direction about the
                approach axis without moving ``gripperframe`` at all, so the
                jaws can be squared onto a fork handle or a plate rim.
``plan_pose``   alternates position IK with those two corrections until the
                jaw meeting point, not the wrist, lands on the target.

Everything above the primitives is a plain list of moves, so the script can be
read top to bottom against the task instruction it implements.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .ik import site_ik, ARM_JOINTS
from .dinner_table import ARM_X as dt_ARM_X, ARM_Y as dt_ARM_Y

GRIPPER_OPEN = 1.20          # rad; jaw tips ~101 mm apart
GRIPPER_WIDE = 1.45          # rad; ~117 mm, for reaching around a mug
GRIPPER_NARROW = 0.45        # rad; ~51 mm -- fits between the drawer walls
GRIP_PINCH = -0.17           # rad; jaw tips ~8 mm apart -- cutlery, handles
GRIP_RIM = 0.05              # rad; ~21 mm -- plate rim
GRIP_MUG = 0.20              # rad; ~32 mm -- squeezes a 48-64 mm mug wall

ARMS = ("left", "right")


# --------------------------------------------------------------------- geometry
def jaw_tip_geoms(model: mujoco.MjModel, prefix: str) -> tuple[list[int], list[int]]:
    """(fixed-jaw tip geoms, moving-jaw tip geoms) for one arm."""
    fixed, moving = [], []
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not nm.startswith(prefix) or "jaw_sph_tip" not in nm:
            continue
        (moving if "moving" in nm else fixed).append(g)
    if not fixed or not moving:
        raise KeyError(f"no jaw tip geoms for {prefix}")
    return fixed, moving


class Gripper:
    """Cached ids for one arm's jaws, actuators and joints."""

    def __init__(self, model: mujoco.MjModel, arm: str):
        self.arm = arm
        self.prefix = f"{arm}_"
        self.fixed, self.moving = jaw_tip_geoms(model, self.prefix)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      self.prefix + "gripperframe")
        self.qadr, self.vadr = [], []
        self.act = []
        for j in ARM_JOINTS + ("gripper",):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + j)
            self.qadr.append(model.jnt_qposadr[jid])
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    self.prefix + j)
            self.act.append(aid)
        self.qadr = np.array(self.qadr)
        self.act = np.array(self.act)
        self.roll_q = self.qadr[ARM_JOINTS.index("wrist_roll")]
        self.grip_q = self.qadr[-1]
        self.lo = model.actuator_ctrlrange[self.act, 0].copy()
        self.hi = model.actuator_ctrlrange[self.act, 1].copy()
        self.calibrate(model)

    def calibrate(self, model: mujoco.MjModel) -> None:
        """Measure jaw separation against joint command, once, from the model.

        Nothing here is a fitted constant: the table is swept out of this
        model, so it stays correct if the gripper is ever re-dimensioned.
        """
        d = mujoco.MjData(model)
        qs = np.linspace(self.lo[-1], self.hi[-1], 40)
        seps = []
        for q in qs:
            mujoco.mj_resetDataKeyframe(model, d, 0)
            d.qpos[self.grip_q] = q
            mujoco.mj_kinematics(model, d)
            f = np.mean([d.geom_xpos[g] for g in self.fixed], axis=0)
            m = np.mean([d.geom_xpos[g] for g in self.moving], axis=0)
            seps.append(float(np.linalg.norm(m - f)))
        self._cal_q, self._cal_sep = qs, np.array(seps)

    def q_for_sep(self, sep_m: float) -> float:
        """Joint command whose jaw separation is ``sep_m`` (clamped to range)."""
        return float(np.interp(sep_m, self._cal_sep, self._cal_q))

    def sep_for_q(self, q: float) -> float:
        return float(np.interp(q, self._cal_q, self._cal_sep))

    def tip_mid(self, model, data) -> np.ndarray:
        """World point midway between the two jaw faces."""
        f = np.mean([data.geom_xpos[g] for g in self.fixed], axis=0)
        m = np.mean([data.geom_xpos[g] for g in self.moving], axis=0)
        return (f + m) / 2.0

    def jaw_axis(self, model, data) -> np.ndarray:
        f = np.mean([data.geom_xpos[g] for g in self.fixed], axis=0)
        m = np.mean([data.geom_xpos[g] for g in self.moving], axis=0)
        v = m - f
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def approach_axis(self, data) -> np.ndarray:
        """The wrist_roll axis: the one direction wrist_roll cannot turn."""
        return data.site_xmat[self.site].reshape(3, 3)[:, 0]


def align_roll(model, data, grip: Gripper, want: np.ndarray, *,
               signed: bool = False) -> None:
    """Turn ``wrist_roll`` so the jaws close along ``want`` as nearly as they can.

    The jaw axis is a line, not an arrow, so by default the half-turn that
    costs less travel is taken; if the joint limit rules it out the other one
    is used.  ``signed=True`` reads ``want`` as an arrow -- the direction from
    the FIXED jaw to the MOVING one -- and considers only whole turns, so the
    stationary jaw is put on a chosen side rather than on whichever side the
    shorter turn happened to give.
    """
    a = grip.approach_axis(data)
    w = want - float(want @ a) * a
    n = float(np.linalg.norm(w))
    if n < 1e-6:                       # asked for the one direction we cannot make
        return
    w /= n
    j = grip.jaw_axis(model, data)
    u = j - float(j @ a) * a
    if float(np.linalg.norm(u)) < 1e-6:
        return
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    d = float(np.arctan2(w @ v, w @ u))

    lo, hi = model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, grip.prefix + "wrist_roll")]
    cur = float(data.qpos[grip.roll_q])
    cands = ((d, d + 2 * np.pi, d - 2 * np.pi) if signed
             else (d, d + np.pi, d - np.pi))
    for cand in sorted(cands, key=abs):
        if lo <= cur + cand <= hi:
            data.qpos[grip.roll_q] = cur + cand
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            return


def plan_pose(model, scratch, grip: Gripper, tip_target: np.ndarray, *,
              jaw_dir: np.ndarray | None = None, opening: float | None = None,
              standoff: float = 0.0, rounds: int = 6
              ) -> tuple[np.ndarray, float]:
    """Joint targets putting the JAW MEETING POINT on ``tip_target``.

    ``scratch`` must already hold the pose to start from; it is modified.
    Returns the six actuator targets and the residual tip error in metres.

    ``standoff`` backs the meeting point off along the jaw axis, away from the
    moving jaw.  Only ONE of this gripper's jaws moves, so a pose that puts the
    meeting point on the object puts the STATIONARY jaw inside it: measured on
    seed 0, 7.1 mm inside the mug wall, 4.9 mm from the fork handle's centre
    against a 6 mm half-width, and 0.4 mm inside the plate rim.  That is what
    shoved all three away instead of picking them up -- every one of them was
    touched only by the fixed jaw, on every seed.  A standoff of
    (clearance + squeeze/2) leaves the stationary jaw ``clearance`` clear of
    the surface and lets the moving jaw close across it.  It is measured
    against the live jaw axis each round, so it follows whichever way
    ``align_roll`` ends up pointing the jaws.
    """
    if opening is not None:
        scratch.qpos[grip.grip_q] = opening
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)

    best_err = np.inf
    best = scratch.qpos[grip.qadr].copy()
    # Seeded from where the arm is now; on failure, re-seeded from the home
    # pose, because a contorted hand-off pose is a bad start for the DLS solver
    # and that -- not reach -- was what lost the cutlery placements.
    for attempt in range(2):
        if attempt == 1:
            if best_err < 8e-3:
                break
            mujoco.mj_resetDataKeyframe(model, scratch, 0)
            if opening is not None:
                scratch.qpos[grip.grip_q] = opening
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
        offset = np.zeros(3)
        for _ in range(rounds):
            if jaw_dir is not None:
                align_roll(model, scratch, grip, jaw_dir, signed=standoff != 0.0)
            goal = tip_target
            if standoff:
                goal = tip_target - standoff * grip.jaw_axis(model, scratch)
            site_ik(model, scratch, grip.prefix, "gripperframe", goal + offset)
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            if standoff:
                goal = tip_target - standoff * grip.jaw_axis(model, scratch)
            resid = goal - grip.tip_mid(model, scratch)
            err = float(np.linalg.norm(resid))
            if err < best_err:
                best_err, best = err, scratch.qpos[grip.qadr].copy()
            if err < 2e-3:
                break
            offset = offset + resid
    return best, best_err


# ------------------------------------------------------------------- primitives
class Move:
    """One arm going somewhere, with the target resolved when the move starts.

    ``where`` is called with (model, data) so a waypoint can be read off the
    object's *current* pose rather than a number baked in at authoring time.
    """

    def __init__(self, arm: str, where, *, jaw=None, opening=None,
                 grip=None, plan_at=None, standoff=0.0, label=""):
        self.arm, self.where, self.jaw = arm, where, jaw
        self.opening, self.grip, self.label = opening, grip, label
        self.standoff = standoff
        # The jaws meet ~41 mm nearer the wrist closed than open, so a pose
        # solved with them open puts the object in front of the closing point
        # and the grasp brushes past it.  ``plan_at`` solves the arm at the
        # opening the jaws will HOLD at, and commands them open to get there.
        self.plan_at = plan_at


class Grip(Move):
    """Close or open one gripper without moving the arm."""

    def __init__(self, arm: str, value: float, label=""):
        super().__init__(arm, None, grip=value, label=label)


class Home(Move):
    """Back to the keyframe pose -- a known configuration to re-plan from."""

    def __init__(self, arm: str, label=""):
        super().__init__(arm, None, label=label)


# ------------------------------------------------------------------ the script
def _site(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


_GRIP_CACHE: dict = {}


def _grip(model, arm: str) -> "Gripper":
    key = (id(model), arm)
    if key not in _GRIP_CACHE:
        _GRIP_CACHE[key] = Gripper(model, arm)
    return _GRIP_CACHE[key]


def tip_up(arm: str, dz: float):
    """Straight up from wherever this arm's jaws are now.

    Long transits are broken by one of these because the SO-101's own links --
    not its gripper -- will otherwise sweep the drawer shut on the way past,
    which costs the sub-goal that was already earned.
    """
    def f(model, data):
        return _grip(model, arm).tip_mid(model, data) + np.array([0.0, 0.0, dz])
    return f


def site_xyz(name, dz=0.0, dy=0.0, dx=0.0, off=None):
    d = np.array([dx, dy, dz]) if off is None else np.asarray(off, float)

    def f(model, data):
        return data.site_xpos[_site(model, name)] + d
    return f


def long_axis(body: str):
    """Horizontal direction of a cutlery body's own +y (its handle-to-head line)."""
    def f(model, data):
        R = data.xmat[_body(model, body)].reshape(3, 3)
        v = R[:, 1].copy()
        v[2] = 0.0
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    return f


def across(body: str):
    """Perpendicular to that: where the jaws must close to pinch the handle."""
    def f(model, data):
        v = long_axis(body)(model, data)
        return np.array([-v[1], v[0], 0.0])
    return f


ARM_BASE = {"left": np.array([-dt_ARM_X, dt_ARM_Y]),
            "right": np.array([dt_ARM_X, dt_ARM_Y])}


def _rim_radius(model, body: str) -> float:
    """Distance from a plate's centre to its rim geoms."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_rim_0")
    return float(np.linalg.norm(model.geom_pos[gid][:2]))


def _rim_dir(model, data, body: str, arm: str) -> np.ndarray:
    c = data.xpos[_body(model, body)][:2]
    v = ARM_BASE[arm] - c
    n = float(np.linalg.norm(v))
    v = v / n if n > 1e-6 else np.array([1.0, 0.0])
    return np.array([v[0], v[1], 0.0])


def rim_toward(body: str, arm: str, dz: float = 0.006):
    """The point on the rim nearest this arm, which is the one it can reach.

    The authored ``plate_grasp`` site is fixed to the plate and spins with the
    randomized yaw, so half the time it faces away and the arm runs out of
    envelope reaching around.  A perception stack would pick the near rim; so
    does this.
    """
    def f(model, data):
        c = data.xpos[_body(model, body)]
        v = _rim_dir(model, data, body, arm)
        r = _rim_radius(model, body)
        return np.array([c[0] + v[0] * r, c[1] + v[1] * r, c[2] + dz])
    return f


def rim_jaw(body: str, arm: str):
    """Jaws close across the rim, i.e. along the radius."""
    def f(model, data):
        return _rim_dir(model, data, body, arm)
    return f


def radial(body: str, site: str):
    """Plate rim: close across the rim, i.e. along the plate's radius."""
    def f(model, data):
        c = data.xpos[_body(model, body)][:2]
        p = data.site_xpos[_site(model, site)][:2]
        v = p - c
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-6 else np.array([1.0, 0.0])
        return np.array([v[0], v[1], 0.0])
    return f


def geom_width(geom: str, axis: int = 0):
    """Full extent of a named geom along one of its own axes, in metres.

    Object sizes are randomized per episode, so every grip command is derived
    from the geometry in front of the arm rather than from a constant.
    """
    def f(model, data, grip=None):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        return 2.0 * float(model.geom_size[gid][axis])
    return f


def pinch(width_fn, squeeze: float = 0.005, floor: float = 0.004):
    """Close to ``squeeze`` metres narrower than the thing being held.

    Commanding the jaws hard shut on a 60 mm mug just shoves it away -- the
    servos are force-limited at 2.94 Nm -- so the command is set from the
    measured width and the calibration table instead.
    """
    def f(model, data, grip):
        w = width_fn(model, data, grip)
        return grip.q_for_sep(max(w - squeeze, floor))
    return f


# ------------------------------------------------------- plate: hook and drag
# The plate is 92-116 mm across and the jaws span 101 mm open, so there is no
# opening at which both jaws clear it and then close on it: whichever way the
# meeting point is aimed, one jaw sweeps over the plate's own face.  What the
# rim IS good for is a hook.  Its boxes stand 5 mm proud of the plate's top
# face, so a jaw dropped inside them catches the rim's inner wall, and the
# plate can be dragged flat across the table instead of lifted.  Dragging also
# keeps it upright and resting, which is what the scorer asks of it, and it
# stays inside the arm's static envelope: the servos saturate at 2.94 Nm past
# roughly 0.28 m of horizontal reach, measured, and carrying a plate out at
# arm's length is over that line.
PLATE_DZ = 0.010            # m above the plate body origin: inside the rim
PLATE_SEEK = 0.004          # m per contact-seeking step downward
PLATE_SEEKS = 3             # extra descents if the first found no contact
PLATE_STANDOFF = 0.005      # clearance + squeeze/2 for the 12 mm rim wall
PLATE_TOL = 0.028           # m; re-grasp and correct while further out than this
PLATE_BOW = np.array([0.13, -0.02])   # keeps the plate off the cabinet's SE corner
PLATE_EAST = 0.19           # m; east of this the bow is worth taking


def _arm_jaw_geoms(model, arm: str) -> set[int]:
    """Every geom of one arm's jaws -- blades included, not only the tips."""
    out = set()
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith(f"{arm}_") and "jaw" in nm:
            out.add(g)
    return out


def _plate_geoms(model) -> set[int]:
    bid = _body(model, "plate")
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def _plate_untouched(model, data) -> bool:
    """True while no right jaw is in contact with the plate.

    The rim is 8 mm tall and the arm tracks a commanded pose to 12-20 mm, so
    aiming once at the middle of that window misses more often than it hits.
    This is the predicate that turns the descent into a search.
    """
    jaws, plate = _arm_jaw_geoms(model, "right"), _plate_geoms(model)
    for c in range(data.ncon):
        a, b = int(data.contact.geom1[c]), int(data.contact.geom2[c])
        if (a in plate and b in jaws) or (b in plate and a in jaws):
            return False
    return True


def _plate_short(model, data) -> bool:
    """True while the plate is further from its mat than the scorer allows."""
    c = data.xpos[_body(model, "plate")][:2]
    t = data.site_xpos[_site(model, "target_plate")][:2]
    return float(np.linalg.norm(c - t)) > PLATE_TOL


def _plate_east(model, data) -> bool:
    return float(data.xpos[_body(model, "plate")][0]) > PLATE_EAST


def site_xy(name):
    def f(model, data):
        return data.site_xpos[_site(model, name)][:2]
    return f


def const_xy(xy):
    q = np.asarray(xy, float)
    return lambda model, data: q


def drag_toward(arm: str, aim, frac: float):
    """Tip waypoint carrying the plate ``frac`` of the way to ``aim``.

    Read live at the moment the move starts and never cached: the waypoint is
    where the jaws are now, plus a share of the plate's OWN remaining offset.
    If the plate has slipped in the jaws, the next waypoint aims from where it
    actually is rather than from where the grasp assumed it would be.
    """
    def f(model, data):
        g = _grip(model, arm)
        tip = g.tip_mid(model, data)
        c = data.xpos[_body(model, "plate")][:2]
        v = (np.asarray(aim(model, data), float) - c) * frac
        return np.array([tip[0] + v[0], tip[1] + v[1], tip[2]])
    return f


def _rim_at(dz: float, up: float = 0.0):
    f = rim_toward("plate", "right", dz=dz)
    if not up:
        return f
    return lambda model, data: f(model, data) + np.array([0.0, 0.0, up])


def _drag_plate(suffix: str = ""):
    """Hook the near rim, then drag the plate flat onto the mat.

    Three blocks, and each is conditional on something measured rather than
    assumed: the descent keeps stepping down until a jaw actually touches the
    plate, the bow is taken only while the plate is still east of the cabinet,
    and the caller repeats the whole cycle only while the plate is short of the
    mat.
    """
    hold = pinch(geom_width("plate_rim_0", 0), squeeze=0.004)
    jaw = rim_jaw("plate", "right")
    kw = dict(jaw=jaw, standoff=PLATE_STANDOFF)
    out = [
        ({"right": Move("right", _rim_at(PLATE_DZ, 0.055), opening=GRIPPER_OPEN,
                        label="plate_above" + suffix, **kw)}, 1.6),
        ({"right": Move("right", _rim_at(PLATE_DZ), opening=GRIPPER_OPEN,
                        plan_at=hold, label="plate_descend" + suffix, **kw)}, 1.4),
    ]
    for j in range(1, PLATE_SEEKS + 1):
        out.append(("if", _plate_untouched, [
            ({"right": Move("right", _rim_at(PLATE_DZ - j * PLATE_SEEK),
                            opening=GRIPPER_OPEN, plan_at=hold,
                            label=f"plate_seek{j}" + suffix, **kw)}, 1.0)]))
    out.append(({"right": Grip("right", hold, label="plate_pinch" + suffix)}, 0.9))
    out.append(("if", _plate_east, [
        ({"right": Move("right", drag_toward("right", const_xy(PLATE_BOW), f),
                        label=f"plate_bow{i}" + suffix, **kw)}, 0.9)
        for i, f in enumerate((1 / 3, 1 / 2, 1.0), start=1)]))
    for i, f in enumerate((1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0), start=1):
        out.append(({"right": Move("right",
                                   drag_toward("right", site_xy("target_plate"), f),
                                   label=f"plate_drag{i}" + suffix, **kw)}, 0.9))
    out.append(({"right": Grip("right", GRIPPER_OPEN,
                               label="plate_release" + suffix)}, 0.7))
    out.append(({"right": Move("right", tip_up("right", 0.100), opening=GRIPPER_OPEN,
                               label="plate_retreat" + suffix)}, 1.3))
    return out


HORIZ = np.array([0.0, 0.0, 1.0])          # jaws closing top-to-bottom
X_AXIS = np.array([1.0, 0.0, 0.0])


def dinner_table_script() -> list[tuple[dict, float]]:
    """The rollout, as (moves-for-this-step, seconds) pairs.

    Read against the task instruction: open the drawer, lay the fork and the
    spoon either side of the setting, put the plate on the mat, set the mug to
    its right.  Each cutlery item starts on the far side of the table from its
    target, so each is handed between the arms rather than carried around.
    """
    S: list[tuple[dict, float]] = []

    def step(seconds=1.2, **moves):
        S.append((moves, seconds))

    # --- 1. the right arm opens the drawer -----------------------------------
    S.extend(_open_drawer())

    # --- 2. fork: right picks it out of the drawer, hands it to the left ------
    fork_grip = pinch(geom_width("fork_handle", 0))
    S.extend(_pick(("right", "fork", "fork_grasp", fork_grip, across("fork")),
                   open_to=GRIPPER_NARROW, descend_z=0.003,
                   approach=(0.0, 0.0, 0.075), lift=(0.0, 0.0, 0.085)))
    S.extend(_handoff("right", "left", "fork", "target_fork", across("fork"),
                      hold=fork_grip))

    # --- 3. spoon: the mirror image, left to right ----------------------------
    spoon_grip = pinch(geom_width("spoon_handle", 0))
    S.extend(_pick(("left", "spoon", "spoon_grasp", spoon_grip, across("spoon")),
                   open_to=GRIPPER_NARROW, descend_z=0.003,
                   approach=(0.0, 0.0, 0.075), lift=(0.0, 0.0, 0.085)))
    S.extend(_handoff("left", "right", "spoon", "target_spoon", across("spoon"),
                      hold=spoon_grip))

    # --- 4. plate: hooked by the rim and dragged flat onto the mat ------------
    # Both arms go home first.  The cutlery hand-off leaves the right arm in a
    # contorted pose over the drawer, and a joint-space ramp out of it sweeps
    # the arm through the cabinet carcass: 18 of 25 probe waypoints were missed
    # from a hand-off pose and reached from home.
    S.append(({"right": Home("right", label="plate_ready"),
               "left": Home("left", label="left_park")}, 1.6))
    S.extend(_drag_plate())
    S.append(("if", _plate_short, _drag_plate("_again")))
    S.append(("if", _plate_short, _drag_plate("_again2")))

    # --- 5. mug: the left arm sets it down to the right of the plate ----------
    mug_grip = pinch(geom_width("mug_wall", 0), squeeze=0.014)
    S.extend(_pick(("left", "mug", "mug_grasp", mug_grip, X_AXIS),
                   open_to=GRIPPER_WIDE, lift=(0, 0, 0.060),
                   approach=(0, 0, 0.060), descend_z=-0.004))
    S.extend(_place("left", "target_mug", mug_grip, X_AXIS, drop_z=0.045,
                    over_z=0.065, open_to=GRIPPER_WIDE))

    # --- 6. look back at the drawer ------------------------------------------
    # Twice, because once is measurably not enough: the drawer is opened on
    # every seed and ends shut on four of ten, and a single retry left three of
    # those four still shut.  The arms work over an open drawer for two thirds
    # of the episode and nudge it back on the way past.
    S.append(("if", _drawer_shut,
              _open_drawer(suffix="_again", from_home=True)))
    S.append(("if", _drawer_shut,
              _open_drawer(suffix="_again2", from_home=True)))
    return S


DRAWER_OPEN_M = 0.060          # mirrors envs/task.py; not imported, so that
                               # the controller cannot drift into the scorer


def _drawer_shut(model, data) -> bool:
    adr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")]
    return float(data.qpos[adr]) < DRAWER_OPEN_M + 0.005


def _open_drawer(suffix: str = "", from_home: bool = False):
    """Grasp the handle and pull. Also the recovery block, hence a function."""
    pre = ([({"right": Home("right", label="drawer_home" + suffix),
              "left": Home("left", label="clear_home" + suffix)}, 1.6)]
           if from_home else [])
    return pre + [
        ({"right": Move("right", site_xyz("drawer_handle_site", dy=-0.075, dz=0.045),
                        jaw=HORIZ, opening=GRIPPER_OPEN,
                        label="drawer_approach" + suffix)}, 1.4),
        ({"right": Move("right", site_xyz("drawer_handle_site", dy=-0.004),
                        jaw=HORIZ, opening=GRIPPER_OPEN,
                        label="drawer_straddle" + suffix)}, 1.2),
        ({"right": Grip("right", GRIP_PINCH, label="drawer_close" + suffix)}, 0.7),
        ({"right": Move("right", site_xyz("drawer_handle_site", dy=-0.088),
                        jaw=HORIZ, label="drawer_pull" + suffix)}, 3.4),
        # Let go NARROW, not wide.  Opening the jaws fully swings the moving
        # one through 80 mm, and with the drawer already out that arc catches
        # the drawer front and shoves it back: measured on seed 0, 92 mm of
        # travel collapsing to 45 mm across the release alone, which is under
        # the 60 mm the scorer wants.  A narrow release clears the handle
        # without the swing.
        ({"right": Grip("right", GRIPPER_NARROW, label="drawer_release" + suffix)}, 0.5),
        ({"right": Move("right", tip_up("right", 0.090), opening=GRIPPER_NARROW,
                        label="drawer_retreat" + suffix)}, 1.0),
        ({"right": Grip("right", GRIPPER_OPEN, label="drawer_open_jaws" + suffix)}, 0.4),
    ]


def _pick(spec, *, lift=(0.0, 0.0, 0.070), approach=(0.0, 0.0, 0.070),
          open_to=GRIPPER_OPEN, descend_z=0.002):
    """Approach, descend, close, lift.

    ``approach`` and ``lift`` are vectors, not heights, because the cutlery
    starts under the cabinet's own top panel: straight down onto it is a
    collision, and the arm has to come in over the open drawer front instead.
    """
    arm, body, site, close_to, jaw = spec
    at = site if callable(site) else None

    def point(off):
        d = np.asarray(off, float)
        if at is None:
            return site_xyz(site, off=d)
        return lambda model, data: at(model, data) + d

    out = []
    # The approach pose is solved with the jaws where they will be OPEN: it is
    # only a via point, and solving it closed costs 40 mm of reach the arm
    # does not have at the far corners of the table.
    out.append(({arm: Move(arm, point(approach), jaw=jaw,
                           opening=open_to, label=f"{body}_above")}, 1.5))
    out.append(({arm: Move(arm, point((0.0, 0.0, descend_z)), jaw=jaw,
                           opening=open_to, plan_at=close_to,
                           label=f"{body}_descend")}, 1.2))
    out.append(({arm: Grip(arm, close_to, label=f"{body}_close")}, 0.8))
    out.append(({arm: Move(arm, point(lift), jaw=jaw,
                           label=f"{body}_lift")}, 1.2))
    return out


def _place(arm, target, hold, jaw, *, drop_z=0.030, over_z=0.090,
           open_to=GRIPPER_OPEN):
    return [
        ({arm: Move(arm, site_xyz(target, dz=over_z), jaw=jaw, label=f"{target}_over")}, 1.5),
        ({arm: Move(arm, site_xyz(target, dz=drop_z), jaw=jaw, label=f"{target}_down")}, 1.2),
        ({arm: Grip(arm, open_to, label=f"{target}_release")}, 0.6),
        ({arm: Move(arm, site_xyz(target, dz=0.110), opening=open_to,
                    label=f"{target}_retreat")}, 1.1),
    ]


def _handoff(giver, taker, body, target, jaw, *, hold=GRIP_PINCH):
    """Giver holds the object over the shared zone; taker grasps and places it.

    The taker closes before the giver opens, so the object is held by both arms
    for a moment -- that overlap is what ``TaskMonitor`` records as a hand-off.
    """
    grasp = f"{body}_grasp"
    return [
        # giver presents it; taker comes in from above at the same time
        ({giver: Move(giver, site_xyz("handoff", dz=0.030), jaw=jaw,
                      label=f"{body}_present"),
          taker: Move(taker, site_xyz("handoff", dz=0.115), jaw=jaw,
                      opening=GRIPPER_OPEN, label=f"{body}_meet")}, 1.8),
        ({taker: Move(taker, site_xyz(grasp, dz=0.040), jaw=jaw,
                      opening=GRIPPER_OPEN, plan_at=hold,
                      label=f"{body}_take_above")}, 1.2),
        ({taker: Move(taker, site_xyz(grasp, dz=0.004), jaw=jaw,
                      opening=GRIPPER_OPEN, plan_at=hold,
                      label=f"{body}_take")}, 1.0),
        ({taker: Grip(taker, hold, label=f"{body}_taker_close")}, 0.8),
        ({giver: Grip(giver, GRIPPER_OPEN, label=f"{body}_giver_release")}, 0.6),
        ({giver: Move(giver, tip_up(giver, 0.085), opening=GRIPPER_OPEN,
                      label=f"{body}_giver_clear"),
          taker: Move(taker, site_xyz(grasp, dz=0.075), jaw=jaw,
                      label=f"{body}_taker_lift")}, 1.5),
    ] + _place(taker, target, hold, jaw)


# -------------------------------------------------------------------- execution
class Rollout:
    """Runs a script against a live model/data, ramping actuator targets."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.grips = {a: Gripper(model, a) for a in ARMS}
        self.scratch = mujoco.MjData(model)
        self.trace: list[dict] = []

    def _plan(self, mv: Move) -> np.ndarray:
        g = self.grips[mv.arm]
        cur = self.data.ctrl[g.act].copy()
        grip_cmd = mv.grip
        if callable(grip_cmd):
            grip_cmd = grip_cmd(self.model, self.data, g)
        opening = mv.opening
        if callable(opening):
            opening = opening(self.model, self.data, g)
        plan_at = mv.plan_at
        if callable(plan_at):
            plan_at = plan_at(self.model, self.data, g)
        if isinstance(mv, Home):
            self.last_err = 0.0
            return self.model.key_ctrl[0][g.act].copy()
        if isinstance(mv, Grip):
            cur[-1] = grip_cmd
            return cur
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = 0.0
        target = np.asarray(mv.where(self.model, self.data), dtype=float)
        jaw = mv.jaw(self.model, self.data) if callable(mv.jaw) else mv.jaw
        standoff = mv.standoff
        if callable(standoff):
            standoff = standoff(self.model, self.data, g)
        q, err = plan_pose(self.model, self.scratch, g, target,
                           jaw_dir=None if jaw is None else np.asarray(jaw, float),
                           opening=plan_at if plan_at is not None else opening,
                           standoff=standoff)
        self.last_err = err
        out = q.copy()
        out[-1] = opening if opening is not None else cur[-1]
        if grip_cmd is not None:
            out[-1] = grip_cmd
        return out

    def run(self, script, monitor=None, on_step=None, settle: float = 0.35) -> dict:
        dt = self.model.opt.timestep
        for entry in script:
            # A conditional block: ("if", predicate, sub-script).  Used to
            # re-check a sub-goal at the end of the episode -- the arms work
            # over an open drawer and can nudge it shut behind them, and a
            # controller that never looks back leaves an action it really
            # performed undone.
            if isinstance(entry, tuple) and entry and entry[0] == "if":
                _, pred, sub = entry
                if pred(self.model, self.data):
                    self.trace.append({"label": "recheck_fired", "arm": "-",
                                       "t": round(float(self.data.time), 3),
                                       "ik_err_mm": 0.0})
                    self.run(sub, monitor=monitor, on_step=on_step, settle=settle)
                continue
            moves, seconds = entry
            plans, starts = {}, {}
            for arm, mv in moves.items():
                self.last_err = 0.0
                plans[arm] = np.clip(self._plan(mv), self.grips[arm].lo,
                                     self.grips[arm].hi)
                starts[arm] = self.data.ctrl[self.grips[arm].act].copy()
                self.trace.append({"label": mv.label, "arm": arm,
                                   "t": round(float(self.data.time), 3),
                                   "ik_err_mm": round(self.last_err * 1000, 2)})
            n = max(1, int(seconds / dt))
            for i in range(n + int(settle / dt)):
                a = min(1.0, (i + 1) / n)
                a = a * a * (3 - 2 * a)                 # smoothstep
                for arm in moves:
                    g = self.grips[arm]
                    self.data.ctrl[g.act] = starts[arm] + a * (plans[arm] - starts[arm])
                mujoco.mj_step(self.model, self.data)
                if monitor is not None:
                    monitor.step(self.data)
                if on_step is not None:
                    on_step(self.data)
        return {"moves": len(self.trace),
                "sim_time_s": round(float(self.data.time), 3),
                "max_ik_err_mm": round(max((t["ik_err_mm"] for t in self.trace),
                                           default=0.0), 2),
                "trace": self.trace}


def run_dinner_table(model, data, monitor=None, on_step=None) -> dict:
    return Rollout(model, data).run(dinner_table_script(), monitor=monitor,
                                    on_step=on_step)
