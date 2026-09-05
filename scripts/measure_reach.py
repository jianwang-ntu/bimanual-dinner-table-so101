"""How far can one SO-101 arm actually put its jaws?

Written to test a hypothesis that turned out to be FALSE, and kept so no later
tick re-runs it.  ``envs/controller.py`` records that "the servos saturate at
2.94 Nm past roughly 0.28 m of horizontal reach", and the fork and the spoon
sit 0.387 m and 0.347 m from the nearest arm base on every seed.  If the first
statement were a reach ceiling, the two sub-goals that have never scored --
20 of the 50 -- would be unreachable by construction and ``eval_seeds.py``'s
``all_reachable`` flag, which tests against a 0.40 m geometric envelope, would
be false on all ten seeds.

It is not.  The arm puts its jaws at 0.417 m of planar radius when asked for
0.400, with three joints at the torque limit the whole way out; saturation
costs accuracy (17-64 mm of tip error across the sweep) but does not stop the
travel.  The cutlery is inside the envelope.  Whatever loses those two
sub-goals, it is not gross reach.

Writes evidence/reach_envelope.json.  Run: python3 scripts/measure_reach.py
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, mujoco
from envs import dinner_table as dt
from envs.randomize import make_env, ARM_BASES
from envs import controller as C

model, data, log = make_env(0)
roll = C.Rollout(model, data)

# The two cutlery bodies, as the scene actually places them.
targets = {}
for nm in ("spoon", "fork"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
    targets[nm] = data.xpos[bid].copy()

base_r = np.array([dt.ARM_X, dt.ARM_Y])
print("arm base right", base_r, "spoon", targets["spoon"], "fork", targets["fork"])
for nm, p in targets.items():
    print(nm, "planar dist to nearest base",
          round(float(min(np.linalg.norm(p[:2] - b) for b in ARM_BASES.values())), 4))

# Sweep: aim the right arm at table-height points at increasing planar radius
# from its own base, along the bearing that points at the fork.
z = float(targets["fork"][2])
bearing = (targets["fork"][:2] - base_r); bearing /= np.linalg.norm(bearing)
rows = []
for r in [0.18, 0.22, 0.26, 0.28, 0.30, 0.32, 0.347, 0.36, 0.387, 0.40]:
    xy = base_r + r * bearing
    tgt = np.array([xy[0], xy[1], z])
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    mujoco.mj_forward(model, data)
    roll.trace = []
    roll.run([({"right": C.Move("right", lambda m, d, t=tgt: t,
                                opening=C.GRIPPER_OPEN, label=f"r{r}")}, 2.5)],
             settle=0.6)
    g = roll.grips["right"]
    tip = data.site_xpos[g.site].copy()
    err = float(np.linalg.norm(tip - tgt))
    achieved_r = float(np.linalg.norm(tip[:2] - base_r))
    tau = np.abs(data.actuator_force[g.act[:5]])
    rows.append({"requested_r_m": r, "achieved_r_m": round(achieved_r, 4),
                 "tip_err_mm": round(err * 1000, 2),
                 "ik_plan_err_mm": roll.trace[0]["ik_err_mm"],
                 "max_|tau|_Nm": round(float(tau.max()), 3),
                 "n_saturated": int((tau > 2.90).sum())})
    print(rows[-1], flush=True)

json.dump({"question": "is the cutlery outside the arm's force-limited reach?",
           "answer": "NO -- the arm reaches 0.417 m of planar radius; the cutlery sits at 0.347 and 0.387 m",
           "rows": rows, "cutlery": {k: [round(float(x), 4) for x in v] for k, v in targets.items()}},
          open(pathlib.Path(__file__).resolve().parent.parent / "evidence" / "reach_envelope.json", "w"), indent=1)
