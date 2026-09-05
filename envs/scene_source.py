#!/usr/bin/env python3
"""Where the controller's object positions come from.

Until this module existed, ``envs/controller.py`` read the pose of every object
it was about to touch straight out of ``MjData``.  That is privileged state a
real bimanual cell does not have, and it is why the track's VLA / multi-modal
criterion scored nothing: there was a trained perception model in the
repository (``envs/perception.py``) but it sat beside the control loop rather
than inside it.

A ``SceneSource`` is the seam.  The controller asks *it* for object positions,
and what comes back depends on which source is installed:

``PrivilegedScene``   ``MjData`` verbatim.  The behaviour the entry has always
                      had, kept so it stays available as the control.
``PerceivedScene``    one ``top_cam`` frame per planning instant, pushed
                      through the exported OpenVINO IR, decoded to the table
                      layout.  Nothing in the control loop reads an object's
                      true pose.
``BlindScene``        the nominal table layout from ``envs/randomize.py``,
                      ignoring the per-episode randomization.  This is the
                      negative control: if the controller scores the same with
                      it as without it, the controller is not actually
                      consuming what the source returns and every perceived
                      number below would be meaningless.

What is and is not replaced -- stated here rather than left to be inferred:

  REPLACED   the planar (x, y) centre of plate, mug and bottle, and the drawer
             opening.  Those are the quantities ``envs/perception.py`` regresses
             and the ones domain randomization actually moves (+-45 mm of
             placement jitter, 0-20 mm of drawer).  A site rigidly attached to
             one of those bodies -- ``plate_grasp``, ``mug_grasp``,
             ``bottle_grasp``, ``drawer_handle_site`` -- moves with its parent's
             estimate, which is what an estimate of a rigid body means.

  NOT REPLACED, still read from the simulator, and each one is a real privilege:
    * object HEIGHT (z).  The network has no z output.
    * object YAW.  ``long_axis``/``across`` still read ``data.xmat``.
    * object DIMENSIONS.  ``geom_width`` still reads ``model.geom_size``.
    * SPOON and FORK.  They are not in the network's output head at all, so no
      source here can estimate them; their reads stay privileged.
    * the arms' own joints, jaw geoms and contacts.  That is proprioception and
      touch, not vision, and a real cell has both.

So this is a partial substitution and must be reported as one.  What it does
establish is that the loop now closes through a camera and a network for the
quantity the task varies most, and that the cost of doing so is a measured
number rather than an assumption.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import perception as P
from .randomize import NOMINAL_XY, RANGES

# Exactly the bodies envs/perception.py has outputs for.  Derived, not retyped:
# if the network's head changes, this follows it.
PERCEIVED_BODIES: tuple[str, ...] = tuple(P.OBJECTS)
DRAWER_JOINT = "drawer_slide"
DRAWER_BODY = "drawer"


def _bid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _sid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _drawer_adr(model) -> int:
    return int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)])


def _drawer_axis(model) -> np.ndarray:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
    return np.asarray(model.jnt_axis[j], dtype=float)


def truth(model, data) -> dict:
    """The quantities a source is allowed to estimate, read privileged.

    Used by the sources to form their offset, and by the evaluation to score
    the estimate.  The controller never calls it.
    """
    out = {}
    for obj in PERCEIVED_BODIES:
        c = data.xpos[_bid(model, obj)]
        out[f"{obj}_x"], out[f"{obj}_y"] = float(c[0]), float(c[1])
    out["drawer_q"] = float(data.qpos[_drawer_adr(model)])
    return out


# --------------------------------------------------------------- the sources
class PrivilegedScene:
    """``MjData`` verbatim.  Identity: installing it changes nothing."""

    name = "privileged"
    estimates = ()

    def body_xpos(self, model, data, body: str) -> np.ndarray:
        return np.array(data.xpos[_bid(model, body)], dtype=float)

    def site_xpos(self, model, data, site: str) -> np.ndarray:
        return np.array(data.site_xpos[_sid(model, site)], dtype=float)

    def drawer_q(self, model, data) -> float:
        return float(data.qpos[_drawer_adr(model)])

    def report(self) -> dict:
        return {"source": self.name, "inferences": 0,
                "note": "no estimate was made; the controller read MjData"}

    def close(self) -> None:
        pass


class _EstimatingScene(PrivilegedScene):
    """Common machinery: estimate the layout, then serve everything as an offset.

    Serving an *offset* rather than an absolute is what lets a centre estimate
    reach a grasp site.  ``plate_grasp`` is welded to the plate; if the plate is
    believed to be 4 mm east of where it is, then so is every point on it.  The
    controller therefore gets a consistent, wrong-by-the-estimate's-error view
    of the object, which is exactly what a perception stack hands a planner.
    """

    name = "estimating"
    estimates = PERCEIVED_BODIES

    def __init__(self):
        self._key = None
        self._delta: dict[str, np.ndarray] = {}
        self._ddraw = 0.0
        self.n_infer = 0
        self.errors: list[dict] = []

    # -- to be provided by the subclass ------------------------------------
    def _estimate(self, model, data) -> dict:
        raise NotImplementedError

    # -- offset bookkeeping -------------------------------------------------
    def _refresh(self, model, data) -> None:
        key = (id(data), round(float(data.time), 9))
        if key == self._key:
            return
        est = self._estimate(model, data)
        tru = truth(model, data)
        self._delta = {
            obj: np.array([est[f"{obj}_x"] - tru[f"{obj}_x"],
                           est[f"{obj}_y"] - tru[f"{obj}_y"], 0.0])
            for obj in PERCEIVED_BODIES}
        self._ddraw = float(est["drawer_q"] - tru["drawer_q"])
        # Logged for the report only.  Nothing in the control path reads it.
        row = {"t": round(float(data.time), 3)}
        for obj in PERCEIVED_BODIES:
            row[f"{obj}_mm"] = round(
                float(np.linalg.norm(self._delta[obj][:2])) * 1000.0, 3)
        row["drawer_mm"] = round(abs(self._ddraw) * 1000.0, 3)
        self.errors.append(row)
        self._key = key

    def body_xpos(self, model, data, body: str) -> np.ndarray:
        self._refresh(model, data)
        c = np.array(data.xpos[_bid(model, body)], dtype=float)
        return c + self._delta.get(body, 0.0)

    def site_xpos(self, model, data, site: str) -> np.ndarray:
        self._refresh(model, data)
        sid = _sid(model, site)
        p = np.array(data.site_xpos[sid], dtype=float)
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.site_bodyid[sid]))
        if parent in self._delta:
            return p + self._delta[parent]
        if parent == DRAWER_BODY:
            return p + _drawer_axis(model) * self._ddraw
        return p

    def drawer_q(self, model, data) -> float:
        self._refresh(model, data)
        return float(data.qpos[_drawer_adr(model)]) + self._ddraw

    def report(self) -> dict:
        e = self.errors
        out = {"source": self.name, "inferences": self.n_infer,
               "planning_instants": len(e)}
        if e:
            for k in ("plate_mm", "mug_mm", "bottle_mm", "drawer_mm"):
                v = [r[k] for r in e]
                out[k] = {"mean": round(float(np.mean(v)), 3),
                          "max": round(float(np.max(v)), 3),
                          "first": v[0]}
            out["worst_object_mean_mm"] = round(max(
                out[k]["mean"] for k in ("plate_mm", "mug_mm", "bottle_mm")), 3)
        return out


class PerceivedScene(_EstimatingScene):
    """One rendered ``top_cam`` frame per planning instant, through the IR.

    The renderer is the same one ``scripts/make_perception_dataset.py`` used to
    build the training set, at the same size, so the controller sees the
    distribution the network was fitted on -- at t=0.  It is emphatically NOT
    the distribution later in the episode, when two arms are over the table;
    that drift is measured per episode and reported rather than assumed away.
    """

    def __init__(self, precision: str = "FP32", device: str = "CPU",
                 models_dir=None, backend: str = "openvino"):
        super().__init__()
        import pathlib
        self.precision = precision.upper()
        self.device = device
        self.backend = backend
        self.name = f"perceived[{backend}:{self.precision}@{device}]"
        root = pathlib.Path(__file__).resolve().parent.parent
        self._models = pathlib.Path(models_dir) if models_dir else root / "models"
        self._renderer = None
        self._infer_fn = None

    # -- lazy resources -----------------------------------------------------
    def _runtime(self):
        if self._infer_fn is not None:
            return self._infer_fn
        if self.backend == "openvino":
            import openvino as ov
            xml = self._models / f"scene_state_cnn_{self.precision.lower()}.xml"
            if not xml.exists():
                raise FileNotFoundError(
                    f"{xml} is missing -- run scripts/export_openvino.py")
            compiled = ov.Core().compile_model(str(xml), self.device)
            port = compiled.output(0)
            req = compiled.create_infer_request()

            def run(x):
                return np.asarray(req.infer(x)[port])[0]
        elif self.backend == "torch":
            import torch
            ckpt = self._models / "scene_state_cnn.pt"
            net = P.build_model()
            net.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
            net.eval()

            def run(x):
                with torch.no_grad():
                    return net(torch.from_numpy(x)).numpy()[0]
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        self._infer_fn = run
        return run

    def _render(self, model, data) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=P.IMG_H, width=P.IMG_W)
        self._renderer.update_scene(data, camera=P.CAMERA)
        return self._renderer.render()

    def _estimate(self, model, data) -> dict:
        x = P.preprocess(self._render(model, data))
        units = self._runtime()(x)
        self.n_infer += 1
        metres = P.decode(np.asarray(units, dtype=np.float32).reshape(-1))
        return {n: float(v) for n, v in zip(P.OUT_NAMES, metres)}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class BlindScene(_EstimatingScene):
    """The nominal layout, ignoring randomization.  The negative control.

    Every number here is taken from ``envs/randomize.py`` -- the nominal object
    positions the sampler jitters around, and the midpoint of the drawer's own
    start range -- so the control corrupts by exactly as much as the
    randomization moves things, and not by an invented constant.
    """

    name = "blind[nominal_layout]"

    def _estimate(self, model, data) -> dict:
        lo, hi = RANGES["drawer_q"]
        out = {"drawer_q": 0.5 * (lo + hi)}
        for obj in PERCEIVED_BODIES:
            nx, ny = NOMINAL_XY[obj]
            out[f"{obj}_x"], out[f"{obj}_y"] = float(nx), float(ny)
        return out


# ------------------------------------------------------------------ registry
SOURCES = {"privileged": PrivilegedScene, "perceived": PerceivedScene,
           "blind": BlindScene}

_ACTIVE: PrivilegedScene = PrivilegedScene()


def active() -> PrivilegedScene:
    return _ACTIVE


def install(source) -> None:
    """Install the source the controller will read from.

    Module-level because the controller's waypoints are closures built by
    ``dinner_table_script()`` long before a model or data exists; threading a
    source through every one of them would change the shape of a script the
    rest of the entry is measured against.
    """
    global _ACTIVE
    _ACTIVE = source


def make(kind: str, **kw):
    if kind not in SOURCES:
        raise ValueError(f"unknown scene source {kind!r}; have {sorted(SOURCES)}")
    cls = SOURCES[kind]
    return cls(**kw) if kind == "perceived" else cls()


def reset() -> None:
    install(PrivilegedScene())
