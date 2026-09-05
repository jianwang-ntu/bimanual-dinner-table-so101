#!/usr/bin/env python3
"""Scene-state perception: table-top object layout read from one camera.

The scripted controller in ``envs/controller.py`` reads object positions
straight out of ``MjData``.  That is privileged information a real robot does
not have, and it is also the reason this entry has no inference cost to
optimise.  This module is the first piece that replaces it: a small
convolutional network that takes the ``top_cam`` RGB frame and regresses the
planar layout of the table -- plate, mug and bottle centres, plus how far the
drawer is out.

Scope, stated here so nothing downstream has to guess:

  * It is a perception model, not a policy.  It emits scene state, not actions.
  * It is NOT wired into the control loop.  The 16/50 sub-goal number reported
    in ``evidence/eval_seeds_scripted.json`` is produced by the privileged
    scripted controller and is unaffected by anything in this file.
  * It is trained on initial states with both arms at the home keyframe.  It
    has not been evaluated mid-rollout, when the arms occlude the table.

What it is for: it is a real, trained, exportable model, so the OpenVINO
conversion and the Intel bench-test script have something of this project's own
to measure instead of a stand-in.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- input spec
CAMERA = "top_cam"
IMG_H, IMG_W = 128, 224
IN_SHAPE = (1, 3, IMG_H, IMG_W)          # NCHW, float32, [0, 1]

# --------------------------------------------------------------- output spec
# One row per regressed quantity: (name, scale in metres).  The network works
# in units of `scale` so every output is roughly [-1, 1]; `decode` puts them
# back into metres.  Scales are the table half-extents from envs/dinner_table.py
# and the drawer's full travel.
OUTPUTS: tuple[tuple[str, float], ...] = (
    ("plate_x", 0.45), ("plate_y", 0.35),
    ("mug_x", 0.45), ("mug_y", 0.35),
    ("bottle_x", 0.45), ("bottle_y", 0.35),
    ("drawer_q", 0.09),
)
OUT_NAMES = tuple(n for n, _ in OUTPUTS)
OUT_SCALE = np.array([s for _, s in OUTPUTS], dtype=np.float32)
N_OUT = len(OUTPUTS)

OBJECTS = ("plate", "mug", "bottle")
DRAWER_TRAVEL_M = 0.09


def encode(metres: np.ndarray) -> np.ndarray:
    """Metres -> network units."""
    return np.asarray(metres, dtype=np.float32) / OUT_SCALE


def decode(units: np.ndarray) -> np.ndarray:
    """Network units -> metres."""
    return np.asarray(units, dtype=np.float32) * OUT_SCALE


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC frame straight out of ``mujoco.Renderer`` -> float32 NCHW."""
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    if a.ndim == 3:
        a = a[None]
    return np.ascontiguousarray(a.transpose(0, 3, 1, 2))


def position_error_mm(pred_units: np.ndarray, true_units: np.ndarray) -> dict:
    """Per-output absolute error in millimetres, plus the object-centre norm."""
    pred_m = decode(np.asarray(pred_units, dtype=np.float32))
    true_m = decode(np.asarray(true_units, dtype=np.float32))
    err = np.abs(pred_m - true_m) * 1000.0
    out = {n: float(err[..., i].mean()) for i, n in enumerate(OUT_NAMES)}
    for i, obj in enumerate(OBJECTS):
        d = pred_m[..., 2 * i:2 * i + 2] - true_m[..., 2 * i:2 * i + 2]
        out[f"{obj}_centre_mm"] = float(np.linalg.norm(d, axis=-1).mean() * 1000.0)
    out["worst_centre_mm"] = max(out[f"{o}_centre_mm"] for o in OBJECTS)
    return out


# ------------------------------------------------------------------- network
CHANNELS = (3, 24, 48, 96, 128)          # one stride-2 block per step: /16 total


def build_model():
    """The regressor.  Imported lazily so the env package stays torch-free."""
    import torch
    from torch import nn

    class SpatialSoftmax(nn.Module):
        """Per-channel 2-D expectation of a softmax over the feature map.

        Coordinate regression is the one thing a global average pool cannot do:
        averaging over space throws away exactly the position being asked for.
        A spatial softmax keeps it -- each channel becomes a soft keypoint whose
        (x, y) is read off in normalized image coordinates, which is the
        representation the downstream head actually needs.
        """

        def __init__(self, temperature: float = 1.0):
            super().__init__()
            self.log_t = nn.Parameter(torch.tensor(float(np.log(temperature))))

        def forward(self, feat):
            n, c, h, w = feat.shape
            flat = (feat.reshape(n, c, h * w) / self.log_t.exp()).softmax(-1)
            # arange, not linspace: torch.onnx's TorchScript tracer emits
            # linspace as a float64 constant whatever dtype is asked for, which
            # promotes the whole head to f64 and makes the CPU plugin refuse the
            # INT8 subgraph. This form traces as f32 and is otherwise identical.
            ys = torch.arange(h, device=feat.device, dtype=feat.dtype) \
                / max(h - 1, 1) * 2.0 - 1.0
            xs = torch.arange(w, device=feat.device, dtype=feat.dtype) \
                / max(w - 1, 1) * 2.0 - 1.0
            grid_y = ys.view(h, 1).expand(h, w).reshape(1, 1, h * w)
            grid_x = xs.view(1, w).expand(h, w).reshape(1, 1, h * w)
            ex = (flat * grid_x).sum(-1)
            ey = (flat * grid_y).sum(-1)
            return torch.cat([ex, ey], dim=1)             # (n, 2c)

    class SceneStateCNN(nn.Module):
        """Four stride-2 conv blocks, a spatial softmax, then a small MLP."""

        def __init__(self, n_out: int = N_OUT):
            super().__init__()
            blocks = []
            for cin, cout in zip(CHANNELS[:-1], CHANNELS[1:]):
                blocks += [nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                           nn.BatchNorm2d(cout),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(cout, cout, 3, stride=1, padding=1),
                           nn.BatchNorm2d(cout),
                           nn.ReLU(inplace=True)]
            self.trunk = nn.Sequential(*blocks)           # 128x224 -> 8x14
            self.keypoints = SpatialSoftmax()
            c = CHANNELS[-1]
            self.head = nn.Sequential(
                nn.Linear(2 * c, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, n_out))

        def forward(self, x):
            return self.head(self.keypoints(self.trunk(x)))

    return SceneStateCNN()
