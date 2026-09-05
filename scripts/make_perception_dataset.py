#!/usr/bin/env python3
"""Render a labelled scene-layout dataset from the randomized environment.

Every sample is one ``top_cam`` frame plus the table layout that produced it,
read out of MuJoCo after ``mj_forward``.  No hand annotation and no external
data: the simulator is both the renderer and the label source.

Splits are separated by SEED, not by shuffling, so no compiled scene appears in
more than one split:

  train  --seed-base 1000, one compile per seed, several placement draws each
  val    --seed-base 2000
  eval10 the ten seeds ``scripts/eval_seeds.py`` scores, at exactly their
         evaluation initial state (no extra draw), so perception error can be
         quoted on the same scenes the task result is quoted on

Run:
  python3 scripts/make_perception_dataset.py --split train --compiles 500 --per-compile 8
  python3 scripts/make_perception_dataset.py --split val   --compiles 60  --per-compile 4
  python3 scripts/make_perception_dataset.py --split eval10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402
import mujoco                                               # noqa: E402

from envs.randomize import make_env, randomize_state        # noqa: E402
from envs import perception as P                            # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLITS = {                       # split -> (seed base, default compiles, draws)
    "train":  (1000, 500, 8),
    "val":    (2000, 60, 4),
    "eval10": (0, 10, 1),
}


def _labels(model, data) -> np.ndarray:
    """Table layout in network units, straight out of MjData."""
    m = []
    for name in P.OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        m += [float(data.xpos[bid][0]), float(data.xpos[bid][1])]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    m.append(float(data.qpos[model.jnt_qposadr[jid]]))
    return P.encode(np.array(m, dtype=np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--compiles", type=int, default=None)
    ap.add_argument("--per-compile", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base, n_comp, n_draw = SPLITS[args.split]
    n_comp = args.compiles if args.compiles is not None else n_comp
    n_draw = args.per_compile if args.per_compile is not None else n_draw
    out = pathlib.Path(args.out or ROOT / "data" / f"perception_{args.split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    # The eval10 split must reproduce the evaluated scenes byte for byte, so it
    # takes make_env's own initial state and nothing else.
    extra_draws = args.split != "eval10"

    imgs, labs, seeds, t0, skipped = [], [], [], time.time(), 0
    for i in range(n_comp):
        seed = base + i
        model, data, _ = make_env(seed)
        with mujoco.Renderer(model, height=P.IMG_H, width=P.IMG_W) as r:
            for k in range(n_draw):
                if extra_draws and k > 0:
                    # Independent placement draw on the already-compiled scene.
                    # envs/randomize.py is NOT modified: the wider drawer range
                    # below is applied here, on top of it, so the evaluation
                    # distribution used by eval_seeds.py is untouched.
                    try:
                        randomize_state(model, data, np.random.default_rng(
                            900_000 + 97 * seed + k))
                    except RuntimeError:      # rejection sampler gave up
                        skipped += 1
                        continue
                if extra_draws:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                            "drawer_slide")
                    rng = np.random.default_rng(500_000 + 131 * seed + k)
                    data.qpos[model.jnt_qposadr[jid]] = float(
                        rng.uniform(0.0, P.DRAWER_TRAVEL_M))
                    mujoco.mj_forward(model, data)
                r.update_scene(data, camera=P.CAMERA)
                imgs.append(r.render().copy())
                labs.append(_labels(model, data))
                seeds.append(seed)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_comp} compiles  {len(imgs)} samples  "
                  f"{time.time()-t0:.0f}s", flush=True)

    images = np.stack(imgs).astype(np.uint8)
    labels = np.stack(labs).astype(np.float32)
    np.savez_compressed(out, images=images, labels=labels,
                        seeds=np.asarray(seeds, dtype=np.int32))

    meta = {
        "split": args.split, "samples": int(len(images)),
        "compiles": n_comp, "per_compile": n_draw,
        "seed_base": base, "seed_range": [base, base + n_comp - 1],
        "camera": P.CAMERA, "image": [P.IMG_H, P.IMG_W, 3],
        "outputs": list(P.OUT_NAMES),
        "extra_placement_draws": extra_draws,
        "drawer_range_m": [0.0, P.DRAWER_TRAVEL_M] if extra_draws
                          else "envs/randomize.py RANGES['drawer_q'] only",
        "label_source": "MjData after mj_forward -- body xpos and drawer_slide qpos",
        "file": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
        "bytes": out.stat().st_size,
        "skipped_draws": skipped,
        "seconds": round(time.time() - t0, 1),
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=1),
                                          encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
