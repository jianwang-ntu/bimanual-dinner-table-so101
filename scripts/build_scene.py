#!/usr/bin/env python3
"""Write the nominal dinner-table scene to envs/dinner_table.xml.

Run:  python3 scripts/build_scene.py
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mujoco                                                 # noqa: E402
from envs import dinner_table as dt                           # noqa: E402


def main() -> None:
    spec = dt.build()
    model = spec.compile()          # fails loudly if the scene is malformed
    data = mujoco.MjData(model)

    qpos, ctrl, errs = dt.home_keyframe(model, data)
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    model = spec.compile()

    dt.OUT.parent.mkdir(parents=True, exist_ok=True)
    spec.meshdir = os.path.relpath(dt.ASSETS, dt.OUT.parent)
    dt.OUT.write_text(spec.to_xml(), encoding="utf-8")

    print(f"wrote {dt.OUT.relative_to(dt.ROOT)}")
    print(f"  nq={model.nq} nu={model.nu} nbody={model.nbody} "
          f"ngeom={model.ngeom} nkey={model.nkey}")
    print("  home-pose IK residual: "
          + ", ".join(f"{k}{v * 1000:.1f}mm" for k, v in errs.items()))


if __name__ == "__main__":
    main()
