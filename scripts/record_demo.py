#!/usr/bin/env python3
"""Record the scripted rollout across N randomized seeds as one MP4.

Every frame is captioned with the seed, the randomized quantities that make
that seed different, the elapsed simulation time and the LIVE sub-goal state
read from ``envs/task.py`` -- the same predicates ``scripts/eval_seeds.py``
scores.  The caption is drawn from the simulator, not typed in, so what the
video claims and what the evaluation reports cannot drift apart.

Run:  python3 scripts/record_demo.py --seeds 10 --out evidence/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                             # noqa: E402
import mujoco                                                  # noqa: E402
from PIL import Image, ImageDraw, ImageFont                    # noqa: E402

from envs.randomize import make_env                            # noqa: E402
from envs.task import TaskMonitor, SUBGOAL_ORDER, INSTRUCTION  # noqa: E402
from envs.controller import run_dinner_table                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
W, H, FPS, SPEED = 960, 540, 25, 8.0
FONT = ImageFont.load_default(15)
SMALL = ImageFont.load_default(12)


def caption(rgb: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 26 + 18 * len(lines)], fill=(0, 0, 0, 165))
    y = 5
    for text, colour in lines:
        d.text((10, y), text, font=FONT if y == 5 else SMALL, fill=colour)
        y += 18
    return np.asarray(img)


def episode_frames(seed: int, writer) -> dict:
    model, data, log = make_env(seed)
    mon = TaskMonitor(model)
    every = max(1, int(SPEED / (FPS * model.opt.timestep)))
    dims = log["dims"]
    subtitle = (f"plate r={dims['plate_r']*1000:.0f}mm  mug r={dims['mug_r']*1000:.0f}mm  "
                f"cutlery {dims['cutlery_l']*1000:.0f}mm  "
                f"friction x{log['model']['friction_scale']:.2f}  "
                f"light {log['model']['light']['diffuse']:.2f}")
    state = {"n": 0}

    with mujoco.Renderer(model, height=H, width=W) as r:
        def on_step(d):
            state["n"] += 1
            if state["n"] % every:
                return
            r.update_scene(d, camera="scene_cam")
            sg = mon.subgoals(d)
            met = sum(sg.values())
            writer.write(caption(r.render(), [
                (f"seed {seed:02d}   t={float(d.time):5.1f}s   "
                 f"sub-goals {met}/5   [{SPEED:.0f}x speed]",
                 (255, 255, 255)),
                (subtitle, (185, 195, 210)),
                ("  ".join(("+ " if sg[g] else "- ") + g for g in SUBGOAL_ORDER),
                 (140, 235, 150) if met else (235, 175, 140)),
            ]))

        roll = run_dinner_table(model, data, monitor=mon, on_step=on_step)
    rep = mon.report(data)
    return {"seed": seed, "randomization": log, "rollout":
            {k: v for k, v in roll.items() if k != "trace"},
            "task": {k: v for k, v in rep.items() if k != "instruction"}}


class FFmpeg:
    def __init__(self, out: pathlib.Path):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not on PATH")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
            stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def not_claimed(rows: list[dict]) -> str:
    """Say what the video does NOT show, derived from the rollouts it filmed.

    This sentence was a hand-written literal until 2026-09-05 and it went stale
    the moment the controller started placing the plate: it still said the only
    sub-goal earned was the drawer.  Deriving it means it cannot say that again.
    """
    counts: dict[str, int] = {}
    for r in rows:
        for goal, ok in r["task"]["subgoals"].items():
            counts[goal] = counts.get(goal, 0) + bool(ok)
    n = len(rows)
    earned = [f"{g} {c}/{n}" for g, c in counts.items() if c]
    never = [g for g, c in counts.items() if not c]
    wins = sum(1 for r in rows if r["task"]["task_success"])
    return (f"This video does NOT show the task being completed: task_success is "
            f"{wins}/{n}. It shows the scripted controller running on {n} "
            f"randomized seeds and the live sub-goal state it earns, which is "
            + (", ".join(earned) if earned else "nothing")
            + (f". Never earned on any seed: {', '.join(never)}." if never else "."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="evidence/demo_scripted_10seeds.mp4")
    args = ap.parse_args()

    out = ROOT / args.out
    w = FFmpeg(out)
    rows = []
    for s in range(args.seeds):
        rows.append(episode_frames(s, w))
        t = rows[-1]["task"]
        print(f"seed {s:2d}  sub-goals {t['subgoals_met']}/5  "
              f"drawer {t['max_drawer_travel_m']:.3f}m  frames {w.frames}")
    w.close()

    met = [r["task"]["subgoals_met"] for r in rows]
    meta = {
        "video": str(out.relative_to(ROOT)),
        "sha256": None,
        "seeds": args.seeds,
        "fps": FPS, "speed": SPEED, "size": [W, H], "frames": w.frames,
        "instruction": INSTRUCTION,
        "subgoals_met_per_seed": met,
        "subgoals_met_total": f"{sum(met)}/{5 * len(met)}",
        "task_success_count": sum(1 for r in rows if r["task"]["task_success"]),
        "not_claimed": not_claimed(rows),
        "episodes": rows,
    }
    import hashlib
    meta["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT / "evidence" / "demo_scripted_10seeds.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB, {w.frames} frames)  "
          f"sub-goals {sum(met)}/{5*len(met)}, task_success "
          f"{meta['task_success_count']}/{len(met)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
