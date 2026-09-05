#!/usr/bin/env python3
"""Render the submission cover image: PNG, 16:9, from the simulator itself.

The lablab Rule Book asks for one thing here -- *"Cover Image: Use PNG or JPG
format with 16:9 aspect ratio."* -- and says nothing about content.  That makes
a cover the easiest place in a submission to show something that was never
achieved, so this one is built the same way the demo video is: the picture is a
render of a REAL final state, and every word printed on it is derived from
``evidence/`` rather than typed.

  * the seed is chosen by rule (lowest seed reaching the best recorded score),
    not pinned, so it follows the controller when the controller improves
  * that seed is re-run live and its sub-goals are compared against the ones
    ``evidence/eval_seeds_scripted.json`` recorded -- a mismatch is fatal, so
    the frame cannot drift away from the number printed beside it
  * the caption's tallies come from the evaluation JSON, and the absence line
    is derived from the sub-goals that never fired, so it cannot go stale the
    way a hand-written sentence does

Run:  python3 scripts/render_cover.py --out evidence/cover_image.png
Check: python3 scripts/test_cover_image.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
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
EV = ROOT / "evidence"

W, H = 1920, 1080                       # 16:9 exactly
INSET = (480, 270)                      # 16:9 exactly
BAND_TOP = 792

FONT_CANDIDATES = {
    "title": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "body": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[kind]:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def font_name(kind: str) -> str:
    for path in FONT_CANDIDATES[kind]:
        if pathlib.Path(path).exists():
            return path
    return "PIL.ImageFont.load_default"


# --------------------------------------------------------------- derivation
def tallies(run: dict) -> dict:
    """Per-sub-goal counts over the seeds an evaluation run actually scored."""
    eps = run["episodes"]
    out = {g: sum(bool(e["task"]["subgoals"][g]) for e in eps) for g in SUBGOAL_ORDER}
    return {
        "seeds": len(eps),
        "per_goal": out,
        "subgoals_met": sum(out.values()),
        "subgoals_total": len(eps) * len(SUBGOAL_ORDER),
        "task_success": sum(bool(e["task"]["task_success"]) for e in eps),
    }


def pick_seed(run: dict) -> int:
    """Lowest seed reaching the best score this run recorded.

    A rule, not a literal: when the controller earns a new sub-goal the frame
    follows it, and no tick has to remember to move a pinned number.
    """
    eps = run["episodes"]
    best = max(e["task"]["subgoals_met"] for e in eps)
    return min(e["seed"] for e in eps if e["task"]["subgoals_met"] == best)


def project_title() -> str:
    """The repository's own H1 -- one name for the project, in one place."""
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError("README.md has no H1")


def caption_lines(scripted: dict, control: dict, seed: int, live: dict,
                  sim_time_s: float) -> list[dict]:
    s, c = tallies(scripted), tallies(control)
    earned = [g for g in SUBGOAL_ORDER if s["per_goal"][g]]
    never = [g for g in SUBGOAL_ORDER if not s["per_goal"][g]]

    measured = (
        "Measured over %d randomized seeds: " % s["seeds"]
        + "  ".join("%s %d/%d" % (g, s["per_goal"][g], s["seeds"])
                    for g in SUBGOAL_ORDER)
    )
    scored = (
        "%d/%d sub-goals against a %d/%d no-policy control on the same seeds "
        "and the same scorer." % (s["subgoals_met"], s["subgoals_total"],
                                  c["subgoals_met"], c["subgoals_total"])
    )
    absent = (
        "NOT SHOWN: task_success is %d/%d -- the table is never fully set."
        % (s["task_success"], s["seeds"])
        + (" Never earned on any seed: %s." % ", ".join(never) if never else "")
        + " No learned policy: envs/controller.py is scripted IK and reads no camera."
    )
    frame = (
        "This frame is the real final state of seed %d after %.1f s of simulation "
        "-- earned here: %s." % (seed, sim_time_s,
                                 ", ".join(g for g in SUBGOAL_ORDER if live[g]) or "nothing")
    )
    return [
        {"role": "title", "text": project_title()},
        {"role": "instruction", "text": INSTRUCTION},
        {"role": "measured", "text": measured},
        {"role": "scored", "text": scored},
        {"role": "absent", "text": absent},
        {"role": "frame", "text": frame},
        {"role": "earned_goals", "text": ", ".join(earned) or "nothing"},
    ]


# ------------------------------------------------------------------ drawing
def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=f) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def compose(scene: np.ndarray, top: np.ndarray, lines: list[dict],
            tag: str) -> Image.Image:
    img = Image.fromarray(scene).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")

    # top-right inset: the same instant from overhead, so the placement the
    # caption claims is visible rather than taken on trust
    ins = Image.fromarray(top).convert("RGB").resize(INSET)
    x0, y0 = W - INSET[0] - 48, 48
    d.rectangle([x0 - 3, y0 - 3, x0 + INSET[0] + 2, y0 + INSET[1] + 2],
                fill=(255, 255, 255, 210))
    img.paste(ins, (x0, y0))
    d.rectangle([x0, y0 + INSET[1] - 30, x0 + INSET[0], y0 + INSET[1]],
                fill=(0, 0, 0, 170))
    d.text((x0 + 10, y0 + INSET[1] - 26), "top_cam — same instant",
           font=font("body", 18), fill=(225, 230, 240))

    # top-left provenance tag
    tf = font("body", 22)
    d.rectangle([0, 0, 26 + int(d.textlength(tag, font=tf)), 52],
                fill=(0, 0, 0, 165))
    d.text((16, 14), tag, font=tf, fill=(235, 240, 248))

    # bottom band
    d.rectangle([0, BAND_TOP, W, H], fill=(8, 10, 14, 226))
    d.line([0, BAND_TOP, W, BAND_TOP], fill=(120, 200, 160, 255), width=3)

    by = {r["role"]: r["text"] for r in lines}
    y = BAND_TOP + 22
    d.text((56, y), by["title"], font=font("title", 46), fill=(255, 255, 255))
    y += 62
    for ln in wrap(d, '"' + by["instruction"] + '"', font("body", 25), W - 112):
        d.text((56, y), ln, font=font("body", 25), fill=(178, 190, 208))
        y += 32
    y += 8
    d.text((56, y), by["measured"], font=font("body", 24), fill=(140, 230, 160))
    y += 32
    d.text((56, y), by["scored"], font=font("body", 24), fill=(140, 230, 160))
    y += 34
    for ln in wrap(d, by["absent"], font("body", 23), W - 112):
        d.text((56, y), ln, font=font("body", 23), fill=(240, 180, 120))
        y += 30
    return img


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/cover_image.png")
    ap.add_argument("--meta", default="evidence/cover_image.json")
    args = ap.parse_args()

    scripted = json.loads((EV / "eval_seeds_scripted.json").read_text(encoding="utf-8"))
    control = json.loads((EV / "eval_seeds.json").read_text(encoding="utf-8"))
    seed = pick_seed(scripted)
    recorded = {e["seed"]: e for e in scripted["episodes"]}[seed]["task"]

    model, data, log = make_env(seed)
    mon = TaskMonitor(model)
    run_dinner_table(model, data, monitor=mon)
    rep = mon.report(data)
    live = {g: bool(rep["subgoals"][g]) for g in SUBGOAL_ORDER}
    rec = {g: bool(recorded["subgoals"][g]) for g in SUBGOAL_ORDER}
    if live != rec:
        raise SystemExit(
            "seed %d re-ran to %s but evidence/eval_seeds_scripted.json records "
            "%s -- refusing to render a frame the evaluation does not back"
            % (seed, live, rec))

    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera="scene_cam")
        scene = r.render()
        r.update_scene(data, camera="top_cam")
        top = r.render()

    lines = caption_lines(scripted, control, seed, live, float(rep["sim_time_s"]))
    tag = ("MuJoCo %s  ·  dual SO-101  ·  seed %d final state  ·  t=%.1f s"
           % (mujoco.__version__, seed, float(rep["sim_time_s"])))
    img = compose(scene, top, lines, tag)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)

    s, c = tallies(scripted), tallies(control)
    meta = {
        "image": str(out.relative_to(ROOT)),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
        "format": "PNG",
        "size": [W, H],
        "aspect_ratio": "16:9",
        "caption_band_top_px": BAND_TOP,
        "requirement": ("lablab Hackathon Rule Book, Cover Image and "
                        "Presentation: \"Cover Image: Use PNG or JPG format "
                        "with 16:9 aspect ratio.\""),
        "source_seed": seed,
        "seed_choice_rule": ("lowest seed reaching the best subgoals_met in "
                             "evidence/eval_seeds_scripted.json"),
        "rendered_from": {
            "main_camera": "scene_cam",
            "inset_camera": "top_cam",
            "state": "final state of a live rollout of envs/controller.py",
            "mujoco": mujoco.__version__,
            "sim_time_s": round(float(rep["sim_time_s"]), 3),
        },
        "cross_check": {"live": live, "evidence": rec, "match": live == rec},
        "derived": {"scripted": s, "control": c},
        "caption_lines": lines,
        "fonts": {k: font_name(k) for k in FONT_CANDIDATES},
        "not_claimed": (
            "The cover shows one real final state and the measured tallies "
            "beside it. It does not show a completed dinner table: "
            "task_success is %d/%d. %s were never earned on any seed. The "
            "controller is scripted IK over privileged simulator state, not a "
            "learned or camera-driven policy, and no Intel hardware "
            "measurement appears on this image."
            % (s["task_success"], s["seeds"],
               ", ".join(g for g in SUBGOAL_ORDER if not s["per_goal"][g])
               or "no sub-goals")),
        "reproducibility": (
            "Deterministic on this machine: same seed, same controller bytes, "
            "same MuJoCo build. The sha256 is an integrity anchor for THIS "
            "file; a different GPU or font build re-renders equivalent pixels, "
            "not identical bytes."),
    }
    (ROOT / args.meta).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    print("wrote %s  %dx%d  %.2f MB  sha256 %s"
          % (out, W, H, out.stat().st_size / 1e6, meta["sha256"][:16]))
    print("seed %d, live sub-goals %s" % (seed, [g for g in SUBGOAL_ORDER if live[g]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
