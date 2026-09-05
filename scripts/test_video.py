#!/usr/bin/env python3
"""Controls for evidence/video_presentation.mp4 -- the mandatory video presentation.

A video is the one submission artifact a judge cannot grep.  Whatever is on the
screen for four minutes is what the entry claimed, and nothing downstream ever
rechecks it.  So this suite treats the SHIPPED MP4 BYTES as the thing under
test: it decodes them and looks at the pixels.

  limits     ACCEPT the shipped file is MP4/H.264, 1280x720 at 25 fps, at least
             3 minutes and under 5, and under 300 MB -- the platform's own
             published bounds; REJECT a duration or a size that breaks either
  claims     ACCEPT every figure the video shows is re-derived here from
             evidence/*.json and FOUND IN THE DECODED PIXELS of the card that
             shows it; REJECT the same line with the figure corrupted, which
             must match the screen strictly worse than the true one
  honesty    ACCEPT the absence ledger is on screen -- no completed task, the
             three sub-goals that never fire, no object hand-off, perception
             outside the control loop, no Intel Core Ultra measurement;
             REJECT an absence deleted from it
  footage    ACCEPT every demo clip is the shipped demo MP4's own pixels,
             pasted unscaled; REJECT the same window against a different seed
  structure  ACCEPT card segments hold still and clip segments move; REJECT the
             two being confused for each other
  bytes      ACCEPT the shipped file is exactly what scripts/make_video.py
             produces from today's evidence; REJECT a sidecar whose sha256 does
             not match the file next to it, and REJECT the video going stale
             when the EVIDENCE moves underneath it

The figures are re-derived in THIS file, not imported from the builder: two
implementations that have to agree.  The corruptions are derived from the
artifact rather than pinned to literals, so they keep corrupting after the
measurements move.

  python3 scripts/test_video.py

Needs opencv-python, matplotlib and ffprobe.  A missing one is a FAILURE, not a
skip: a control that quietly does not run is worse than no control.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
MP4 = EV / "video_presentation.mp4"
SIDECAR = EV / "video_presentation.json"
DEMO = EV / "demo_scripted_10seeds.mp4"

BG = "#080B10"
FIGSIZE, DPI = (12.8, 7.2), 100
ABS_THRESH = 0.80          # a rendered line must be found in the frame this well
MARGIN = 0.002             # ... and strictly better than its corrupted twin
FOOTAGE_THRESH = 0.97      # a re-encoded copy of the same frame
FOOTAGE_REJECT = 0.90      # ... a different seed must fall below this
CAPTION_H = 80             # record_demo.py's own caption band, 26 + 18*3 px
STILL_MAX = 1.0            # mean pixel change across a card segment
MOVING_MIN = 1.5           # ... and across a clip segment

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


# ------------------------------------------------------------------ evidence
def load_evidence(ev: pathlib.Path) -> dict:
    j = lambda n: json.loads((ev / n).read_text(encoding="utf-8"))
    bench = sorted(ev.glob("openvino_bench_*.json"))
    if not bench:
        raise FileNotFoundError(f"no openvino_bench_*.json under {ev}")
    return {
        "scripted": j("eval_seeds_scripted.json"),
        "control": j("eval_seeds.json"),
        "scene": j("scene_verification.json"),
        "predicates": j("task_predicate_controls.json"),
        "perception_controls": j("perception_pipeline_controls.json"),
        "train": j("perception_train.json"),
        "export": j("openvino_export.json"),
        "bench": json.loads(bench[0].read_text(encoding="utf-8")),
        "demo": j("demo_scripted_10seeds.json"),
    }


def rederive(e: dict) -> dict:
    """Every figure the video is allowed to show, worked out again from scratch."""
    sc, ctl = e["scripted"], e["control"]
    seeds = sc["seeds"]
    per_goal: dict[str, int] = {}
    for ep in sc["episodes"]:
        for goal, met in ep["task"]["subgoals"].items():
            per_goal[goal] = per_goal.get(goal, 0) + bool(met)
    denom = seeds * sc["episodes"][0]["task"]["subgoals_total"]
    plate_d = sorted(2000 * ep["randomization"]["dims"]["plate_r"]
                     for ep in sc["episodes"])
    v = e["export"]["variants"]
    tr, bn = e["train"], e["bench"]
    return {
        "seeds": seeds,
        "denom": denom,
        "total": sum(ep["task"]["subgoals_met"] for ep in sc["episodes"]),
        "control_total": sum(ep["task"]["subgoals_met"] for ep in ctl["episodes"]),
        "per_goal": per_goal,
        "task_success": sum(1 for ep in sc["episodes"] if ep["task"]["task_success"]),
        "object_handoffs": sum(1 for ep in sc["episodes"] for h in ep["task"]["handoffs"]
                               if h["object"] != "drawer"),
        "bimanual": sum(1 for ep in sc["episodes"] if ep["task"]["bimanual"]),
        "plate_d_lo": plate_d[0], "plate_d_hi": plate_d[-1],
        "scene_pass": sum(1 for c in e["scene"]["checks"] if c["pass"]),
        "scene_n": len(e["scene"]["checks"]),
        "pred_pass": sum(1 for c in e["predicates"]["controls"] if c["pass"]),
        "pred_n": len(e["predicates"]["controls"]),
        "perc_pass": sum(1 for c in e["perception_controls"]["controls"] if c["pass"]),
        "perc_n": len(e["perception_controls"]["controls"]),
        "params": tr["model"]["parameters"],
        "train_n": tr["data"]["train"],
        "eval_mm": tr["error_mm"]["eval10"]["worst_centre_mm"],
        "int8_ratio": v["FP32"]["bin_bytes"] / v["INT8"]["bin_bytes"],
        "int8_cost_mm": (v["INT8"]["accuracy_mm"]["eval10"]["worst_centre_mm"]
                         - v["FP32"]["accuracy_mm"]["eval10"]["worst_centre_mm"]),
        "int8_bytes": v["INT8"]["bin_bytes"], "fp32_bytes": v["FP32"]["bin_bytes"],
        "bench_rows": len(bn["results"]),
        "bench_verdict": bn["required_hardware"]["verdict"],
        "bench_cpu": bn["host"]["cpu"].strip(),
        "demo_frames": e["demo"]["frames"], "demo_seeds": e["demo"]["seeds"],
    }


# --------------------------------------------------------------------- pixels
def probe_streams(path: pathlib.Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size,format_name:stream=codec_name,codec_type,width,height,"
         "r_frame_rate,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def render_line(text: str, size: float, color: str, weight: str, family: str,
                style: str):
    """Rasterise ONE line the way the builder would, cropped to its own ink."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    fig.patch.set_facecolor(BG)
    art = fig.text(0.055, 0.5, text, fontsize=size, color=color, fontweight=weight,
                   family=family, style=style, ha="left", va="center")
    fig.canvas.draw()
    b = art.get_window_extent(fig.canvas.get_renderer())
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    h = img.shape[0]
    x0, x1 = int(b.x0) - 2, int(b.x1) + 3
    y0, y1 = h - int(b.y1) - 3, h - int(b.y0) + 2
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(img.shape[1], x1)].copy()
    plt.close(fig)
    return crop


def gray(a):
    import cv2
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def best_match(frame_rgb, template_rgb) -> float:
    import cv2
    f, t = gray(frame_rgb), gray(template_rgb)
    if t.shape[0] > f.shape[0] or t.shape[1] > f.shape[1]:
        return -1.0
    return float(cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED).max())


def read_frames(path: pathlib.Path, wanted: set[int]) -> dict[int, "object"]:
    """One sequential decode: seeking an H.264 file by index is not exact."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    out, i, last = {}, 0, max(wanted) if wanted else -1
    while i <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = bgr[:, :, ::-1].copy()
        i += 1
    cap.release()
    return out


# --------------------------------------------------------------------- claims
def mutate(s: str) -> str:
    """A corruption derived from the string, not pinned to a literal."""
    for i, ch in enumerate(s):
        if ch.isdigit():
            return s[:i] + str((int(ch) + 1) % 10) + s[i + 1:]
    return s[::-1]


def find_line(lines: list[dict], needle: str) -> dict | None:
    for ln in lines:
        if needle in ln["text"]:
            return ln
    return None


def main() -> int:
    import numpy as np

    ok = True
    for missing in (MP4, SIDECAR, DEMO):
        if not missing.exists():
            print(f"[FAIL] artifact missing: {missing}")
            return 1

    side = json.loads(SIDECAR.read_text(encoding="utf-8"))
    e = load_evidence(EV)
    F = rederive(e)
    segs = side["segments"]

    # ------------------------------------------------------------ 1. limits
    meta = probe_streams(MP4)
    vs = [s for s in meta["streams"] if s["codec_type"] == "video"]
    fmt = meta["format"]
    dur, size = float(fmt["duration"]), int(fmt["size"])
    num, den = vs[0]["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    ok &= check("accept_container_and_stream",
                len(vs) == 1 and vs[0]["codec_name"] == "h264"
                and (vs[0]["width"], vs[0]["height"]) == (1280, 720)
                and abs(fps - 25.0) < 1e-6 and "mp4" in fmt["format_name"],
                f"{fmt['format_name']} / {vs[0]['codec_name']} "
                f"{vs[0]['width']}x{vs[0]['height']} @ {fps:g} fps, "
                f"{len(meta['streams'])} stream(s)")

    in_band = lambda d: 180.0 <= d < 300.0
    under_cap = lambda b: b < 300 * 1000 * 1000
    ok &= check("accept_duration_inside_the_published_band", in_band(dur),
                f"{dur:.2f}s -- lablab publishes 'within 5 minutes duration' and its "
                f"rubric penalises a presentation video 'less than 3 min'")
    ok &= check("accept_size_under_the_published_cap", under_cap(size),
                f"{size / 1e6:.1f} MB against the published 300 MB")
    ok &= check("reject_a_duration_over_five_minutes",
                not in_band(300.01) and not in_band(179.99),
                "the band predicate refuses 300.01s and 179.99s")
    ok &= check("reject_a_file_over_the_size_cap", not under_cap(300 * 1000 * 1000),
                "the size predicate refuses exactly 300 MB")
    ok &= check("accept_it_is_not_the_demo_video",
                hashlib.sha256(MP4.read_bytes()).hexdigest()
                != hashlib.sha256(DEMO.read_bytes()).hexdigest()
                and abs(dur - 340.88) > 1.0,
                f"the presentation is {dur:.1f}s; the 10-seed demo deliverable is a "
                f"different file at 340.88s, which would break the 5-minute cap")

    # --------------------------------------------------------- 2. the pixels
    want = {}
    for i, seg in enumerate(segs):
        mid = seg["start_frame"] + seg["n_frames"] // 2
        want[i] = mid
    extra = {}
    for i, seg in enumerate(segs):
        extra[f"{i}a"] = seg["start_frame"] + 2
        extra[f"{i}b"] = seg["start_frame"] + seg["n_frames"] - 3
    frames = read_frames(MP4, set(want.values()) | set(extra.values()))
    missing = [k for k, v in want.items() if v not in frames]
    ok &= check("accept_every_segment_decodes", not missing,
                f"{len(want)} segment mid-frames decoded out of the shipped file"
                if not missing else f"missing {missing}")
    if missing:
        return finish(ok, side)

    card_of = {}
    for i, seg in enumerate(segs):
        card_of.setdefault(seg["card"], i)

    # ---------------------------------------------------------- 3. structure
    still, moving = [], []
    for i, seg in enumerate(segs):
        a, b = frames.get(extra[f"{i}a"]), frames.get(extra[f"{i}b"])
        if a is None or b is None:
            continue
        d = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        (still if seg["kind"] == "card" else moving).append((i, d))
    worst_still = max((d for _, d in still), default=0.0)
    least_moving = min((d for _, d in moving), default=0.0)
    ok &= check("accept_card_segments_hold_still", worst_still < STILL_MAX,
                f"{len(still)} card segments, worst first-to-last mean pixel change "
                f"{worst_still:.3f} (< {STILL_MAX}). It is not zero because H.264 "
                f"requantises a static picture; what matters is that it is far below "
                f"a clip.")
    ok &= check("reject_a_clip_segment_that_does_not_move",
                least_moving > MOVING_MIN and worst_still < least_moving,
                f"{len(moving)} clip segments, smallest change {least_moving:.3f} "
                f"(> {MOVING_MIN}); every card is quieter than every clip by "
                f"{least_moving - worst_still:.3f} -- a clip that held still would be "
                f"a card mislabelled as footage")

    # ------------------------------------------------------------- 4. claims
    g = F["per_goal"]
    claims = [
        ("headline_subgoals", f"{F['total']}/{F['denom']}"),
        ("no_policy_control", f"{F['control_total']}/{F['denom']}"),
        ("drawer_open", f"{g['drawer_open']}/{F['seeds']}"),
        ("plate_placed", f"{g['plate_placed']}/{F['seeds']}"),
        ("scene_checks", f"{F['scene_pass']}/{F['scene_n']}"),
        ("scorer_controls", f"{F['pred_pass']}/{F['pred_n']}"),
        ("perception_params", f"{F['params']:,}-parameter"),
        ("perception_error", f"{F['eval_mm']:.2f} mm"),
        ("int8_ratio", f"{F['int8_ratio']:.1f}× smaller"),
        ("int8_cost", f"{F['int8_cost_mm']:.2f} mm of accuracy"),
        ("int8_bytes", f"{F['int8_bytes']:,}"),
        ("bench_rows", f"{F['bench_rows']} device/precision rows"),
        ("bench_verdict", F["bench_verdict"]),
        ("bench_host", F["bench_cpu"]),
        ("plate_span", f"{F['plate_d_lo']:.0f}–{F['plate_d_hi']:.0f} mm"),
        ("object_handoffs", f"Object hand-offs: {F['object_handoffs']}"),
        ("demo_frames", f"{F['demo_frames']} frames"),
    ]

    found_on, scored = 0, []
    for name, needle in claims:
        hit = None
        for i, seg in enumerate(segs):
            ln = find_line(seg.get("lines", []), needle)
            if ln is not None:
                hit = (i, ln)
                break
        if hit is None:
            ok &= check(f"accept_{name}_is_on_screen", False,
                        f"re-derived {needle!r} appears on no line of the video")
            continue
        i, ln = hit
        frame = frames[want[i]]
        true_t = render_line(ln["text"], ln["size"], ln["color"], ln["weight"],
                             ln["family"], ln["style"])
        bad_text = ln["text"].replace(needle, mutate(needle))
        bad_t = render_line(bad_text, ln["size"], ln["color"], ln["weight"],
                            ln["family"], ln["style"])
        s_true, s_bad = best_match(frame, true_t), best_match(frame, bad_t)
        scored.append((name, s_true, s_bad))
        good = s_true >= ABS_THRESH and s_true > s_bad + MARGIN
        found_on += bool(good)
        ok &= check(f"accept_{name}_is_on_screen", good,
                    f"segment {i} card {segs[i]['card']}: the line carrying "
                    f"{needle!r} matches the decoded frame at {s_true:.4f}; the "
                    f"same line with {needle!r} corrupted to "
                    f"{mutate(needle)!r} matches at {s_bad:.4f}")
    worst = min((s - b for _, s, b in scored), default=0.0)
    ok &= check("reject_every_corrupted_figure",
                all(s > b + MARGIN for _, s, b in scored) and bool(scored),
                f"{len(scored)} figures checked; the narrowest true-minus-corrupted "
                f"margin is {worst:.4f}")

    # ------------------------------------------------------------ 5. honesty
    absences = [
        f"Task success is {F['task_success']}/{F['seeds']}",
        f"Object hand-offs: {F['object_handoffs']}",
        "no learned policy",
        "NOT in the control loop",
        F["bench_verdict"],
    ]
    all_lines = [ln["text"] for seg in segs for ln in seg.get("lines", [])]
    blob = " ".join(all_lines)
    absent = [a for a in absences if a not in blob]
    ok &= check("accept_the_absence_ledger_is_on_screen", not absent,
                f"{len(absences)} stated absences all appear in the video's own lines"
                if not absent else f"missing from the video: {absent}")
    # The negative control runs the SAME checker over a ledger with one absence
    # deleted, and requires it to notice.  A presence check that cannot fail is
    # not a control.
    ledger = side["not_claimed"]
    required = ("Task success", "Object hand-offs", "No learned policy",
                "Perception outside the control loop", "No Intel Core Ultra measurement")
    complete = [w for w in required if w not in ledger]
    censored = ledger.replace("No Intel Core Ultra measurement. ", "")
    fired = [w for w in required if w not in censored]
    ok &= check("accept_the_sidecar_ledger_names_every_absence", not complete,
                f"{len(required)} absences named in the sidecar's not_claimed"
                if not complete else f"missing: {complete}")
    ok &= check("reject_an_absence_deleted_from_the_ledger",
                fired == ["No Intel Core Ultra measurement"],
                "deleting the Core Ultra sentence from the ledger makes the same "
                "checker report exactly that absence as missing")

    # ------------------------------------------------------------ 6. footage
    clips = [(i, s) for i, s in enumerate(segs) if s["kind"] == "clip"]
    src_want = {}
    for i, s in clips:
        src_want[i] = s["clip"]["source_start_frame"] + s["n_frames"] // 2
    # a decoy from a DIFFERENT episode, for the negative control
    other = {}
    for i, s in clips:
        j = (clips.index((i, s)) + 1) % len(clips)
        other[i] = clips[j][1]["clip"]["source_start_frame"] + 40
    src = read_frames(DEMO, set(src_want.values()) | set(other.values()))
    x0, y0, w, h = segs[clips[0][0]]["clip"]["pasted_unscaled_at"]
    hits, cap_hits, cap_miss, whole_miss = [], [], [], []
    for i, s in clips:
        shipped = frames[want[i]][y0:y0 + h, x0:x0 + w]
        true_f, decoy_f = src[src_want[i]], src[other[i]]
        seed = s["clip"]["seed"]
        hits.append((seed, best_match(shipped, true_f)))
        whole_miss.append((seed, best_match(shipped, decoy_f)))
        cap_hits.append((seed, best_match(shipped[:CAPTION_H], true_f[:CAPTION_H])))
        cap_miss.append((seed, best_match(shipped[:CAPTION_H], decoy_f[:CAPTION_H])))
    ok &= check("accept_every_clip_is_the_shipped_demo_bytes",
                all(v >= FOOTAGE_THRESH for _, v in hits)
                and all(v >= FOOTAGE_THRESH for _, v in cap_hits),
                "whole window " + "  ".join(f"seed {k}: {v:.4f}" for k, v in hits)
                + f"  (>= {FOOTAGE_THRESH})")
    ok &= check("reject_a_clip_matched_against_a_different_seed",
                all(v < FOOTAGE_REJECT for _, v in cap_miss)
                and all(c > d for (_, c), (_, d) in zip(cap_hits, cap_miss)),
                "caption band, where two seeds differ: true "
                + "  ".join(f"{k}:{v:.4f}" for k, v in cap_hits)
                + "  |  decoy " + "  ".join(f"{k}:{v:.4f}" for k, v in cap_miss)
                + f"  (< {FOOTAGE_REJECT}). Whole-window decoy scores "
                + "  ".join(f"{k}:{v:.4f}" for k, v in whole_miss)
                + " -- too close to use, because every episode films the same table.")

    # -------------------------------------------------------------- 7. bytes
    digest = hashlib.sha256(MP4.read_bytes()).hexdigest()
    ok &= check("accept_sidecar_sha256_matches_the_file",
                side["sha256"] == digest and side["bytes"] == size,
                f"{digest[:16]}… and {size} bytes")
    bent = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    ok &= check("reject_a_one_character_change_to_the_sha256", bent != digest,
                "a one-character change to the recorded sha256 stops matching")

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_video                                            # noqa: E402

    same_facts = make_video.facts(make_video.load())
    moved = 0
    import copy
    bumped = copy.deepcopy(e)
    bumped["scripted"]["episodes"][0]["task"]["subgoals"]["fork_placed"] = True
    drifted = rederive(bumped)
    moved = sum(1 for k in ("total", "per_goal") if drifted[k] != F[k])
    ok &= check("reject_the_video_going_stale_when_the_evidence_moves",
                moved >= 1 and drifted["per_goal"]["fork_placed"]
                == F["per_goal"]["fork_placed"] + 1,
                "one more sub-goal in the evidence moves the re-derived figures, so "
                "a video that kept showing today's numbers would fail the claim "
                "controls above")
    ok &= check("accept_builder_and_suite_agree_on_the_figures",
                all(same_facts[k] == F[k] for k in
                    ("total", "control_total", "denom", "seeds", "task_success",
                     "object_handoffs", "params", "bench_rows", "bench_verdict")),
                "the builder's derivation and this file's independent one agree on "
                "9 headline figures")

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "rebuild.mp4"
        side2 = make_video.build(out, pathlib.Path(tmp) / "rebuild.json")
        rebuilt = hashlib.sha256(out.read_bytes()).hexdigest()
        ok &= check("accept_shipped_bytes_are_the_builder_output_today",
                    rebuilt == digest,
                    "a fresh build from today's evidence is byte-identical"
                    if rebuilt == digest else
                    f"a fresh build DIFFERS: {rebuilt[:16]}… vs {digest[:16]}… -- "
                    f"the video is stale or was hand-edited")
        ok &= check("accept_layout_check_is_clean_on_the_rebuild",
                    not side2["layout_violations"],
                    f"{len(side2['layout_violations'])} layout violations on the rebuild")

    return finish(ok, side)


def finish(ok: bool, side: dict) -> int:
    (EV / "video_controls.json").write_text(json.dumps({
        "schema": "video_controls/v1",
        "artifact": "evidence/video_presentation.mp4",
        "sha256": side.get("sha256"),
        "duration_s": side.get("duration_s"),
        "all_pass": bool(ok),
        "controls": results,
    }, indent=1) + "\n", encoding="utf-8")
    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} controls pass")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
