#!/usr/bin/env python3
"""Controls for the submission cover image.

A cover image is the one required artifact a judge sees before reading
anything, and the one a checker is most tempted to wave through -- "the file
exists, it is a PNG, move on".  A presence check like that cannot fail, so it
proves nothing.  These controls decode the image and re-derive every word
printed on it, and each mechanism is driven from BOTH sides: an ACCEPT on the
shipped bytes and a REJECT on a corruption derived from the artifact itself.

  format     ACCEPT the file is a real non-interlaced 8-bit truecolour PNG
             whose IHDR is exactly 16:9; REJECT an IHDR that is not
  integrity  ACCEPT the bytes hash to what evidence/cover_image.json records;
             REJECT a single flipped byte
  figures    ACCEPT every tally printed on the cover is re-derived from the
             evaluation JSON; REJECT the evidence moving underneath it
  honesty    ACCEPT the caption states the absences the evidence records --
             no completed task, every sub-goal that never fired, no learned
             policy; REJECT an inflated task-success figure and REJECT the
             absence clause being deleted
  provenance ACCEPT the seed obeys the published choice rule and its live
             sub-goals match the recorded episode; REJECT a frame whose
             sub-goals disagree with the evaluation
  pixels     ACCEPT the rendered region carries real structure and the caption
             band is dark; REJECT a flat placeholder of the right size

Standard library only, so it runs on a fresh clone before anything is
installed.  Run:  python3 scripts/test_cover_image.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
META = EV / "cover_image.json"

SUBGOAL_ORDER = ["drawer_open", "fork_placed", "spoon_placed",
                 "plate_placed", "mug_placed"]

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


# ------------------------------------------------------------- PNG decoding
def ihdr(raw: bytes) -> dict:
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    if raw[12:16] != b"IHDR":
        raise ValueError("first chunk is not IHDR")
    w, h, depth, colour, comp, filt, inter = struct.unpack(">IIBBBBB", raw[16:29])
    return {"width": w, "height": h, "bit_depth": depth, "colour_type": colour,
            "compression": comp, "filter": filt, "interlace": inter}


def pixels(raw: bytes, head: dict) -> list[bytearray]:
    """Decode an 8-bit truecolour non-interlaced PNG to RGB scanlines."""
    if (head["bit_depth"], head["colour_type"], head["interlace"]) != (8, 2, 0):
        raise ValueError("decoder handles 8-bit truecolour non-interlaced only")
    idat, i = b"", 8
    while i < len(raw):
        (length,) = struct.unpack(">I", raw[i:i + 4])
        kind = raw[i + 4:i + 8]
        if kind == b"IDAT":
            idat += raw[i + 8:i + 8 + length]
        i += 12 + length
    data = zlib.decompress(idat)

    w, h, bpp = head["width"], head["height"], 3
    stride = w * bpp
    rows: list[bytearray] = []
    prev = bytearray(stride)
    pos = 0
    for _ in range(h):
        ftype = data[pos]
        line = bytearray(data[pos + 1:pos + 1 + stride])
        pos += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if ftype == 1:
                line[x] = (line[x] + a) & 0xFF
            elif ftype == 2:
                line[x] = (line[x] + b) & 0xFF
            elif ftype == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 0xFF
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 0xFF
            elif ftype != 0:
                raise ValueError(f"bad filter type {ftype}")
        rows.append(line)
        prev = line
    return rows


def band_stats(rows: list[bytearray], y0: int, y1: int, step: int = 7) -> dict:
    vals, seen = [], set()
    for y in range(y0, y1, step):
        row = rows[y]
        for x in range(0, len(row) - 2, step * 3):
            r, g, b = row[x], row[x + 1], row[x + 2]
            vals.append((r + g + b) / 3.0)
            seen.add((r >> 3, g >> 3, b >> 3))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": round(mean, 2), "std": round(var ** 0.5, 2),
            "distinct_colours": len(seen), "samples": len(vals)}


# ------------------------------------------------------------- re-derivation
def tallies(run: dict) -> dict:
    eps = run["episodes"]
    per = {g: sum(bool(e["task"]["subgoals"][g]) for e in eps) for g in SUBGOAL_ORDER}
    return {"seeds": len(eps), "per_goal": per, "subgoals_met": sum(per.values()),
            "subgoals_total": len(eps) * len(SUBGOAL_ORDER),
            "task_success": sum(bool(e["task"]["task_success"]) for e in eps)}


def measured_text(s: dict) -> str:
    return ("Measured over %d randomized seeds: " % s["seeds"]
            + "  ".join("%s %d/%d" % (g, s["per_goal"][g], s["seeds"])
                        for g in SUBGOAL_ORDER))


def scored_text(s: dict, c: dict) -> str:
    return ("%d/%d sub-goals against a %d/%d no-policy control on the same "
            "seeds and the same scorer." % (s["subgoals_met"], s["subgoals_total"],
                                            c["subgoals_met"], c["subgoals_total"]))


def role(meta: dict, name: str) -> str:
    for line in meta["caption_lines"]:
        if line["role"] == name:
            return line["text"]
    raise KeyError(name)


def figures_match(meta: dict, scripted: dict, control: dict) -> tuple[bool, str]:
    s, c = tallies(scripted), tallies(control)
    want_m, want_s = measured_text(s), scored_text(s, c)
    if role(meta, "measured") != want_m:
        return False, "measured line is %r, evidence derives %r" % (
            role(meta, "measured")[:70], want_m[:70])
    if role(meta, "scored") != want_s:
        return False, "scored line is %r, evidence derives %r" % (
            role(meta, "scored")[:70], want_s[:70])
    return True, "%d/%d sub-goals and the %d/%d control re-derived verbatim" % (
        s["subgoals_met"], s["subgoals_total"], c["subgoals_met"], c["subgoals_total"])


def honesty_holds(meta: dict, scripted: dict) -> tuple[bool, str]:
    s = tallies(scripted)
    absent = role(meta, "absent")
    if "task_success is %d/%d" % (s["task_success"], s["seeds"]) not in absent:
        return False, "the cover does not state task_success %d/%d" % (
            s["task_success"], s["seeds"])
    never = [g for g in SUBGOAL_ORDER if not s["per_goal"][g]]
    missing = [g for g in never if g not in absent]
    if missing:
        return False, "never-earned sub-goals not named on the cover: %s" % missing
    if "No learned policy" not in absent:
        return False, "the cover does not disclose that no policy was learned"
    return True, ("states task_success %d/%d, names all %d never-earned "
                  "sub-goals, discloses the scripted controller"
                  % (s["task_success"], s["seeds"], len(never)))


# --------------------------------------------------------------------- main
def main() -> int:
    meta = json.loads(META.read_text(encoding="utf-8"))
    img_path = ROOT / meta["image"]
    raw = img_path.read_bytes()
    scripted = json.loads((EV / "eval_seeds_scripted.json").read_text(encoding="utf-8"))
    control = json.loads((EV / "eval_seeds.json").read_text(encoding="utf-8"))

    ok = True

    # ---------------------------------------------------------- format
    head = ihdr(raw)
    square = (head["width"] * 9 == head["height"] * 16)
    ok &= check("format.accept",
                square and head["bit_depth"] == 8 and head["colour_type"] == 2
                and head["interlace"] == 0 and head["width"] == meta["size"][0]
                and head["height"] == meta["size"][1],
                "IHDR %dx%d, 8-bit truecolour, non-interlaced, %d*9 == %d*16"
                % (head["width"], head["height"], head["width"], head["height"]))

    # REJECT: the same geometry test on a 4:3 IHDR built from this image's own
    # width, so the control keeps corrupting when the resolution changes.
    four_three = {**head, "height": head["width"] * 3 // 4}
    ok &= check("format.reject_non_16_9",
                not (four_three["width"] * 9 == four_three["height"] * 16),
                "a %dx%d IHDR is refused by the same test"
                % (four_three["width"], four_three["height"]))

    # ------------------------------------------------------- integrity
    digest = hashlib.sha256(raw).hexdigest()
    ok &= check("integrity.accept",
                digest == meta["sha256"] and len(raw) == meta["bytes"],
                "sha256 %s over %d bytes matches the record" % (digest[:16], len(raw)))

    flipped = bytearray(raw)
    flipped[len(flipped) // 2] ^= 0x01
    ok &= check("integrity.reject_flipped_byte",
                hashlib.sha256(bytes(flipped)).hexdigest() != meta["sha256"],
                "one flipped byte at offset %d is caught" % (len(raw) // 2))

    # --------------------------------------------------------- figures
    good, detail = figures_match(meta, scripted, control)
    ok &= check("figures.accept", good, detail)

    # REJECT: move the EVIDENCE, not the caption -- the failure mode that
    # actually happens is a document going stale while the numbers improve.
    moved = copy.deepcopy(scripted)
    ep0 = moved["episodes"][0]["task"]["subgoals"]
    flipped_goal = SUBGOAL_ORDER[0]
    ep0[flipped_goal] = not ep0[flipped_goal]          # negate, never set
    bad, detail = figures_match(meta, moved, control)
    ok &= check("figures.reject_stale_after_evidence_moves", not bad,
                "flipping %s on seed %d in the evaluation is caught: %s"
                % (flipped_goal, moved["episodes"][0]["seed"], detail))

    # --------------------------------------------------------- honesty
    good, detail = honesty_holds(meta, scripted)
    ok &= check("honesty.accept", good, detail)

    s = tallies(scripted)
    inflated = copy.deepcopy(meta)
    for line in inflated["caption_lines"]:
        if line["role"] == "absent":
            line["text"] = line["text"].replace(
                "task_success is %d/%d" % (s["task_success"], s["seeds"]),
                "task_success is %d/%d" % (s["seeds"], s["seeds"]))
    bad, detail = honesty_holds(inflated, scripted)
    ok &= check("honesty.reject_inflated_success", not bad,
                "a cover claiming %d/%d success is refused: %s"
                % (s["seeds"], s["seeds"], detail))

    stripped = copy.deepcopy(meta)
    never = [g for g in SUBGOAL_ORDER if not s["per_goal"][g]]
    for line in stripped["caption_lines"]:
        if line["role"] == "absent":
            # when every sub-goal has fired at least once there is no
            # never-earned list to delete, so delete the disclosure instead --
            # the corruption follows the evidence rather than assuming a gap
            line["text"] = (line["text"].replace("No learned policy", "")
                            if not never else
                            "".join(line["text"].replace(g, "") for g in never[:1]))
            for g in never[1:]:
                line["text"] = line["text"].replace(g, "")
    bad, detail = honesty_holds(stripped, scripted)
    ok &= check("honesty.reject_deleted_absence_clause", not bad,
                "deleting the %s is caught: %s"
                % ("never-earned list" if never else "no-policy disclosure", detail))

    # ------------------------------------------------------ provenance
    eps = {e["seed"]: e for e in scripted["episodes"]}
    best = max(e["task"]["subgoals_met"] for e in eps.values())
    want_seed = min(sd for sd, e in eps.items() if e["task"]["subgoals_met"] == best)
    rec = {g: bool(eps[meta["source_seed"]]["task"]["subgoals"][g]) for g in SUBGOAL_ORDER}
    live = {g: bool(meta["cross_check"]["live"][g]) for g in SUBGOAL_ORDER}
    ok &= check("provenance.accept",
                meta["source_seed"] == want_seed and live == rec
                and meta["cross_check"]["match"] is True,
                "seed %d is the choice rule's answer and its live sub-goals "
                "equal the recorded episode" % meta["source_seed"])

    disagree = copy.deepcopy(meta)
    g0 = SUBGOAL_ORDER[0]
    disagree["cross_check"]["live"][g0] = not disagree["cross_check"]["live"][g0]
    live2 = {g: bool(disagree["cross_check"]["live"][g]) for g in SUBGOAL_ORDER}
    ok &= check("provenance.reject_frame_disagreeing_with_evaluation",
                live2 != rec,
                "a frame claiming a sub-goal the evaluation did not record is caught")

    # ----------------------------------------------------------- pixels
    rows = pixels(raw, head)
    band_top = int(meta["caption_band_top_px"])     # declared by the renderer
    scene = band_stats(rows, 60, band_top - 20)
    caption = band_stats(rows, band_top + 30, head["height"] - 10)
    ok &= check("pixels.accept_real_render",
                scene["std"] > 15 and scene["distinct_colours"] > 200
                and caption["mean"] < 70,
                "render region std %.1f over %d distinct colours; caption band "
                "mean luma %.1f" % (scene["std"], scene["distinct_colours"],
                                    caption["mean"]))

    flat = [bytearray(bytes([128, 128, 128]) * head["width"]) for _ in range(20)]
    flat_stats = band_stats(flat, 0, 20, step=3)
    ok &= check("pixels.reject_flat_placeholder",
                not (flat_stats["std"] > 15 and flat_stats["distinct_colours"] > 200),
                "a uniform image of the same width is refused: std %.1f, %d "
                "distinct colours" % (flat_stats["std"], flat_stats["distinct_colours"]))

    # ------------------------------------------------------------ title
    h1 = next(l[2:].strip() for l in (ROOT / "README.md").read_text(
        encoding="utf-8").splitlines() if l.startswith("# "))
    ok &= check("title.accept_matches_readme", role(meta, "title") == h1,
                "cover title is the repository H1: %r" % h1)

    out = EV / "cover_image_controls.json"
    out.write_text(json.dumps({
        "image": meta["image"],
        "sha256": meta["sha256"],
        "ihdr": head,
        "pixel_stats": {"render_region": scene, "caption_band": caption},
        "controls": results,
        "passed": sum(r["pass"] for r in results),
        "total": len(results),
        "all_pass": bool(ok),
    }, indent=1) + "\n", encoding="utf-8")

    print("\n%d/%d controls passed -> %s"
          % (sum(r["pass"] for r in results), len(results), out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
