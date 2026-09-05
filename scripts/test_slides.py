#!/usr/bin/env python3
"""Controls for evidence/slides_presentation.pdf -- the mandatory slide deck.

A deck is a submission's most quotable artifact and its least checked one: the
numbers on it are read by a judge and by nobody else.  So this suite treats the
SHIPPED PDF BYTES as the thing under test, not the script that made them.

  claims     ACCEPT every figure on the slides is re-derived here, from
             evidence/*.json, and found in the text extracted from the PDF;
             REJECT a corrupted figure, and REJECT the deck going stale when
             the EVIDENCE moves underneath it
  honesty    ACCEPT the deck states every absence the evidence records -- no
             completed task, the three sub-goals that never fire, no object
             hand-off, perception outside the control loop, no Intel Core Ultra
             measurement; REJECT an inflated task-success figure and REJECT the
             absence ledger being deleted
  bytes      ACCEPT the shipped PDF is exactly what scripts/make_slides.py
             produces from today's evidence, is 16:9, and has the page count
             the sidecar claims; REJECT a sidecar whose sha256 does not match
             the file next to it
  layout     ACCEPT the build's own layout check found nothing; REJECT a
             deliberately broken layout going unreported

The claim strings are re-derived from the evidence in THIS file, not imported
from the builder: two implementations that have to agree.  The corruptions are
derived from the artifact rather than pinned to literals, so they keep
corrupting after the measurements move.

  python3 scripts/test_slides.py

Needs pypdf (to read the shipped bytes) and matplotlib (to rebuild).  Both are
in requirements.txt.  A missing one is a FAILURE, not a skip: a control that
quietly does not run is worse than no control.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
PDF = EV / "slides_presentation.pdf"
SIDECAR = EV / "slides_presentation.json"

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


def subgoal_counts(run: dict) -> dict:
    out: dict[str, int] = {}
    for ep in run["episodes"]:
        for goal, met in ep["task"]["subgoals"].items():
            out[goal] = out.get(goal, 0) + bool(met)
    return out


def claims_from_evidence(e: dict) -> list[tuple[str, str]]:
    """Every figure the deck is allowed to print, rendered as it must appear."""
    sc, ctl = e["scripted"], e["control"]
    seeds = sc["seeds"]
    denom = seeds * sc["episodes"][0]["task"]["subgoals_total"]
    per = subgoal_counts(sc)
    total = sum(sc["subgoals_met_per_seed"])
    ctl_total = sum(ctl["subgoals_met_per_seed"])
    bimanual = sum(1 for ep in sc["episodes"] if ep["task"].get("bimanual"))
    obj_handoffs = sum(1 for ep in sc["episodes"]
                       for h in ep["task"]["handoffs"] if h["object"] != "drawer")
    drawer_handoffs = sum(1 for ep in sc["episodes"] if ep["task"]["handoff_occurred"])
    plate = [2000 * ep["randomization"]["dims"]["plate_r"] for ep in sc["episodes"]]
    tr, ex, bn = e["train"], e["export"], e["bench"]
    v = ex["variants"]
    cpu = bn["results"]["CPU/FP32"]
    fp32b, int8b = v["FP32"]["bin_bytes"], v["INT8"]["bin_bytes"]
    int8_mm = v["INT8"]["accuracy_mm"]["eval10"]["worst_centre_mm"]
    fp32_mm = v["FP32"]["accuracy_mm"]["eval10"]["worst_centre_mm"]

    c: list[tuple[str, str]] = [
        ("headline", f"{total} / {denom} sub-goals over {seeds} seeds"),
        ("no_policy_control",
         f"{ctl_total} / {denom} with the arms held in the home pose"),
        ("task_success", f"task success {sc['task_success_count']} / {seeds}"),
        ("scene_checks", f"{sum(k['pass'] for k in e['scene']['checks'])} / "
                         f"{len(e['scene']['checks'])} scene checks"),
        ("scorer_controls",
         f"{sum(k['pass'] for k in e['predicates']['controls'])} / "
         f"{len(e['predicates']['controls'])} scorer controls"),
        ("perception_controls",
         f"{sum(k['pass'] for k in e['perception_controls']['controls'])} / "
         f"{len(e['perception_controls']['controls'])} perception + OpenVINO "
         f"controls"),
        ("in_order", f"in-order prefix "
                     f"{max(ep['task']['in_order_prefix'] for ep in sc['episodes'])}"
                     f" of {sc['episodes'][0]['task']['subgoals_total']} on every "
                     f"seed"),
        ("object_handoffs", f"object hand-offs {obj_handoffs} / {seeds}"),
        ("dropped", f"objects dropped "
                    f"{sum(len(ep['task']['objects_dropped']) for ep in sc['episodes'])}"),
        ("bimanual", f"both arms touched an object on {bimanual} / {seeds} seeds"),
        ("drawer_handoffs", f"all {drawer_handoffs} recorded hand-offs are the two "
                            f"arms taking the drawer handle in turn"),
        ("plate_span", f"{min(plate):.0f}–{max(plate):.0f} mm across the ten "
                       f"evaluation seeds"),
        ("scene_detail", next(k["detail"] for k in e["scene"]["checks"]
                              if k["check"] == "standalone_load")),
        ("instruction", sc["episodes"][0]["task"]["instruction"]),
        ("params", f"{tr['model']['parameters']:,} parameters"),
        ("dataset", f"{tr['data']['train']:,} training frames, "
                    f"{tr['data']['val']} validation, {tr['data']['eval10']} "
                    f"evaluation"),
        ("epochs", f"{tr['training']['epochs']} epochs"),
        ("perception_error",
         f"{tr['error_mm']['val']['worst_centre_mm']:.2f} mm validation, "
         f"{tr['error_mm']['eval10']['worst_centre_mm']:.2f} mm on the ten "
         f"evaluation seeds"),
        ("perception_baselines",
         f"no-vision baseline "
         f"{tr['controls']['constant_baseline_val']['worst_centre_mm']:.2f} mm"),
        ("shuffled",
         f"shuffled labels "
         f"{tr['controls']['shuffled_labels_val']['worst_centre_mm']:.2f} mm"),
        ("int8_ratio", f"INT8 is {fp32b / int8b:.1f}× smaller than FP32 and costs "
                       f"{int8_mm - fp32_mm:.2f} mm"),
        ("bench_rows", f"{len(bn['results'])} device/precision rows on this host"),
        ("bench_host", bn["host"]["cpu"].strip()),
        ("bench_verdict", bn["required_hardware"]["verdict"]),
        ("bench_asks", f"The track asks for: "
                       f"{bn['required_hardware']['track_asks_for']}"),
        ("bench_numbers",
         f"OpenVINO {bn['host']['openvino'].split('-')[0]}: CPU/FP32 "
         f"{cpu['latency_stream']['p50_ms']:.3f} ms p50, "
         f"{cpu['throughput']['fps']:,.0f} fps async"),
        ("demo_seeds", f"The demo video covers the same {e['demo']['seeds']} seeds"),
        ("demo_sha", f"sha256 {e['demo']['sha256'][:12]}"),
        ("ik", f"Worst IK residual over the ten rollouts: "
               f"{max(ep['rollout']['max_ik_err_mm'] for ep in sc['episodes']):.1f} mm "
               f"across {sum(ep['rollout']['moves'] for ep in sc['episodes']):,} "
               f"planned moves"),
    ]
    for goal in sorted(per):
        c.append((f"subgoal_{goal}", f"{goal} {per[goal]} / {seeds}"))
    for name in ("FP32", "FP16", "INT8"):
        var = v[name]
        c.append((f"ir_{name.lower()}",
                  f"{name:<11}{round(var['bin_bytes'] / 1024):>8,} KiB"
                  f"{var['accuracy_mm']['eval10']['worst_centre_mm']:>11.2f} mm"))
    return c


def honesty_claims(e: dict) -> list[tuple[str, str]]:
    """The absences the deck is required to state, derived from the evidence."""
    sc = e["scripted"]
    seeds = sc["seeds"]
    per = subgoal_counts(sc)
    out = [
        ("no_learned_policy",
         "No learned policy: no VLA, no imitation learning, no language "
         "conditioning."),
        ("never_set",
         f"Task success {sc['task_success_count']} / {seeds} — the table has "
         f"never been set."),
        ("not_in_loop",
         "Perception is not in the control loop; the controller reads privileged "
         "simulator state."),
        ("no_intel",
         f"No Intel Core Ultra measurement: "
         f"{e['bench']['required_hardware']['verdict']}."),
        ("plate_dragged", "The plate is dragged by its rim, not carried."),
    ]
    goal_order = list(sc["episodes"][0]["task"]["subgoals"])
    zero = [g for g in goal_order if per[g] == 0]
    if zero:
        out.append(("zero_subgoals",
                    ", ".join(f"{g} {per[g]} / {seeds}" for g in zero) + "."))
    return out


# ----------------------------------------------------------------- the bytes
def pdf_text(path: pathlib.Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return normalise("\n".join(p.extract_text() or "" for p in reader.pages))


def normalise(s: str) -> str:
    """Strip ALL whitespace before comparing.

    A PDF text stream breaks on kerning pairs, so pypdf returns "T ask success"
    for a line that reads "Task success", and a wrapped claim arrives split
    across lines.  Neither is a defect in the deck, and collapsing runs of
    whitespace does not fix the first one.  Comparing whitespace-free strings
    does, and still catches every changed character and digit.
    """
    return re.sub(r"\s+", "", s)


def missing(text: str, claims) -> list[str]:
    return [label for label, s in claims if normalise(s) not in text]


def corrupt(claim: str) -> str:
    """Break a claim the way a stale deck breaks: bump its first number.

    Derived from the claim, not pinned to a literal, so it keeps corrupting
    after the measurements move.  A claim with no number has its last two
    characters transposed -- dropping its last word was tried first and was
    useless, because a prefix of a true sentence is still in the document.
    """
    m = re.search(r"\d+", claim)
    if m:
        return claim[:m.start()] + str(int(m.group()) + 1) + claim[m.end():]
    body = claim.rstrip()
    for i in range(len(body) - 1, 0, -1):
        if body[i] != body[i - 1] and not body[i].isspace() and not body[i - 1].isspace():
            return body[:i - 1] + body[i] + body[i - 1] + body[i + 1:]
    return body + "X"


def main() -> int:
    for mod in ("pypdf", "matplotlib"):
        try:
            __import__(mod)
        except ImportError as exc:
            check(f"dependency_{mod}", False, f"{exc} -- controls cannot run")
            print(json.dumps({"all_pass": False}, indent=1))
            return 1

    if not PDF.exists() or not SIDECAR.exists():
        check("artifacts_exist", False, f"missing {PDF.name} or {SIDECAR.name}")
        return 1

    e = load_evidence(EV)
    claims = claims_from_evidence(e)
    absences = honesty_claims(e)
    text = pdf_text(PDF)
    side = json.loads(SIDECAR.read_text(encoding="utf-8"))
    ok = True

    # ---------------------------------------------------------------- claims
    gaps = missing(text, claims)
    ok &= check("accept_every_figure_is_in_the_pdf", not gaps,
                f"{len(claims) - len(gaps)} / {len(claims)} claims found"
                + (f"; missing {gaps}" if gaps else ""))

    broken = [(f"{lab}_corrupt", corrupt(s)) for lab, s in claims]
    survived = [lab for lab in
                [b[0] for b in broken if b[0] not in missing(text, broken)]]
    ok &= check("reject_a_corrupted_figure", not survived,
                f"{len(broken)} corrupted claims, {len(broken) - len(survived)} "
                f"correctly absent from the PDF"
                + (f"; still found {survived}" if survived else ""))

    # the deck must go stale when the EVIDENCE moves, not only when the deck does
    moved = json.loads(json.dumps(e))
    moved["scripted"]["episodes"][0]["task"]["subgoals"]["mug_placed"] = True
    moved["scripted"]["subgoals_met_per_seed"][0] += 1
    moved_claims = claims_from_evidence(moved)
    stale = missing(text, moved_claims)
    ok &= check("reject_the_pdf_when_the_evidence_moves", bool(stale),
                f"one extra mug placement in the evidence makes {len(stale)} "
                f"claim(s) stop matching the shipped PDF: {stale[:4]}")

    # ------------------------------------------------------------- honesty
    hgaps = missing(text, absences)
    ok &= check("accept_every_absence_is_stated", not hgaps,
                f"{len(absences) - len(hgaps)} / {len(absences)} absences stated"
                + (f"; missing {hgaps}" if hgaps else ""))

    seeds = e["scripted"]["seeds"]
    inflated = [(f"inflated_{n}", f"task success {n} / {seeds}")
                for n in range(1, seeds + 1)]
    found = [lab for lab, s in inflated if normalise(s) in text]
    ok &= check("reject_an_inflated_task_success", not found,
                f"no inflated task-success string among {len(inflated)} tested"
                if not found else f"deck claims {found}")

    scrubbed = text
    for _, s in absences:
        scrubbed = scrubbed.replace(normalise(s), "")
    ok &= check("reject_a_deck_with_the_absences_deleted",
                len(missing(scrubbed, absences)) == len(absences),
                f"with the ledger removed, all {len(absences)} absence controls "
                f"report missing")

    # --------------------------------------------------------------- bytes
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    ok &= check("accept_sidecar_matches_the_file", digest == side["sha256"],
                f"sha256 {digest[:16]}… == sidecar {side['sha256'][:16]}…"
                if digest == side["sha256"] else
                f"sha256 {digest[:16]}… != sidecar {side['sha256'][:16]}…")
    ok &= check("reject_a_sidecar_sha_that_does_not_match",
                corrupt(side["sha256"]) != digest,
                "a one-character change to the recorded sha256 stops matching")

    from pypdf import PdfReader
    reader = PdfReader(str(PDF))
    box = reader.pages[0].mediabox
    ratio = float(box.width) / float(box.height)
    ok &= check("accept_pages_and_aspect_ratio",
                len(reader.pages) == side["pages"] and abs(ratio - 16 / 9) < 0.01,
                f"{len(reader.pages)} pages (sidecar says {side['pages']}), "
                f"{float(box.width):.0f}x{float(box.height):.0f} pt, "
                f"ratio {ratio:.4f}")

    # the strongest link: the shipped bytes ARE this builder's output today
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_slides                                          # noqa: E402
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = pathlib.Path(tmp) / "rebuild.pdf"
        side2 = make_slides.build(rebuilt, pathlib.Path(tmp) / "rebuild.json")
        same = hashlib.sha256(rebuilt.read_bytes()).hexdigest() == digest
        ok &= check("accept_shipped_bytes_are_the_builder_output_today", same,
                    "a fresh build from today's evidence is byte-identical"
                    if same else
                    "a fresh build DIFFERS from the shipped PDF -- the deck is "
                    "stale or was hand-edited")
        ok &= check("accept_layout_check_is_clean",
                    not side2["layout_violations"],
                    f"{len(side2['layout_violations'])} layout violations on the "
                    f"rebuild")

        # and the layout check itself has to be able to fail
        broken_title = "A deliberately overlong slide title used only as a negative control for the layout checker"
        fired = len(make_slides.wrap(__import__("matplotlib.pyplot",
                                                fromlist=["figure"])
                                     .figure(figsize=make_slides.FIGSIZE),
                                     broken_title, 29,
                                     make_slides.R - make_slides.L,
                                     weight="bold")) > 1
        ok &= check("reject_a_title_that_would_collide_with_the_rule", fired,
                    "the builder's title-wrap detector fires on an overlong title")

    (EV / "slides_controls.json").write_text(json.dumps({
        "schema": "slides_controls/v1",
        "artifact": "evidence/slides_presentation.pdf",
        "sha256": digest,
        "claims_checked": len(claims),
        "absences_checked": len(absences),
        "all_pass": bool(ok),
        "controls": results,
    }, indent=1) + "\n", encoding="utf-8")

    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} controls pass")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
