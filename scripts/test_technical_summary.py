#!/usr/bin/env python3
"""Controls for TECHNICAL_SUMMARY.md -- Required Deliverable 5.

Deliverable 5 asks for an architecture summary whose claims a judge can check.
A prose document is the easiest place in a repository for a number to go stale,
so this suite binds the document to the JSON in evidence/ and drives every
mechanism from both sides -- a check that can only pass proves nothing.

  claims    ACCEPT every figure quoted in the document is re-derived from
            evidence/ and found verbatim; REJECT a single corrupted figure, and
            REJECT the document going stale when the EVIDENCE moves instead
  topics    ACCEPT all seven Deliverable 5 topics have a section; REJECT any
            one of them being removed
  honesty   ACCEPT the document states every absence the evidence records --
            no completed task, sub-goals that never fired, no object hand-off,
            perception outside the control loop, no Intel measurement;
            REJECT an inflated task-success figure, and REJECT the absence
            ledger being deleted

The corruptions are derived from the artifact rather than pinned to a literal,
so they keep corrupting after the numbers move.

Standard library only, so it runs on a fresh clone before anything is
installed.  Run:  python3 scripts/test_technical_summary.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
DOC = ROOT / "TECHNICAL_SUMMARY.md"

TOPICS = [
    ("solution architecture", "## 1. Solution architecture"),
    ("VLA/VLM model choice", "## 2. VLA / VLM model choice"),
    ("bimanual coordination strategy", "## 3. Bimanual coordination strategy"),
    ("training approach", "## 4. Training approach"),
    ("robustness methods", "## 5. Robustness methods"),
    ("OpenVINO optimization", "## 6. OpenVINO optimization"),
    ("Intel hardware mapping", "## 7. Intel hardware mapping"),
]

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


# ------------------------------------------------------------------ evidence
def load_evidence(ev: pathlib.Path) -> dict:
    def j(name):
        return json.loads((ev / name).read_text(encoding="utf-8"))

    def opt(name):
        f = ev / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    bench = sorted(ev.glob("openvino_bench_*.json"))
    if not bench:
        raise FileNotFoundError(f"no openvino_bench_*.json under {ev}")
    return {
        # Optional because the perceived and blind runs are a second pass over
        # the same seeds; a clone that has only run the headline evaluation
        # still gets a document check, it just gets fewer claims.
        "scripted_perceived": opt("eval_seeds_scripted_perceived.json"),
        "scripted_blind": opt("eval_seeds_scripted_blind.json"),
        "scene_controls": opt("scene_source_controls.json"),
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
    """Every figure the document is allowed to quote, rendered as it must appear."""
    sc, ctl = e["scripted"], e["control"]
    seeds = sc["seeds"]
    per = subgoal_counts(sc)
    total = sum(sc["subgoals_met_per_seed"])
    denom = seeds * sc["episodes"][0]["task"]["subgoals_total"]
    ctl_total = sum(ctl["subgoals_met_per_seed"])
    bimanual = sum(1 for ep in sc["episodes"] if ep["task"].get("bimanual"))

    tr = e["train"]
    ex = e["export"]
    bn = e["bench"]
    v = ex["variants"]
    fp32b, int8b = v["FP32"]["bin_bytes"], v["INT8"]["bin_bytes"]
    cpu32 = bn["results"]["CPU/FP32"]
    ab = ex["precision_hint_ab"]

    c: list[tuple[str, str]] = [
        ("subgoals_total", f"**{total} / {denom}** sub-goals over {seeds} seeds"),
        ("task_success", f"task success **{sc['task_success_count']} / {seeds}**"),
        ("control_total", f"| total | **{total} / {denom}** | **{ctl_total} / {denom}** |"),
        ("bimanual", f"object on **{bimanual} / {seeds}** seeds"),
        ("scene_checks",
         f"**{sum(k['pass'] for k in e['scene']['checks'])} / "
         f"{len(e['scene']['checks'])}** structural"),
        ("predicate_controls",
         f"**{sum(k['pass'] for k in e['predicates']['controls'])} / "
         f"{len(e['predicates']['controls'])}** accept *and* reject"),
        ("perception_controls",
         f"**{sum(k['pass'] for k in e['perception_controls']['controls'])} / "
         f"{len(e['perception_controls']['controls'])}**"),
        ("params", f"**{tr['model']['parameters']:,}** parameters"),
        ("dataset",
         f"**{tr['data']['train']:,}** training frames, **{tr['data']['val']}** "
         f"validation, **{tr['data']['eval10']}** evaluation"),
        ("epochs", f"**{tr['training']['epochs']}** epochs, batch {tr['training']['batch']}"),
        ("perception_val_eval",
         f"**{tr['error_mm']['val']['worst_centre_mm']:.2f} mm** on unseen validation "
         f"seeds and **{tr['error_mm']['eval10']['worst_centre_mm']:.2f} mm** on"),
        ("perception_baseline",
         f"**{tr['controls']['constant_baseline_val']['worst_centre_mm']:.2f} mm** "
         f"for the no-vision baseline"),
        ("int8_ratio", f"INT8 is {fp32b / int8b:.1f}× smaller than FP32"),
        ("int8_cost",
         f"costs {v['INT8']['accuracy_mm']['eval10']['worst_centre_mm'] - v['FP32']['accuracy_mm']['eval10']['worst_centre_mm']:.1f} mm of accuracy"),
        ("hint_default", f"| plugin default (`bfloat16` on this host) | "
                         f"**{ab['plugin_default']['eval10_max_abs_vs_torch_mm']:.4f} mm** |"),
        ("hint_forced", f"**{ab['forced_f32']['eval10_max_abs_vs_torch_mm']:.5f} mm**"),
        ("bench_rows", f"**{len(bn['results'])}** device/precision rows"),
        ("bench_host", bn["host"]["cpu"]),
        ("bench_verdict", bn["required_hardware"]["verdict"]),
        ("bench_cpu_latency", f"{cpu32['latency_stream']['p50_ms']:.3f} ms p50"),
        ("bench_cpu_throughput", f"{cpu32['throughput']['fps']:.0f} fps async"),
    ]
    if e.get("scene_controls"):
        sco = e["scene_controls"]
        c.append(("scene_source_controls",
                  f"**{sco['passed']} / {sco['total']}**"))
    pv, bl = e.get("scripted_perceived"), e.get("scripted_blind")
    if pv:
        pv_total = sum(pv["subgoals_met_per_seed"])
        err = pv["estimate_error_mm"]
        c.append(("perceived_total",
                  f"**{pv_total} / {denom}** sub-goals with perception in the "
                  f"loop"))
        c.append(("perceived_error",
                  f"**{err['at_t0_worst_object_mean']:.2f} mm** at t=0 and "
                  f"**{err['worst_object_mean_over_seeds']:.2f} mm** averaged "
                  f"over every planning instant"))
        c.append(("perceived_inferences",
                  f"**{err['inferences_total']:,}** inferences"))
    if bl:
        c.append(("blind_total",
                  f"**{sum(bl['subgoals_met_per_seed'])} / {denom}** for the "
                  f"blind negative control"))
    for goal, n in sorted(per.items()):
        c.append((f"subgoal_{goal}", f"| `{goal}` | **{n} / {seeds}** |"))
    for name in ("FP32", "FP16", "INT8"):
        var = v[name]
        c.append((f"ir_{name.lower()}",
                  f"**{round(var['bin_bytes'] / 1024)} KiB** | "
                  f"**{var['accuracy_mm']['eval10']['worst_centre_mm']:.2f} mm** | "
                  f"**{var['vs_torch_mm']['eval10_max_abs']:.2f} mm** |"))
    return c


# ------------------------------------------------------------------ checkers
def corrupt_claim(claim: str) -> str:
    """Break one claim the way a stale document breaks: bump its first number.

    Derived from the claim itself rather than pinned to a literal, so it keeps
    corrupting after the measurements move.  A claim with no number in it is
    corrupted by dropping its last character.
    """
    m = re.search(r"\d+", claim)
    if not m:
        return claim[:-1]
    return claim[:m.start()] + str(int(m.group()) + 1) + claim[m.end():]


def demo_prose_gaps(demo: dict) -> list[str]:
    """The demo record's `not_claimed` sentence, re-derived from its own rows.

    Independent of scripts/record_demo.py on purpose -- two implementations that
    have to agree.  This exact sentence shipped stale once: it said the only
    sub-goal earned was the drawer, months after the plate started landing.
    """
    rows = demo["episodes"]
    n = len(rows)
    counts: dict[str, int] = {}
    for r in rows:
        for goal, met in r["task"]["subgoals"].items():
            counts[goal] = counts.get(goal, 0) + bool(met)
    said = demo.get("not_claimed", "")
    gaps = []
    wins = sum(1 for r in rows if r["task"]["task_success"])
    if f"task_success is {wins}/{n}" not in said:
        gaps.append(f"task_success {wins}/{n} not stated")
    for goal, c in sorted(counts.items()):
        token = f"{goal} {c}/{n}"
        if c and token not in said:
            gaps.append(f"earned but unstated: {token}")
        if not c and goal not in said:
            gaps.append(f"never earned and unnamed: {goal}")
    return gaps


def missing_claims(text: str, claims) -> list[str]:
    return [label for label, s in claims if s not in text]


def missing_topics(text: str) -> list[str]:
    return [name for name, heading in TOPICS if heading not in text]


def inflated_task_success(text: str, measured: int, seeds: int) -> list[str]:
    """Any 'task success' line quoting more successes than were measured."""
    bad = []
    for line in text.splitlines():
        if "task success" not in line.lower():
            continue
        for n in re.findall(rf"(\d+)\s*/\s*{seeds}\b", line):
            if int(n) > measured:
                bad.append(line.strip())
    return bad


def missing_absences(text: str, e: dict) -> list[str]:
    """Absences the evidence records that the document must state."""
    low = text.lower()
    sc = e["scripted"]
    per = subgoal_counts(sc)
    object_handoffs = sum(
        1 for ep in sc["episodes"] for h in ep["task"]["handoffs"]
        if h["object"] != "drawer")
    gaps = []
    if sc["task_success_count"] == 0 and "never been completed" not in low:
        gaps.append("task never completed")
    if any(n == 0 for n in per.values()) and "have never fired" not in low:
        gaps.append("sub-goals that never fired")
    if object_handoffs == 0 and "no object hand-off has ever occurred" not in low:
        gaps.append("no object hand-off")
    # Derived from the run, not asserted: which scene source produced the
    # headline number decides what the document has to admit about it. When
    # this was an unconditional string check it was a test asserting a
    # limitation, and it would have gone on passing after the limitation was
    # lifted -- and failing once the document told the truth.
    if sc.get("scene_source", "privileged") == "privileged" \
            and "reading privileged" not in low:
        gaps.append("headline result is produced from privileged poses")
    pv = e.get("scripted_perceived")
    if pv is not None:
        err = pv["estimate_error_mm"]
        drift = err["worst_object_mean_over_seeds"] / max(
            err["at_t0_worst_object_mean"], 1e-9)
        if drift > 5.0 and "occlu" not in low:
            gaps.append(f"perception drifts {drift:.0f}x mid-rollout and the "
                        "document does not mention occlusion")
        if sum(pv["subgoals_met_per_seed"]) < sum(sc["subgoals_met_per_seed"]) \
                and "costs" not in low:
            gaps.append("perception in the loop costs sub-goals and the "
                        "document does not say so")
    if (e["bench"]["required_hardware"]["verdict"] != "MEASURED_ON_REQUIRED_HARDWARE"
            and "has no intel silicon" not in low):
        gaps.append("no Intel Core Ultra measurement")
    return gaps


def main() -> int:
    if not DOC.exists():
        print(f"[FAIL] TECHNICAL_SUMMARY.md missing at {DOC}")
        return 1
    text = DOC.read_text(encoding="utf-8")
    e = load_evidence(EV)
    claims = claims_from_evidence(e)
    seeds = e["scripted"]["seeds"]
    successes = e["scripted"]["task_success_count"]
    ok = True

    # --- accept -----------------------------------------------------------
    miss = missing_claims(text, claims)
    ok &= check("accept_every_quoted_figure_is_re_derived_from_evidence",
                not miss,
                f"{len(claims) - len(miss)}/{len(claims)} claims found verbatim"
                + (f"; missing {miss}" if miss else ""))
    ok &= check("accept_all_seven_deliverable5_topics_present",
                not missing_topics(text),
                f"{len(TOPICS) - len(missing_topics(text))}/{len(TOPICS)} topics")
    over = inflated_task_success(text, successes, seeds)
    ok &= check("accept_task_success_is_not_inflated", not over,
                f"no line claims more than {successes}/{seeds}" if not over
                else f"{len(over)} line(s) claim more than {successes}/{seeds}: {over}")
    gaps = missing_absences(text, e)
    ok &= check("accept_absence_ledger_states_every_recorded_absence", not gaps,
                "every absence the evidence records is stated" if not gaps
                else f"not stated: {gaps}")

    dgaps = demo_prose_gaps(e["demo"])
    ok &= check("accept_demo_record_prose_matches_its_own_rows", not dgaps,
                "evidence/demo_scripted_10seeds.json not_claimed re-derives"
                if not dgaps else f"gaps: {dgaps}")

    # --- reject -----------------------------------------------------------
    uncaught = []
    for label, claim in claims:
        corrupt = text.replace(claim, corrupt_claim(claim))
        if corrupt == text:
            uncaught.append((label, "claim string is not in the document"))
        elif missing_claims(corrupt, claims) != [label]:
            uncaught.append((label, missing_claims(corrupt, claims)))
    ok &= check("reject_any_one_corrupted_figure_passes", not uncaught,
                f"{len(claims) - len(uncaught)}/{len(claims)} claims corrupted one "
                f"at a time, each caught and no other claim disturbed"
                + (f"; {uncaught}" if uncaught else ""))

    caught = []
    for name, heading in TOPICS:
        cut = text.replace(heading + "\n", "")
        caught.append(cut != text and missing_topics(cut) == [name])
    ok &= check("reject_a_removed_topic_passes", all(caught),
                f"{sum(caught)}/{len(TOPICS)} removals caught")

    ts_claim = dict(claims)["task_success"]
    inflated = text.replace(ts_claim, ts_claim.replace(
        f"{successes} / {seeds}", f"{seeds} / {seeds}"))
    ok &= check("reject_an_inflated_task_success_passes",
                inflated != text and bool(
                    inflated_task_success(inflated, successes, seeds)),
                f"'{successes} / {seeds}' -> '{seeds} / {seeds}' is caught")

    idx = text.lower().find("never been completed")
    stripped = text[:idx] + text[idx + len("never been completed"):]
    ok &= check("reject_a_deleted_absence_passes",
                bool(missing_absences(stripped, e)),
                f"deleting the completion statement is caught: "
                f"{missing_absences(stripped, e)}")

    stale_demo = json.loads(json.dumps(e["demo"]))
    stale_demo["not_claimed"] = ("This video does NOT show the task being completed. "
                                 "It shows the scripted controller running on 10 "
                                 "randomized seeds and the live sub-goal state it "
                                 "earns, which is the drawer only.")
    ok &= check("reject_the_stale_demo_sentence_passes",
                bool(demo_prose_gaps(stale_demo)),
                "the literal that actually shipped -- 'the drawer only' -- is "
                f"caught: {demo_prose_gaps(stale_demo)}")

    moved = json.loads(json.dumps(e["scripted"]))
    moved["task_success_count"] = successes + 1
    stale = dict(e, scripted=moved)
    ok &= check("reject_document_going_stale_when_the_evidence_moves",
                "task_success" in missing_claims(text, claims_from_evidence(stale)),
                f"evidence task_success_count {successes} -> {successes + 1} "
                f"makes this document fail")

    EV.mkdir(parents=True, exist_ok=True)
    (EV / "technical_summary_controls.json").write_text(
        json.dumps({"all_pass": bool(ok),
                    "document": DOC.name,
                    "claims_checked": len(claims),
                    "controls": results}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
