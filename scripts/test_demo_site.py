"""Controls for the shipped demo site (``index.html``).

Every figure is re-derived here from ``evidence/*.json`` *without* calling the
builder, then looked for in the bytes of the shipped page.  Each presence check
is paired with a negative control: the same search re-run against a corrupted
value, which must NOT be found -- otherwise the check cannot fail and proves
nothing.

Run:  PYTHONPATH=. python3 scripts/test_demo_site.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
PAGE = ROOT / "index.html"

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((bool(ok), name, detail))


def load(name: str) -> dict:
    with open(EV / name) as fh:
        return json.load(fh)


def corrupt(text: str) -> str:
    """Change every digit run in ``text`` so the result states something false."""
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 7), text, count=1)


def present(page: str, needle: str, name: str) -> None:
    """Presence check plus the negative control that makes it able to fail."""
    bad = corrupt(needle)
    if bad == needle:
        check(False, name, "negative control is VOID: %r has no digit to corrupt" % needle)
        return
    check(needle in page, name, "expected %r" % needle)
    check(bad not in page, name + " [negative control]", "corrupted %r must be absent" % bad)


def main() -> int:
    page = PAGE.read_text(encoding="utf-8")

    priv = load("eval_seeds_scripted.json")
    perc = load("eval_seeds_scripted_perceived.json")
    blind = load("eval_seeds_scripted_blind.json")
    demo = load("demo_scripted_10seeds.json")
    pres = load("video_presentation.json")
    slides = load("slides_presentation.json")

    goals = ("drawer_open", "fork_placed", "spoon_placed", "plate_placed", "mug_placed")
    n = len(priv["episodes"])

    # --- 1. the three scene-source conditions ------------------------------
    for label, doc in (("privileged", priv), ("perceived", perc), ("blind", blind)):
        total = sum(doc["subgoals_met_per_seed"])
        denom = len(doc["subgoals_met_per_seed"]) * len(goals)
        present(page, "%d / %d" % (total, denom), "condition %s scores %d/%d"
                % (label, total, denom))
        check("--scene %s" % label in page, "condition %s is named on the page" % label)

    # the ordering claim the page makes in prose must hold in the data
    check(sum(blind["subgoals_met_per_seed"]) < sum(perc["subgoals_met_per_seed"])
          < sum(priv["subgoals_met_per_seed"]),
          "blind < perceived < privileged holds in the evidence",
          "%d < %d < %d" % (sum(blind["subgoals_met_per_seed"]),
                            sum(perc["subgoals_met_per_seed"]),
                            sum(priv["subgoals_met_per_seed"])))

    # --- 2. per-sub-goal placement table -----------------------------------
    for g in goals:
        k = sum(1 for ep in priv["episodes"] if ep["task"]["subgoals"][g])
        present(page, "%d / %d seeds" % (k, n), "placement %s = %d/%d" % (g, k, n))

    # --- 3. the interactive per-seed data block ----------------------------
    m = re.search(r'<script type="application/json" id="run-data">(.*?)</script>',
                  page, re.S)
    check(m is not None, "page carries an embedded run-data block")
    if m:
        embedded = json.loads(m.group(1))
        eps = {e["seed"]: e for e in embedded["episodes"]}
        check(sorted(eps) == sorted(ep["seed"] for ep in priv["episodes"]),
              "run-data covers exactly the evaluated seeds", str(sorted(eps)))
        mismatched = []
        for ep in priv["episodes"]:
            got, want = eps.get(ep["seed"], {}), ep["task"]
            if ({g: bool(want["subgoals"][g]) for g in goals} != got.get("subgoals")
                    or got.get("met") != want["subgoals_met"]
                    or got.get("success") != bool(want["task_success"])
                    or got.get("bimanual") != bool(want["bimanual"])):
                mismatched.append(ep["seed"])
        check(not mismatched, "every seed's outcome matches envs/task.py's score",
              "mismatched seeds: %s" % mismatched)
        # negative control: the same comparison against a displaced copy must fail
        spoiled = json.loads(json.dumps(embedded))
        spoiled["episodes"][0]["met"] += 1
        bad = [ep["seed"] for ep in priv["episodes"]
               if {e["seed"]: e for e in spoiled["episodes"]}[ep["seed"]]["met"]
               != ep["task"]["subgoals_met"]]
        check(bad == [priv["episodes"][0]["seed"]],
              "seed-outcome comparison [negative control]",
              "a displaced count must be caught; caught %s" % bad)

    # --- 4. media figures ---------------------------------------------------
    secs = round(int(demo["frames"]) / int(demo["fps"]), 1)
    present(page, "%d frames at %d fps" % (demo["frames"], demo["fps"]),
            "demo video frame count and fps")
    present(page, "%s s at" % secs, "demo video duration")
    present(page, pres["duration_mmss"], "video presentation duration")
    present(page, "%d pages" % slides["pages"], "slide deck page count")

    # --- 5. every local asset the page points at exists ---------------------
    refs = set(re.findall(r'(?:src|href)="((?!https?:|#)[^"]+)"', page))
    refs |= set(re.findall(r'"(evidence/seeds/seed_\d+\.png)"', page))
    missing = sorted(r for r in refs if not (ROOT / r).exists())
    check(not missing, "every local asset referenced by the page exists",
          "missing: %s" % missing)
    check(len(refs) >= 4, "the page references local assets at all", str(sorted(refs)))
    check(not (ROOT / "evidence/seeds/seed_99.png").exists(),
          "asset existence check [negative control]",
          "a path that should not exist must not resolve")

    # --- 6. the absence ledger is not silently contradicted -----------------
    success = priv["task_success_count"]
    check(success == 0, "task success is still 0 in the evidence", str(success))
    check("Task success is %d of %d seeds" % (success, n) in page,
          "the page states the task-success count in its ledger")
    check("no learned policy" in page,
          "the page states that the repository holds no learned policy")
    check("NOT_THE_REQUIRED_MEASUREMENT" in page,
          "the page states the OpenVINO numbers are not the required measurement")
    for word in ("Streamlit", "Replit", "Vercel"):
        check(word not in page, "the page does not claim to be hosted on %s" % word)

    # --- 7. a fresh build is byte-identical ---------------------------------
    before = hashlib.sha256(PAGE.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as td:
        keep = pathlib.Path(td) / "index.html"
        keep.write_bytes(PAGE.read_bytes())
        r = subprocess.run([sys.executable, "scripts/make_demo_site.py"], cwd=ROOT,
                           capture_output=True, text=True, env={"PYTHONPATH": ".",
                                                                "PATH": "/usr/bin:/bin"})
        after = hashlib.sha256(PAGE.read_bytes()).hexdigest()
        if after != before:
            PAGE.write_bytes(keep.read_bytes())
        check(r.returncode == 0, "the builder re-runs cleanly", r.stderr[-300:])
        check(after == before, "a fresh build is byte-identical to the shipped page",
              "%s vs %s" % (before[:12], after[:12]))

    ledger = load("demo_site.json")
    check(ledger["sha256"] == before, "evidence/demo_site.json pins the shipped bytes",
          "%s vs %s" % (ledger["sha256"][:12], before[:12]))

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    for ok, name, detail in RESULTS:
        if not ok:
            print("FAIL  %s -- %s" % (name, detail))
    print("%d/%d controls passed" % (passed, len(RESULTS)))
    out = {"schema": "demo_site_controls/v1", "passed": passed, "total": len(RESULTS),
           "all_pass": passed == len(RESULTS),
           "controls": [{"name": nm, "ok": ok, "detail": dt} for ok, nm, dt in RESULTS]}
    (EV / "demo_site_controls.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
