"""Generate the demo site served at the submission's Application URL.

The lablab Rule Book asks for a *Demo Application Platform* and an
*Application URL*, and criterion 3 (Application of technology) caps at band 2
-- "Demo link is not available or working" -- when there is no link.  This
entry is a MuJoCo simulation package, not a web app, so the honest thing to
publish is a viewer over the run: the recorded rollout, the per-seed outcome
of every evaluation episode, the three scene-source conditions side by side,
and the ledger of what does not work.

Every number on the page is read out of ``evidence/*.json``.  Nothing is
typed in by hand.  ``scripts/test_demo_site.py`` re-derives each figure
independently and looks for it in the *shipped* ``index.html`` bytes.

Run:  PYTHONPATH=. python3 scripts/make_demo_site.py
"""
from __future__ import annotations

import hashlib
import html
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
OUT = ROOT / "index.html"
LEDGER = EV / "demo_site.json"

SUBGOALS = ("drawer_open", "fork_placed", "spoon_placed", "plate_placed", "mug_placed")


def load(name: str) -> dict:
    with open(EV / name) as fh:
        return json.load(fh)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: the evidence files every figure on the page is read out of
INPUTS = (
    "evidence/eval_seeds_scripted.json",
    "evidence/eval_seeds_scripted_perceived.json",
    "evidence/eval_seeds_scripted_blind.json",
    "evidence/demo_scripted_10seeds.json",
    "evidence/video_presentation.json",
    "evidence/slides_presentation.json",
    "evidence/eval_seeds_act.json",
)


def head() -> str:
    """The commit the page's *inputs* were last changed at.

    Deliberately not ``HEAD``: the page is committed, so stamping HEAD would
    make every fresh build differ from the bytes that were shipped and there
    would be no determinism control to run.  The commit that last moved the
    evidence is the one a reader wants anyway.
    """
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", *INPUTS],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip() or "unknown"
    except Exception:                                          # pragma: no cover
        return "unknown"


def collect() -> dict:
    """Read every figure the page will state, from the evidence files."""
    priv = load("eval_seeds_scripted.json")
    perc = load("eval_seeds_scripted_perceived.json")
    blind = load("eval_seeds_scripted_blind.json")
    demo = load("demo_scripted_10seeds.json")
    pres = load("video_presentation.json")
    slides = load("slides_presentation.json")
    export = load("openvino_export.json")
    act = load("eval_seeds_act.json")

    n_goals = len(SUBGOALS)
    conditions = []
    for label, doc, note in (
        ("privileged", priv, "controller reads MjData, as it always did"),
        ("perceived", perc, "controller reads top_cam -> OpenVINO IR -> seven numbers"),
        ("blind", blind, "controller reads the nominal, un-randomized layout"),
    ):
        per_seed = doc["subgoals_met_per_seed"]
        conditions.append({
            "id": label,
            "note": note,
            "per_seed": per_seed,
            "total": sum(per_seed),
            "denominator": len(per_seed) * n_goals,
            "task_success": doc["task_success_count"],
            "seeds": len(per_seed),
        })

    episodes = []
    for ep in priv["episodes"]:
        task = ep["task"]
        episodes.append({
            "seed": ep["seed"],
            "subgoals": {g: bool(task["subgoals"][g]) for g in SUBGOALS},
            "met": task["subgoals_met"],
            "total": task["subgoals_total"],
            "success": bool(task["task_success"]),
            "bimanual": bool(task["bimanual"]),
            "handoffs": [{"object": h["object"], "from": h["from"],
                          "to": h["to"], "t": round(float(h["t"]), 1)}
                         for h in task.get("handoffs", [])],
            "handoff_objects": sorted({h["object"] for h in task.get("handoffs", [])}),
            "sim_time_s": round(float(task["sim_time_s"]), 1),
            "max_ik_err_mm": round(float(ep["rollout"]["max_ik_err_mm"]), 1),
            "moves": int(ep["rollout"]["moves"]),
            "dropped": list(task.get("objects_dropped", [])),
            "image": "evidence/seeds/seed_%02d.png" % ep["seed"],
        })

    handoff_mug = sorted(e["seed"] for e in episodes if "mug" in e["handoff_objects"])
    bimanual_n = sum(1 for e in episodes if e["bimanual"])
    placed = {g: sum(1 for e in episodes if e["subgoals"][g]) for g in SUBGOALS}

    return {
        "head": head(),
        "instruction": priv["episodes"][0]["task"]["instruction"],
        "conditions": conditions,
        "episodes": episodes,
        "handoff_mug_seeds": handoff_mug,
        "bimanual_n": bimanual_n,
        "placed": placed,
        "seeds": len(episodes),
        "n_goals": n_goals,
        "demo_video": {
            "path": "evidence/demo_scripted_10seeds.mp4",
            "frames": int(demo["frames"]), "fps": int(demo["fps"]),
            "speed": demo["speed"],
            "seconds": round(int(demo["frames"]) / int(demo["fps"]), 1),
            "mb": round((ROOT / "evidence/demo_scripted_10seeds.mp4").stat().st_size / 1e6, 2),
            "wh": "%dx%d" % tuple(demo["size"]),
        },
        "presentation": {
            "path": "evidence/video_presentation.mp4",
            "mmss": pres["duration_mmss"],
            "mb": round(int(pres["bytes"]) / 1e6, 2),
        },
        "slides": {
            "path": "evidence/slides_presentation.pdf",
            "pages": int(slides.get("pages") or slides.get("slide_count") or 0),
            "mb": round(int(slides["bytes"]) / 1e6, 2),
        },
        "precisions": sorted(export["variants"].keys()) if isinstance(
            export.get("variants"), dict) else sorted(
            {v.get("precision") for v in export.get("variants", []) if v.get("precision")}),
        "not_working": [
            "Task success is 0 of %d seeds. No episode completes the whole instruction."
            % len(episodes),
            "The fork and the spoon are placed on %d of %d seeds -- the perception "
            "network has no output for either object." % (placed["fork_placed"], len(episodes)),
            "The mug reaches its mat on %d of %d seeds, and on the seed it does it "
            "arrives lying on its side." % (placed["mug_placed"], len(episodes)),
            "The learned policy loses to the script. A LeRobot ACT policy "
            "behaviour-cloned on the scripted controller's own rollouts scores "
            "%d of %d sub-goals against the scripted controller at %d, on the "
            "same seeds and the same scorer, with more simulator time per "
            "episode. "
            "Everything shown on this page is the scripted controller."
            % (sum(act["subgoals_met_per_seed"]),
               5 * len(act["subgoals_met_per_seed"]),
               sum(priv["subgoals_met_per_seed"])),
            "No measurement was taken on Intel Core Ultra silicon. The OpenVINO "
            "numbers in the repository were produced on an AMD EPYC host and the "
            "bench script stamps them NOT_THE_REQUIRED_MEASUREMENT.",
            "Natural-language interpretation, multi-step task context and "
            "re-planning are absent. The instruction is fixed and is not parsed.",
        ],
    }


CSS = """
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--no:#f85149;--card:#161b22;--line:#30363d;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--dim);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
video{width:100%;border-radius:6px;background:#000}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
th{color:var(--dim);font-weight:600}
.seedbar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.seedbar button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 13px;cursor:pointer;font:inherit}
.seedbar button[aria-pressed=true]{border-color:var(--acc);color:var(--acc);background:#132135}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.grid img{width:100%;border-radius:6px;border:1px solid var(--line)}
.yes{color:var(--ok)} .no{color:var(--no)}
ul.ledger li{margin:6px 0}
code{background:#1f2630;padding:1px 5px;border-radius:4px;font-size:13px}
a{color:var(--acc)}
.kv{display:flex;gap:22px;flex-wrap:wrap;color:var(--dim);font-size:14px}
.warn{border-left:3px solid var(--no);padding-left:12px}
"""

JS = """
const D = JSON.parse(document.getElementById('run-data').textContent);
const GOALS = %(goals)s;
function pick(seed){
  const ep = D.episodes.find(e => e.seed === seed);
  document.querySelectorAll('.seedbar button').forEach(b =>
    b.setAttribute('aria-pressed', String(Number(b.dataset.seed) === seed)));
  document.getElementById('seed-img').src = ep.image;
  document.getElementById('seed-img').alt = 'Initial scene, seed ' + seed;
  const rows = GOALS.map(g =>
    '<tr><td>' + g + '</td><td class="' + (ep.subgoals[g] ? 'yes">met' : 'no">not met') +
    '</td></tr>').join('');
  const ho = ep.handoffs.length
    ? ep.handoffs.map(h => h.object + ' ' + h.from + '\\u2192' + h.to + ' at ' + h.t + ' s').join('; ')
    : 'none';
  document.getElementById('seed-table').innerHTML =
    '<table><tr><th>sub-goal</th><th>seed ' + seed + '</th></tr>' + rows +
    '<tr><td>sub-goals met</td><td>' + ep.met + ' / ' + ep.total + '</td></tr>' +
    '<tr><td>task success</td><td class="' + (ep.success ? 'yes">yes' : 'no">no') + '</td></tr>' +
    '<tr><td>both arms used</td><td class="' + (ep.bimanual ? 'yes">yes' : 'no">no') + '</td></tr>' +
    '<tr><td>hand-offs</td><td>' + ho + '</td></tr>' +
    '<tr><td>waypoints</td><td>' + ep.moves + '</td></tr>' +
    '<tr><td>sim time</td><td>' + ep.sim_time_s + ' s</td></tr>' +
    '<tr><td>peak IK error</td><td>' + ep.max_ik_err_mm + ' mm</td></tr>' +
    '<tr><td>objects dropped</td><td>' + (ep.dropped.length ? ep.dropped.join(', ') : 'none') + '</td></tr>' +
    '</table>';
}
document.querySelectorAll('.seedbar button').forEach(b =>
  b.addEventListener('click', () => pick(Number(b.dataset.seed))));
pick(D.episodes[0].seed);
"""


def render(f: dict) -> str:
    e = html.escape
    seedbtns = "".join('<button data-seed="%d">seed %d</button>' % (ep["seed"], ep["seed"])
                       for ep in f["episodes"])

    cond_rows = "".join(
        "<tr><td><code>--scene %s</code></td><td>%s</td><td>%d / %d</td><td>%d / %d</td></tr>"
        % (e(c["id"]), e(c["note"]), c["total"], c["denominator"],
           c["task_success"], c["seeds"])
        for c in f["conditions"])

    placed_rows = "".join(
        "<tr><td>%s</td><td>%d / %d seeds</td></tr>" % (e(g), f["placed"][g], f["seeds"])
        for g in SUBGOALS)

    ledger = "".join("<li>%s</li>" % e(x) for x in f["not_working"])

    data = json.dumps({"episodes": f["episodes"]}, separators=(",", ":"), sort_keys=True)
    js = JS % {"goals": json.dumps(list(SUBGOALS))}

    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bimanual dinner-table manipulation on SO-101 &mdash; demo</title>
<meta name="description" content="Demo viewer for the AI Infra Summit Bimanual VLA Manipulation entry: recorded rollout, per-seed evaluation, and the ledger of what does not work.">
<style>%(css)s</style></head><body><main>

<h1>Bimanual dinner-table manipulation on SO-101</h1>
<p class="sub">Demo viewer for the AI Infra Summit &mdash; Bimanual VLA Manipulation track.
Every figure on this page is read out of the repository's own evidence files by
<code>scripts/make_demo_site.py</code>; nothing here is typed in by hand.
Built at commit <code>%(head)s</code>.</p>

<div class="card"><div class="kv">
<span><a href="https://github.com/jianwang-ntu/bimanual-dinner-table-so101">source repository</a></span>
<span><a href="%(slides_path)s">slide deck (PDF, %(slides_pages)d pages)</a></span>
<span><a href="%(pres_path)s">video presentation (%(pres_mmss)s)</a></span>
</div></div>

<h2>The task</h2>
<div class="card"><p>%(instruction)s</p>
<p class="sub">Two SO-101 arms, MuJoCo, %(seeds)d randomized evaluation seeds,
%(n_goals)d scored sub-goals per seed.</p></div>

<h2>The rollout, as recorded</h2>
<div class="card">
<video controls preload="metadata" poster="evidence/seeds/seed_00.png">
  <source src="%(demo_path)s" type="video/mp4">
</video>
<p class="sub">%(demo_frames)d frames at %(demo_fps)d fps &mdash; %(demo_seconds)s s at
%(demo_speed)sx real time, %(demo_wh)s, %(demo_mb)s MB. This is the simulator's own
output for the same %(seeds)d seeds scored below.</p>
</div>

<h2>Every evaluation seed</h2>
<p class="sub">Pick a seed. The outcome shown is the one in
<code>evidence/eval_seeds_scripted.json</code>, scored by
<code>envs/task.py</code>.</p>
<div class="seedbar">%(seedbtns)s</div>
<div class="grid">
  <div><img id="seed-img" src="" alt=""></div>
  <div id="seed-table"></div>
</div>

<h2>Where the controller gets its object positions</h2>
<div class="card"><table>
<tr><th>scene source</th><th>what the controller reads</th><th>sub-goals</th><th>task success</th></tr>
%(cond_rows)s
</table>
<p class="sub">The <code>blind</code> row is the negative control: it hands the
controller a fixed nominal layout. If the controller ignored what it was handed,
all three rows would be identical. <code>perceived</code> scores below
<code>privileged</code>, and that is the direction stated here rather than the
other way round.</p></div>

<h2>Which objects are actually placed</h2>
<div class="card"><table>
<tr><th>sub-goal</th><th>met on</th></tr>
%(placed_rows)s
</table>
<p class="sub">Both arms are used on %(bimanual_n)d of %(seeds)d seeds. An object
is handed between the arms on seeds %(handoff_seeds)s.</p></div>

<h2>What does not work</h2>
<div class="card warn"><ul class="ledger">%(ledger)s</ul></div>

<script type="application/json" id="run-data">%(data)s</script>
<script>%(js)s</script>
</main></body></html>
""" % {
        "css": CSS,
        "js": js,
        "data": data,
        "head": e(f["head"]),
        "instruction": e(f["instruction"]),
        "seeds": f["seeds"],
        "n_goals": f["n_goals"],
        "seedbtns": seedbtns,
        "cond_rows": cond_rows,
        "placed_rows": placed_rows,
        "ledger": ledger,
        "bimanual_n": f["bimanual_n"],
        "handoff_seeds": ", ".join(str(s) for s in f["handoff_mug_seeds"]) or "none",
        "demo_path": f["demo_video"]["path"],
        "demo_frames": f["demo_video"]["frames"],
        "demo_fps": f["demo_video"]["fps"],
        "demo_seconds": f["demo_video"]["seconds"],
        "demo_speed": ("%g" % f["demo_video"]["speed"]),
        "demo_wh": f["demo_video"]["wh"],
        "demo_mb": f["demo_video"]["mb"],
        "pres_path": f["presentation"]["path"],
        "pres_mmss": e(f["presentation"]["mmss"]),
        "slides_path": f["slides"]["path"],
        "slides_pages": f["slides"]["pages"],
    }


def main() -> int:
    f = collect()
    OUT.write_text(render(f), encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    ledger = {
        "schema": "demo_site/v1",
        "requirement": "req_application_url / req_demo_platform -- lablab Rule Book "
                       "'Application Components': Demo Application Platform, Application URL",
        "built_by": "scripts/make_demo_site.py",
        "built_at_head": f["head"],
        "page": "index.html",
        "bytes": OUT.stat().st_size,
        "sha256": sha256(OUT),
        "figures_derived_from": list(INPUTS),
        "facts": {
            "seeds": f["seeds"],
            "subgoals_per_seed": f["n_goals"],
            "conditions": {c["id"]: {"total": c["total"], "denominator": c["denominator"],
                                     "task_success": c["task_success"]}
                           for c in f["conditions"]},
            "placed": f["placed"],
            "bimanual_seeds": f["bimanual_n"],
            "mug_handoff_seeds": f["handoff_mug_seeds"],
            "demo_video_seconds": f["demo_video"]["seconds"],
            "presentation_mmss": f["presentation"]["mmss"],
            "slides_pages": f["slides"]["pages"],
        },
        "not_claimed": [
            "The page is a static viewer over recorded evidence. It does not run "
            "the simulator in the browser and does not claim to.",
            "It is hosted on GitHub Pages, which is not one of the three platforms "
            "the Rule Book names (Streamlit, Replit, Vercel). Whether the "
            "submission form accepts it is UNKNOWN and is not resolved in our favour.",
        ],
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print("index.html  %d bytes  sha256 %s" % (ledger["bytes"], ledger["sha256"]))
    print("evidence/demo_site.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
