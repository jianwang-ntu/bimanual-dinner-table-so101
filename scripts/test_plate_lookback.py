#!/usr/bin/env python3
"""Controls for the end-of-episode plate look-back and its own threshold.

``PLATE_TOL`` is a BUILD tolerance -- it drives the drag loop toward the middle
of the mat with margin, and firing it early is free because there is more
episode left to fix a miss.  It was also, until 2026-09-09, the trigger for the
END-OF-EPISODE look-back, where there is no episode left; measured on seed 3
that fired on a plate already inside the scorer's bar and pushed it out.
``PLATE_FINAL_TOL`` splits the two.

Every control here drives the predicate from BOTH sides, because a threshold
that only ever says "yes" is indistinguishable from no threshold at all -- and
the accept path is the one that matters, since suppressing the look-back is
what this change does and a look-back that never fires again would be a
regression, not a fix.

Run:  PYTHONPATH=. python3 scripts/test_plate_lookback.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from envs import controller as C                              # noqa: E402
from envs import scene_source                                 # noqa: E402
from envs.task import PLACEMENTS                              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "envs" / "dinner_table.xml"

results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def load():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    scene_source.install(scene_source.make("privileged"))
    return model, data


def put_plate_at(model, data, dist_m: float):
    """Park the plate ``dist_m`` from ``target_plate`` along +x, on the table."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_plate")
    mujoco.mj_forward(model, data)
    t = data.site_xpos[sid].copy()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = [t[0] + dist_m, t[1], 0.76]
    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def main() -> int:
    ok = True
    model, data = load()

    # ---- the two thresholds are separate, and the final one is the looser ---
    ok &= check("thresholds_are_distinct",
                C.PLATE_FINAL_TOL > C.PLATE_TOL,
                f"PLATE_TOL={C.PLATE_TOL}, PLATE_FINAL_TOL={C.PLATE_FINAL_TOL}")

    # The scorer's own bar, read from the scorer rather than restated here, so
    # this control fails if task.py ever moves the tolerance under us.
    scorer_tol = PLACEMENTS["plate_placed"][2]
    ok &= check("final_threshold_is_inside_the_scorer_bar",
                C.PLATE_FINAL_TOL < scorer_tol,
                f"PLATE_FINAL_TOL={C.PLATE_FINAL_TOL} < scorer {scorer_tol}; "
                f"margin {round((scorer_tol - C.PLATE_FINAL_TOL) * 1000, 1)} mm")

    # ---- REJECT: a plate comfortably on its mat must not trigger a re-drag --
    put_plate_at(model, data, 0.035)
    fired = C._plate_short_final(model, data)
    build = C._plate_short(model, data)
    ok &= check("reject_scoring_plate_does_not_trigger_final_lookback",
                not fired,
                f"plate 35.0 mm out: final look-back fires={fired} (want False)")
    # ...and the BUILD predicate must still fire there, or the split did nothing
    ok &= check("build_predicate_still_fires_at_the_same_distance",
                build,
                f"plate 35.0 mm out: _plate_short fires={build} (want True) -- "
                f"this is the difference the split creates")

    # ---- ACCEPT: a plate outside the scorer's bar must still trigger it -----
    put_plate_at(model, data, 0.060)
    fired = C._plate_short_final(model, data)
    ok &= check("accept_failing_plate_still_triggers_final_lookback",
                fired,
                f"plate 60.0 mm out: final look-back fires={fired} (want True)")

    # A value just inside the new bar and just outside it, to pin the edge
    put_plate_at(model, data, C.PLATE_FINAL_TOL - 0.002)
    below = C._plate_short_final(model, data)
    put_plate_at(model, data, C.PLATE_FINAL_TOL + 0.002)
    above = C._plate_short_final(model, data)
    ok &= check("threshold_edge_is_where_it_says_it_is",
                (not below) and above,
                f"2 mm inside fires={below} (want False), "
                f"2 mm outside fires={above} (want True)")

    # ---- the script wires the FINAL block to the final predicate ------------
    script = C.dinner_table_script()
    preds = [e[1].__name__ for e in script
             if isinstance(e, tuple) and e and e[0] == "if"]
    ok &= check("script_uses_final_predicate_exactly_once",
                preds.count("_plate_short_final") == 1,
                f"conditional predicates in order: {preds}")
    ok &= check("earlier_retries_still_use_the_build_predicate",
                preds.count("_plate_short") == 2,
                f"_plate_short appears {preds.count('_plate_short')} times "
                f"(want 2: the two mid-episode retries)")
    # the final plate look-back must come after both retries, or it is not final
    ok &= check("final_predicate_comes_after_both_retries",
                preds.index("_plate_short_final") > max(
                    i for i, p in enumerate(preds) if p == "_plate_short"),
                f"index {preds.index('_plate_short_final')} vs last "
                f"_plate_short at "
                f"{max(i for i, p in enumerate(preds) if p == '_plate_short')}")

    # ---- the environment override still works, in both directions ----------
    # A knob that cannot be turned back is not a knob, and the A/B that adopted
    # this value depended on being able to set it.
    src = (ROOT / "envs" / "controller.py").read_text()
    ok &= check("threshold_is_env_overridable",
                "os.environ.get('PLATE_FINAL_TOL'" in src,
                "PLATE_FINAL_TOL reads the environment")

    # ---- the adopted evidence says what the adoption claimed ---------------
    res = json.loads((ROOT / "evidence" / "plate_lookback_result.json").read_text())
    ok &= check("adoption_evidence_records_no_losses",
                res["Q1_primary_HELD"]["lost"] == [],
                f"lost={res['Q1_primary_HELD']['lost']}, "
                f"gained={res['Q1_primary_HELD']['gained']}")
    trace = json.loads(
        (ROOT / "evidence" / "plate_seed3_with_knob.json").read_text())
    marks = trace["episodes"][0]["bodies"]["plate"]["block_marks"]
    ok &= check("adoption_evidence_shows_the_block_absent",
                not [k for k in marks if k.endswith("_final")],
                f"'_final' waypoints in seed 3's trace: "
                f"{[k for k in marks if k.endswith('_final')] or 'none'}")

    (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence" / "plate_lookback_controls.json").write_text(
        json.dumps({"all_pass": bool(ok), "controls": results}, indent=1),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
