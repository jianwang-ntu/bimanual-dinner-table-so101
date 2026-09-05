#!/usr/bin/env python3
"""Controls for the perception model, its OpenVINO exports and the bench gate.

Same shape as scripts/test_task_predicates.py: every mechanism is driven from
both sides, because a check that can only pass proves nothing.

  perception   ACCEPT it beats a no-vision baseline and tracks an object that
               actually moved in the scene; REJECT it keeps no skill when the
               image is replaced with noise or the labels are permuted
  exports      ACCEPT the FP32 IR reproduces PyTorch; REJECT INT8 being sold as
               free -- its accuracy cost must be present and reported
  Intel gate   ACCEPT a Core Ultra host is recognised as the required hardware;
               REJECT this host, a Xeon and a Core i9 being recognised as it

The Intel ACCEPT control is the reason required_hardware_verdict() is a
function: that branch fires only on hardware this project has no access to, and
an untested accept path is how a gate ships broken.

Run:  python3 scripts/test_perception_pipeline.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402
import mujoco                                               # noqa: E402
import torch                                                # noqa: E402

from envs import perception as P                            # noqa: E402
from envs.randomize import make_env                         # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_sibling(name: str):
    """Import a sibling script by path -- scripts/ is a folder, not a package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bench = _load_sibling("bench_openvino")
required_hardware_verdict = _bench.required_hardware_verdict
host_cpu_name = _bench.host_cpu_name
EV = ROOT / "evidence"
results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def _json(path: pathlib.Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model():
    ck = ROOT / "models" / "scene_state_cnn.pt"
    m = P.build_model()
    m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
    m.eval()
    return m


def main() -> int:
    ok = True
    train = _json(EV / "perception_train.json")
    export = _json(EV / "openvino_export.json")
    benches = sorted(EV.glob("openvino_bench_*.json"))
    bench = _json(benches[0]) if benches else None

    if train is None or export is None or bench is None:
        print("missing evidence -- run train_perception.py, export_openvino.py "
              "and bench_openvino.py first")
        return 2

    # ------------------------------------------------------------ perception
    for split in ("val", "eval10"):
        m = train["error_mm"][split]["worst_centre_mm"]
        b = train["controls"][f"constant_baseline_{split}"]["worst_centre_mm"]
        ok &= check(f"accept_beats_no_vision_baseline_{split}", m < b / 5.0,
                    f"model {m:.2f} mm vs constant baseline {b:.2f} mm")

    sh = train["controls"]["shuffled_labels_val"]["worst_centre_mm"]
    bv = train["controls"]["constant_baseline_val"]["worst_centre_mm"]
    mv = train["error_mm"]["val"]["worst_centre_mm"]
    ok &= check("reject_shuffled_labels_score_like_the_model",
                sh >= bv and sh > 5 * mv,
                f"shuffled {sh:.2f} mm, baseline {bv:.2f} mm, model {mv:.2f} mm")

    # From a clean clone only the eval10 frames are present -- the train and val
    # renders are regenerated, not shipped. Fall back to the committed dataset
    # metadata, which records the seed range each split was built from.
    seeds, basis = {}, {}
    for split in ("train", "val", "eval10"):
        npz = ROOT / "data" / f"perception_{split}.npz"
        meta = ROOT / "data" / f"perception_{split}.json"
        if npz.exists():
            seeds[split] = set(np.load(npz)["seeds"].tolist())
            basis[split] = "npz"
        elif meta.exists():
            lo, hi = json.loads(meta.read_text())["seed_range"]
            seeds[split] = set(range(lo, hi + 1))
            basis[split] = "metadata seed_range"
        else:
            seeds[split] = set()
            basis[split] = "absent"
    if all(seeds.values()):
        overlap = (seeds["train"] & seeds["val"]) | (seeds["train"] & seeds["eval10"])
        ok &= check("accept_splits_are_seed_disjoint", not overlap,
                    f"train {len(seeds['train'])}, val {len(seeds['val'])}, "
                    f"eval10 {len(seeds['eval10'])} compile seeds "
                    f"(from {basis}), overlap {sorted(overlap)}")
    else:
        ok &= check("accept_splits_are_seed_disjoint", False,
                    f"split records absent -- cannot check ({basis})")

    net = _model()
    d = np.load(ROOT / "data" / "perception_eval10.npz")
    x_ev, y_ev = d["images"], d["labels"].astype(np.float32)
    with torch.no_grad():
        pred = net(torch.from_numpy(P.preprocess(x_ev))).numpy()
    real = P.position_error_mm(pred, y_ev)["worst_centre_mm"]

    # corpus swap: the same network, the same scorer, images replaced by noise
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=x_ev.shape, dtype=np.uint8)
    with torch.no_grad():
        pred_n = net(torch.from_numpy(P.preprocess(noise))).numpy()
    noise_err = P.position_error_mm(pred_n, y_ev)["worst_centre_mm"]
    ok &= check("reject_noise_images_score_like_real_ones",
                noise_err > 10 * real,
                f"noise {noise_err:.1f} mm vs real frames {real:.2f} mm")

    # causal control: move one object in the scene, re-render, re-predict
    model, data, _ = make_env(0)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_free")
    adr = model.jnt_qposadr[jid]
    shift = 0.06
    with mujoco.Renderer(model, height=P.IMG_H, width=P.IMG_W) as r:
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=P.CAMERA)
        before = r.render().copy()
        data.qpos[adr] += shift
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=P.CAMERA)
        after = r.render().copy()
    with torch.no_grad():
        pb = P.decode(net(torch.from_numpy(P.preprocess(before))).numpy())[0]
        pa = P.decode(net(torch.from_numpy(P.preprocess(after))).numpy())[0]
    i_mug = P.OUT_NAMES.index("mug_x")
    i_plate = P.OUT_NAMES.index("plate_x")
    d_mug = float(pa[i_mug] - pb[i_mug])
    d_plate = abs(float(pa[i_plate] - pb[i_plate]))
    ok &= check("accept_prediction_follows_a_moved_object",
                abs(d_mug - shift) < 0.010 and d_plate < 0.010,
                f"mug moved {shift*1000:.0f} mm, prediction moved "
                f"{d_mug*1000:.1f} mm; untouched plate prediction moved "
                f"{d_plate*1000:.1f} mm")

    # --------------------------------------------------------------- exports
    v = export["variants"]
    fp32_drift = v["FP32"]["vs_torch_mm"]["eval10_max_abs"]
    ab = export.get("precision_hint_ab", {})
    forced = ab.get("forced_f32", {}).get("eval10_max_abs_vs_torch_mm")
    ok &= check("accept_fp32_ir_reproduces_torch",
                fp32_drift < 1.0 or (forced is not None and forced < 0.01),
                f"plugin default {fp32_drift:.4f} mm, forced f32 "
                f"{forced if forced is None else round(forced, 5)} mm")

    if "INT8" in v:
        s32 = v["FP32"]["bin_bytes"]
        s8 = v["INT8"]["bin_bytes"]
        e32 = v["FP32"]["accuracy_mm"]["eval10"]["worst_centre_mm"]
        e8 = v["INT8"]["accuracy_mm"]["eval10"]["worst_centre_mm"]
        ok &= check("accept_int8_is_smaller_than_fp32", s8 < s32,
                    f"{s8/1024:.0f} KiB vs {s32/1024:.0f} KiB")
        ok &= check("reject_int8_is_free", e8 > e32,
                    f"INT8 {e8:.2f} mm vs FP32 {e32:.2f} mm -- the cost is "
                    f"measured and reported, not hidden")

    # ------------------------------------------------------------ Intel gate
    accepts = ["Intel(R) Core(TM) Ultra 7 268V",
               "Intel(R) Core(TM) Ultra 9 285H",
               "Intel(R) Core(TM) Ultra 5 225U"]
    rejects = [host_cpu_name(),
               "Intel(R) Xeon(R) Platinum 8480+",
               "Intel(R) Core(TM) i9-13900K",
               "AMD EPYC 9654 96-Core Processor",
               "Apple M3 Pro"]
    got_a = [required_hardware_verdict(n)["verdict"] for n in accepts]
    got_r = [required_hardware_verdict(n)["verdict"] for n in rejects]
    ok &= check("accept_core_ultra_host_is_the_required_hardware",
                all(g == "MEASURED_ON_REQUIRED_HARDWARE" for g in got_a),
                dict(zip(accepts, got_a)))
    ok &= check("reject_non_core_ultra_host_is_the_required_hardware",
                all(g == "NOT_THE_REQUIRED_MEASUREMENT" for g in got_r),
                dict(zip(rejects, got_r)))
    ok &= check("reject_this_run_claims_intel_numbers",
                bench["required_hardware"]["verdict"] == "NOT_THE_REQUIRED_MEASUREMENT"
                and bench["host"]["is_intel_core_ultra"] is False,
                f"{benches[0].name}: {bench['required_hardware']['verdict']} on "
                f"{bench['host']['cpu']}")

    got = bench["results"]
    complete = all(("latency_stream" in r and "throughput" in r and
                    "task_quality_mm" in r and "execution_precision" in r)
                   for r in got.values())
    ok &= check("accept_bench_reports_all_four_required_fields",
                bool(got) and complete,
                f"{len(got)} device/precision combinations, each with latency, "
                f"throughput, device precision and task quality")

    EV.mkdir(parents=True, exist_ok=True)
    (EV / "perception_pipeline_controls.json").write_text(
        json.dumps({"all_pass": bool(ok), "controls": results}, indent=1),
        encoding="utf-8")
    print(f"\n{'ALL CONTROLS PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
