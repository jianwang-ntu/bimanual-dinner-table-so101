#!/usr/bin/env python3
"""Convert the trained scene-state regressor to OpenVINO IR at three precisions.

  FP32   direct conversion of the ONNX graph
  FP16   the same graph with weights compressed to half precision
  INT8   post-training quantization with NNCF, calibrated on training frames

Every conversion is checked against the PyTorch model it came from, on data
neither of them was fit to, and the check is reported in millimetres of table
position -- the unit the task is scored in -- not in abstract tensor norms.  A
precision that costs accuracy has to show it here.

Run:  python3 scripts/export_openvino.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

from envs import perception as P                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
STEM = "scene_state_cnn"


def _load_split(split: str):
    f = DATA / f"perception_{split}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run scripts/make_perception_dataset.py")
    d = np.load(f)
    return d["images"], d["labels"].astype(np.float32)


def _torch_model():
    ckpt = MODELS / f"{STEM}.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt} -- run scripts/train_perception.py")
    m = P.build_model()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    m.eval()
    return m


def _run_ov(compiled, images, batch_note="") -> np.ndarray:
    out = []
    for i in range(len(images)):
        x = P.preprocess(images[i])
        out.append(np.asarray(compiled(x)[compiled.output(0)])[0])
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=300,
                    help="training frames used to calibrate the INT8 model")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import openvino as ov

    MODELS.mkdir(exist_ok=True)
    torch_model = _torch_model()
    xva, yva = _load_split("val")
    xev, yev = _load_split("eval10")
    xtr, _ = _load_split("train")

    # ------------------------------------------------------------------ ONNX
    onnx_path = MODELS / f"{STEM}.onnx"
    dummy = torch.zeros(*P.IN_SHAPE)
    torch.onnx.export(torch_model, (dummy,), str(onnx_path),
                      input_names=["image"], output_names=["scene_state"],
                      opset_version=args.opset, dynamo=False)

    # -------------------------------------------------------- torch reference
    with torch.no_grad():
        ref_va = torch_model(torch.from_numpy(P.preprocess(xva))).numpy()
        ref_ev = torch_model(torch.from_numpy(P.preprocess(xev))).numpy()

    ov_fp32 = ov.convert_model(str(onnx_path))
    variants: dict[str, dict] = {}

    def save(tag: str, model, **kw):
        path = MODELS / f"{STEM}_{tag.lower()}.xml"
        ov.save_model(model, str(path), **kw)
        return path

    paths = {
        "FP32": save("fp32", ov_fp32, compress_to_fp16=False),
        "FP16": save("fp16", ov_fp32, compress_to_fp16=True),
    }

    # ------------------------------------------------------------------ INT8
    int8_note = None
    try:
        import nncf
        rng = np.random.default_rng(11)
        idx = rng.choice(len(xtr), size=min(args.calib, len(xtr)), replace=False)
        calib = nncf.Dataset([P.preprocess(xtr[i]) for i in idx])
        t0 = time.time()
        ov_int8 = nncf.quantize(ov.convert_model(str(onnx_path)), calib,
                                subset_size=len(idx))
        paths["INT8"] = save("int8", ov_int8, compress_to_fp16=False)
        int8_note = {"tool": f"nncf {nncf.__version__}",
                     "calibration_frames": int(len(idx)),
                     "calibration_source": "training split, random subset, seed 11",
                     "seconds": round(time.time() - t0, 1)}
    except ImportError as exc:                       # nncf is optional
        int8_note = {"skipped": f"nncf not installed ({exc})"}

    # ------------------------------------------------- accuracy of each export
    core = ov.Core()
    for tag, path in paths.items():
        compiled = core.compile_model(str(path), "CPU")
        pv, pe = _run_ov(compiled, xva), _run_ov(compiled, xev)
        variants[tag] = {
            "xml": str(path.relative_to(ROOT)),
            "checked_on": {
                "device": "CPU",
                "full_device_name": core.get_property("CPU", "FULL_DEVICE_NAME"),
                "execution_precision": str(
                    compiled.get_property("INFERENCE_PRECISION_HINT")),
            },
            "bin_bytes": (path.with_suffix(".bin")).stat().st_size,
            "accuracy_mm": {"val": P.position_error_mm(pv, yva),
                            "eval10": P.position_error_mm(pe, yev)},
            "vs_torch_mm": {
                "val_max_abs": float(np.abs(P.decode(pv) - P.decode(ref_va)).max() * 1000),
                "val_mean_abs": float(np.abs(P.decode(pv) - P.decode(ref_va)).mean() * 1000),
                "eval10_max_abs": float(np.abs(P.decode(pe) - P.decode(ref_ev)).max() * 1000),
            },
        }

    torch_acc = {"val": P.position_error_mm(ref_va, yva),
                 "eval10": P.position_error_mm(ref_ev, yev)}

    # The FP32 IR does not reproduce PyTorch bit for bit on this host, and the
    # reason is a runtime choice rather than a conversion loss.  Rather than
    # assert that, measure it: run the SAME IR under the plugin default and
    # under an explicit f32 hint and report both drifts.
    ab = {}
    for label, cfg in (("plugin_default", {}),
                       ("forced_f32", {"INFERENCE_PRECISION_HINT": "f32"})):
        c = core.compile_model(str(paths["FP32"]), "CPU", cfg)
        pe = _run_ov(c, xev)
        ab[label] = {
            "execution_precision": str(c.get_property("INFERENCE_PRECISION_HINT")),
            "eval10_max_abs_vs_torch_mm": float(
                np.abs(P.decode(pe) - P.decode(ref_ev)).max() * 1000),
        }

    report = {
        "schema": "openvino_export/v1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "openvino": ov.__version__,
        "torch": torch.__version__,
        "source_checkpoint": str((MODELS / f"{STEM}.pt").relative_to(ROOT)),
        "onnx": {"file": str(onnx_path.relative_to(ROOT)),
                 "opset": args.opset,
                 "bytes": onnx_path.stat().st_size,
                 "input": list(P.IN_SHAPE)},
        "torch_reference_accuracy_mm": torch_acc,
        "variants": variants,
        "int8": int8_note,
        "how_to_read_this":
            "accuracy_mm is the exported model's own error against MuJoCo ground "
            "truth. vs_torch_mm is how far the export drifted from the PyTorch "
            "model it was converted from. Both are in millimetres of table "
            "position on data no fitting used: val is 60 unseen compile seeds, "
            "eval10 is the ten seeds the task result is quoted on.",
        "precision_hint_ab": ab,
        "why_fp32_ir_is_not_bit_identical_to_torch":
            "the CPU plugin picks its own execution precision. precision_hint_ab "
            "runs the SAME FP32 IR twice -- once at the plugin default, once with "
            "INFERENCE_PRECISION_HINT=f32 -- so the drift is attributed by "
            "measurement rather than by assertion. Each variant above records the "
            "execution precision it was checked at.",
        "not_claimed": [
            "No measurement here was taken on Intel Core Ultra hardware. "
            "Conversion is host-independent; the latency and throughput the "
            "track asks for are not, and are not reported by this script.",
        ],
    }
    out = ROOT / "evidence" / "openvino_export.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"openvino {ov.__version__}   onnx opset {args.opset}")
    print("%-6s %10s  %12s  %14s" % ("prec", "weights", "eval10 err", "vs torch max"))
    print("%-6s %10s  %10.2f mm  %11.3f mm" % (
        "torch", f"{(MODELS/f'{STEM}.pt').stat().st_size/1024:.0f} KiB",
        torch_acc["eval10"]["worst_centre_mm"], 0.0))
    for tag, v in variants.items():
        print("%-6s %10s  %10.2f mm  %11.3f mm" % (
            tag, f"{v['bin_bytes']/1024:.0f} KiB",
            v["accuracy_mm"]["eval10"]["worst_centre_mm"],
            v["vs_torch_mm"]["eval10_max_abs"]))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
