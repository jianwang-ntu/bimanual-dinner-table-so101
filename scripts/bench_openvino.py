#!/usr/bin/env python3
"""Required Deliverable 3 -- the OpenVINO bench-test script.

Reports, for every OpenVINO device on the host and every exported precision:

  device selection   which plugin ran it, its FULL_DEVICE_NAME, and the
                     execution precision the plugin actually chose (which is
                     not always the precision written in the IR)
  latency            single-stream, LATENCY hint: mean, p50, p90, p99, min, max
  throughput         AsyncInferQueue at the plugin's own optimal request count
  task quality       the same model's position error in millimetres on the ten
                     seeds the task result is quoted on, so a faster precision
                     that lost accuracy cannot hide

It also states plainly whether the host is the hardware the track asks for.
The AI Infra Summit robotics track scores "optimized inference on Intel Core
Ultra Series 2/3".  This script is written so that the operator can run it
unchanged on such a machine; when it runs anywhere else it says so in the
report and marks the run NOT_THE_REQUIRED_MEASUREMENT.  Numbers from other
silicon are never presented as Intel Core Ultra numbers.

Run:
  python3 scripts/bench_openvino.py                       # every device found
  python3 scripts/bench_openvino.py --device CPU NPU      # pick devices
  python3 scripts/bench_openvino.py --iters 500
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                      # noqa: E402

from envs import perception as P                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STEM = "scene_state_cnn"
PRECISIONS = ("FP32", "FP16", "INT8")

# "Core Ultra 7 268V", "Core(TM) Ultra 9 285H" -- series 2 and 3 are the 200V /
# 200H / 300 families.  The pattern is deliberately loose: it is used to decide
# whether to CLAIM the required measurement, and it errs towards not claiming.
CORE_ULTRA = re.compile(r"intel.*core.*ultra", re.I)


def host_cpu_name() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def required_hardware_verdict(cpu_name: str) -> dict:
    """Does this host satisfy the track's hardware requirement?

    Kept as a function, not an inline branch, so that the ACCEPT side -- the
    branch that fires only on an Intel Core Ultra machine this project has no
    access to -- can be exercised by scripts/test_perception_pipeline.py.
    """
    is_core_ultra = bool(CORE_ULTRA.search(cpu_name))
    return {
        "track_asks_for": "Intel Core Ultra Series 2/3 with OpenVINO",
        "this_host_is_intel_core_ultra": is_core_ultra,
        "verdict": ("MEASURED_ON_REQUIRED_HARDWARE" if is_core_ultra
                    else "NOT_THE_REQUIRED_MEASUREMENT"),
        "note": (f"Latency and throughput were measured on '{cpu_name}', which "
                 "is the hardware the track asks for."
                 if is_core_ultra else
                 f"Latency and throughput below were measured on '{cpu_name}'. "
                 "They are NOT Intel Core Ultra numbers and must not be quoted "
                 "as if they were. Running this script unchanged on a Core Ultra "
                 "Series 2/3 machine produces the measurement the track scores, "
                 "including the NPU device if its driver is installed."),
    }


def npu_driver_present() -> bool:
    return pathlib.Path("/dev/accel/accel0").exists() or \
        pathlib.Path("/dev/intel_vpu").exists()


def percentiles(xs: np.ndarray) -> dict:
    return {
        "mean_ms": float(xs.mean()), "p50_ms": float(np.percentile(xs, 50)),
        "p90_ms": float(np.percentile(xs, 90)), "p99_ms": float(np.percentile(xs, 99)),
        "min_ms": float(xs.min()), "max_ms": float(xs.max()),
        "stdev_ms": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
    }


def bench_one(core, xml: pathlib.Path, device: str, frames: np.ndarray,
              labels: np.ndarray, iters: int, warmup: int) -> dict:
    import openvino as ov
    import openvino.properties as props
    import openvino.properties.hint as hints

    inputs = [P.preprocess(f) for f in frames]

    # ---------------------------------------------------------- latency mode
    t_compile = time.perf_counter()
    lat_model = core.compile_model(str(xml), device,
                                   {hints.performance_mode: hints.PerformanceMode.LATENCY})
    compile_s = time.perf_counter() - t_compile
    req = lat_model.create_infer_request()
    out_port = lat_model.output(0)

    for i in range(warmup):
        req.infer(inputs[i % len(inputs)])
    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        req.infer(x)
        times[i] = (time.perf_counter() - t0) * 1e3

    # ---------------------------------------------------- task quality check
    preds = np.stack([np.asarray(req.infer(x)[out_port])[0] for x in inputs])
    quality = P.position_error_mm(preds, labels)

    # ------------------------------------------------------- throughput mode
    tp_model = core.compile_model(
        str(xml), device,
        {hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
    nreq = int(tp_model.get_property(props.optimal_number_of_infer_requests))
    queue = ov.AsyncInferQueue(tp_model, nreq)
    n_async = max(iters, nreq * 8)
    for i in range(nreq):                                   # warm the queue
        queue.start_async(inputs[i % len(inputs)])
    queue.wait_all()
    t0 = time.perf_counter()
    for i in range(n_async):
        queue.start_async(inputs[i % len(inputs)])
    queue.wait_all()
    wall = time.perf_counter() - t0

    return {
        "compile_seconds": round(compile_s, 3),
        "execution_precision": str(
            lat_model.get_property(hints.inference_precision)),
        "latency_stream": {"iterations": iters, "warmup": warmup,
                           **{k: round(v, 4) for k, v in percentiles(times).items()},
                           "fps_single_stream": round(1000.0 / times.mean(), 1)},
        "throughput": {"infer_requests": nreq, "iterations": n_async,
                       "wall_seconds": round(wall, 4),
                       "fps": round(n_async / wall, 1)},
        "task_quality_mm": {k: round(v, 3) for k, v in quality.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", nargs="*", default=None,
                    help="OpenVINO devices; default is every device found")
    ap.add_argument("--precision", nargs="*", default=list(PRECISIONS),
                    choices=list(PRECISIONS))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import openvino as ov

    npz = ROOT / "data" / "perception_eval10.npz"
    if not npz.exists():
        raise SystemExit(f"missing {npz} -- run scripts/make_perception_dataset.py "
                         f"--split eval10")
    d = np.load(npz)
    frames, labels = d["images"], d["labels"].astype(np.float32)

    core = ov.Core()
    devices = args.device if args.device else list(core.available_devices)

    cpu_name = host_cpu_name()
    is_core_ultra = bool(CORE_ULTRA.search(cpu_name))

    host = {
        "cpu": cpu_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "openvino": ov.__version__,
        "available_devices": list(core.available_devices),
        "npu_device_node_present": npu_driver_present(),
        "is_intel_core_ultra": is_core_ultra,
    }
    for dev in host["available_devices"]:
        try:
            host.setdefault("device_names", {})[dev] = core.get_property(
                dev, "FULL_DEVICE_NAME")
        except Exception as exc:                       # plugin may refuse
            host.setdefault("device_names", {})[dev] = f"<unavailable: {exc}>"

    results, errors = {}, {}
    for dev in devices:
        for prec in args.precision:
            xml = MODELS / f"{STEM}_{prec.lower()}.xml"
            if not xml.exists():
                errors[f"{dev}/{prec}"] = f"missing {xml.name} -- run export_openvino.py"
                continue
            key = f"{dev}/{prec}"
            try:
                results[key] = bench_one(core, xml, dev, frames, labels,
                                         args.iters, args.warmup)
                results[key]["ir_precision"] = prec
                results[key]["ir_weight_bytes"] = xml.with_suffix(".bin").stat().st_size
                r = results[key]
                print("%-12s  %7.3f ms p50  %8.1f fps async  %6.2f mm  (%s)" % (
                    key, r["latency_stream"]["p50_ms"], r["throughput"]["fps"],
                    r["task_quality_mm"]["worst_centre_mm"],
                    r["execution_precision"]))
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                print(f"{key:<12}  FAILED  {type(exc).__name__}: {exc}")

    report = {
        "schema": "openvino_bench/v1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deliverable": "AI Infra Summit robotics track, Required Deliverable 3 "
                       "-- bench-test script reporting latency, throughput, "
                       "device selection and model precision",
        "model": {"stem": STEM, "input": list(P.IN_SHAPE),
                  "outputs": list(P.OUT_NAMES),
                  "what_it_does": "regresses the table layout from one top_cam "
                                  "frame; see envs/perception.py"},
        "host": host,
        "run": {"iterations": args.iters, "warmup": args.warmup,
                "devices": devices, "precisions": args.precision},
        "results": results,
        "errors": errors,
        "required_hardware": required_hardware_verdict(cpu_name),
        "how_to_read_this": {
            "latency_stream": "one request at a time under the LATENCY hint; "
                              "p50/p90/p99 over --iters inferences on real frames",
            "throughput": "AsyncInferQueue at the plugin's own "
                          "optimal_number_of_infer_requests under the THROUGHPUT hint",
            "task_quality_mm": "mean position error against MuJoCo ground truth on "
                               "the ten evaluation seeds, so a cheaper precision "
                               "shows its accuracy cost in the same table as its speed",
            "execution_precision": "what the plugin chose at runtime, which can "
                                   "differ from ir_precision",
        },
    }

    tag = re.sub(r"[^a-z0-9]+", "_", cpu_name.lower()).strip("_")[:40] or "host"
    out = pathlib.Path(args.out) if args.out else \
        ROOT / "evidence" / f"openvino_bench_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nhost: {cpu_name}")
    print(f"required hardware: {report['required_hardware']['verdict']}")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
