# Technical summary / architecture

**Required Deliverable 5** for the *Bimanual VLA Manipulation with Multi-Modal
Reasoning* track — a concise description of the solution architecture, VLA/VLM
model choice, bimanual coordination strategy, training approach, robustness
methods, OpenVINO optimization, and Intel hardware mapping.

Every number below is reproduced by a command in the last section and is stored
in a JSON file in `evidence/`. `scripts/test_technical_summary.py` re-derives
each of them from those files and fails if this document and the evidence
disagree, so a stale figure here is a red suite rather than a reader's problem.

## Read this first

| The rubric asks for | This entry has |
|---|---|
| End-to-end dinner-table task, two arms | **16 / 50** sub-goals over 10 seeds; task success **0 / 10** |
| A VLA / multi-modal policy | **nothing.** No VLA, no VLM, no learned policy, no language input |
| OpenVINO on Intel Core Ultra Series 2/3 | OpenVINO yes, measured; **Core Ultra silicon: none, and none claimed** |

Three of the five sub-goals have never fired on any seed. The entry is a
verified environment, a verified scorer, a scripted controller that earns two of
five steps, and a perception model that is not yet in the loop.

---

## 1. Solution architecture

Two layers, and only the first is scored today.

```
                     scripts/eval_seeds.py            (harness: seeds -> score)
                                |
   envs/randomize.py  ---->  MuJoCo model+data  <----  envs/dinner_table.py
   (3-stage domain rand.)        |      |              (scene as code -> XML)
                                 |      |
     envs/controller.py  --------+      +---------->  envs/task.py
     scripted waypoint FSM                            5 sub-goals, sequencing,
     over damped-least-squares IK                     hand-off, drop, stability
     reads MjData object poses  <-- privileged
                                 :
                                 : NOT CONNECTED
                                 :
   evidence/frames -> envs/perception.py -> models/*.onnx -> OpenVINO IR
   one 128x224 top_cam frame       520,768-param CNN        FP32 / FP16 / INT8
                                   -> 7 scene-state numbers  scripts/bench_openvino.py
```

The dotted edge is the honest centre of this entry: `envs/perception.py` can
recover the table layout from a camera to **1.83 mm**, but `envs/controller.py`
still reads object poses out of `MjData`. Closing that edge is the single
largest piece of unbuilt work and is not counted anywhere below.

| File | Role |
|---|---|
| `envs/dinner_table.py` | scene as code; `build(dims=...)` returns an uncompiled `MjSpec` so geometry can vary per episode |
| `envs/dinner_table.xml` | the nominal compiled scene — 48 DoF, 12 actuators, 133 geoms, 5 cameras |
| `envs/ik.py` | damped-least-squares site IK |
| `envs/randomize.py` | geometry / model / state randomization, seeded |
| `envs/task.py` | instruction, 5 sub-goals, sequencing, hand-off and drop detection |
| `envs/controller.py` | the scripted bimanual controller (no learned parameters) |
| `envs/perception.py` | scene-state CNN — the only learned component in the repo |
| `scripts/eval_seeds.py` | the scored run: N seeds, one JSON |
| `scripts/export_openvino.py` | PyTorch → ONNX → IR at three precisions, each checked in millimetres |
| `scripts/bench_openvino.py` | Required Deliverable 3 — latency, throughput, device, precision |

The scorer is deliberately independent of any controller, so the same numbers
mean the same thing for this script, for a learned policy, or for a human
teleoperating the arms.

## 2. VLA / VLM model choice

**No VLA or VLM is used, and none is claimed.** Criterion T2 ("correctly
interprets natural-language instructions and visual observations, maintains
multi-step task context, selects appropriate actions, adapts the plan") scores
zero on this entry. There is no language encoder, no instruction conditioning
and no learned action head anywhere in the repository. The task instruction in
`envs/task.py` is a fixed string that the scorer prints; nothing consumes it.

What exists instead, and what it is worth:

- **`envs/perception.py` — a 520,768-parameter scene-state CNN.** Four stride-2
  convolution blocks over one 128×224 `top_cam` frame, then a **spatial
  softmax** and a small MLP, regressing seven numbers: the planar centres of the
  plate, mug and bottle and how far the drawer is out. A global average pool
  cannot do coordinate regression — averaging over space discards the position
  being asked for — so each channel becomes a soft keypoint instead.
  Worst object centre **1.32 mm** on unseen validation seeds and **1.83 mm** on
  the ten evaluation seeds, against **34.29 mm** for the no-vision baseline.
- This is *visual observation*, one of the four things T2 asks for, and it is
  not in the control loop, so it earns nothing on the scored path.

The choice that would be made next, recorded as a plan and not as work done:
**ACT or SmolVLA via LeRobot**, trained on rollouts of the scripted controller
in this same environment, because the controller already produces labelled
bimanual trajectories at 1.6 sub-goals per episode and the scorer is policy
agnostic. **Nothing of that is built.**

## 3. Bimanual coordination strategy

Two Robot Studio SO-101 arms — 5 positioning joints and a parallel jaw each —
mounted 0.44 m apart on a shared table and yawed 36° inward. Coordination is
explicit and scripted in `dinner_table_script()`, which reads top to bottom
against the task instruction:

1. **Drawer** — the right arm opens it, from home, jaws narrow.
2. **Fork** — right picks it out of the drawer and hands it to the left.
3. **Spoon** — the mirror image, left to right.
4. **Plate** — both arms return home, then the right arm hooks the rim and drags.
5. **Mug** — the left arm sets it to the right of the plate.
6. **Drawer re-check**, twice.

Each cutlery item starts on the far side of the table from its target, so the
hand-off is structural rather than decorative: neither arm can reach both ends.
In `_handoff()` the **taker closes before the giver opens**, so the object is
briefly held by both arms — that overlap is what `TaskMonitor` records.

Three mechanisms make a grasp land at all, and all three came out of
measurement:

- **`tip_mid`** — IK targets the point where the jaws *meet*, not the
  `gripperframe` site. They differ by ~31 mm along the approach axis, and that
  difference is why an unmodified position-IK grasp brushes past everything.
- **`align_roll`** — `wrist_roll` turns the jaw closing direction about the
  approach axis without moving `gripperframe`, squaring the jaws onto a fork
  handle or a plate rim.
- **`plan_pose(..., standoff=)`** — only one SO-101 jaw moves, so aiming the
  meeting point at an object puts the *fixed* jaw inside it (measured: 7.1 mm
  inside the mug wall, 0.4 mm inside the plate rim). The standoff backs the
  meeting point off along the live jaw axis.

**What this earns, measured on 10 seeds against the same scorer:**

| Sub-goal | Scripted | No-policy control |
|---|---|---|
| `drawer_open` | **10 / 10** | 0 |
| `plate_placed` | **6 / 10** | 0 |
| `fork_placed` | **0 / 10** | 0 |
| `spoon_placed` | **0 / 10** | 0 |
| `mug_placed` | **0 / 10** | 0 |
| total | **16 / 50** | **0 / 50** |
| task success | **0 / 10** | 0 / 10 |

Both arms touch a manipulable object on **7 / 10** seeds. `handoff_occurred` is
true on all ten and that number is misleading: **every recorded hand-off is on
the drawer handle**, the two arms taking it in turn during the end-of-episode
re-check. **No object hand-off has ever occurred**, so the coordination strategy
described above is implemented and is not yet demonstrated. The plate is hooked
by its rim and dragged flat, not picked and placed: it is 92–116 mm across
against a 101 mm jaw span, so no jaw opening both clears it and closes on it.

The binding physical limit is servo saturation: at a waypoint 0.297 m out,
`shoulder_pan`, `shoulder_lift` and `elbow_flex` all sit at their ±2.94 N·m
limit with no contact anywhere on the arm, while random joint sampling reaches
0.46 m. IK returns poses the arm never reaches. That is a property of the robot
and it is unfixed.

## 4. Training approach

One model is trained. **No policy is trained.**

| | |
|---|---|
| Model | `SceneStateCNN`, **520,768** parameters, input 1×3×128×224 |
| Data | **4,000** training frames, **240** validation, **10** evaluation |
| Split rule | disjoint **compile seeds** — train 1000–1499, val 2000–2059, eval 0–9 — not a shuffle |
| Labels | from the simulator, not annotated |
| Schedule | **80** epochs, batch 64, AdamW + OneCycle at lr 2e-3, SmoothL1(β=0.02), seed 7 |
| Cost | 103.5 s on one NVIDIA L40S |
| Selection | best validation worst-centre + drawer error |

Frames are rendered by `scripts/make_perception_dataset.py` from the same
randomizer the task uses, so the training distribution *is* the evaluation
distribution apart from the seed split. The evaluation split is the ten seeds
`eval_seeds.py` scores, at exactly their scored initial state.

Two controls decide whether the number means anything, because a metric that
cannot fail proves nothing:

- **No-vision baseline** — the train-set mean layout scored the same way:
  **34.29 mm**, against the model's **1.32 mm**.
- **Shuffled labels** — the same predictions re-scored against permuted
  targets: 46.37 mm, i.e. worse than the constant baseline.
- **Causal control** — move the mug 60 mm in the scene, re-render, re-predict:
  the mug prediction moves 59.0 mm and the untouched plate prediction 0.4 mm.

Trained and evaluated on initial states with both arms at home. Mid-rollout
frames, where the arms occlude the table, are **not** measured.

## 5. Robustness methods

**Randomization** (`envs/randomize.py::RANGES`, three stages because only the
first needs a recompile; every draw is logged per episode):

| Stage | What varies |
|---|---|
| geometry | plate radius, mug radius and height, bottle radius and height, cutlery length |
| model | per-object mass 0.6–1.6×, sliding friction 0.6–1.4×, key-light position and intensity, table and floor lightness |
| state | object x, y and yaw, initial drawer opening |

Placements are rejection-sampled against three conditions — inside an arm's
reach, not overlapping another object, not blocking the drawer — so a failed
episode is a policy failure and not an impossible scene. Across seeds 0–9 every
episode was on-table, reachable, free of initial interpenetration and
numerically stable for its full length, in both the control and the scripted
run. Nothing is dropped on any seed.

**Closed-loop recovery**, each added because an open-loop version was measured
failing:

- **Contact-seeking descent.** The plate rim is 8 mm tall and the arm tracks a
  commanded pose to 12–20 mm, so one aim at the middle of that window misses
  more often than it hits. The controller steps down 4 mm at a time until a jaw
  touches.
- **Re-aiming drag.** Each waypoint is the jaws' current position plus a share
  of the plate's *own* remaining offset, read at the moment the move starts.
  Nothing about the grasp is cached, so a slip changes the next waypoint. The
  whole cycle repeats up to twice if the plate is short of the mat.
- **Drawer re-check, twice.** Before it, the drawer reached its full 90 mm on
  10 of 10 seeds and was knocked shut again on 5; a single retry left 3 of those
  still shut. The drawer is also released with the jaws **narrow**: opening them
  fully swings the moving jaw through 80 mm, which catches the drawer front and
  collapses 92 mm of travel to 45 mm. With both, `drawer_open` is **10 / 10**.

Raising the drawer's slide friction so it would not drift was tried first and
made things worse — 2/50, because the arm could no longer pull it fully open.
It was reverted; the friction in the scene is the original 0.35.

**Verification.** `scripts/verify_scene.py` **16 / 16** structural and physical
checks; `scripts/test_task_predicates.py` **11 / 11** accept *and* reject
controls on the scorer; `scripts/test_perception_pipeline.py` **13 / 13**
accept and reject controls on the model, the exports and the hardware gate.

## 6. OpenVINO optimization

`scripts/export_openvino.py` goes PyTorch → ONNX (opset 17) → OpenVINO IR at
three precisions and checks each against the PyTorch model it came from, in
**millimetres of table position** rather than tensor norms:

| Precision | Weights | Error on the 10 evaluation seeds | Max drift vs PyTorch |
|---|---|---|---|
| IR FP32 | **2031 KiB** | **1.81 mm** | **0.45 mm** |
| IR FP16 | **1015 KiB** | **2.04 mm** | **1.12 mm** |
| IR INT8 (NNCF 3.3.0 PTQ, 300 calibration frames) | **515 KiB** | **3.98 mm** | **5.87 mm** |

INT8 is 3.9× smaller than FP32 and costs 2.2 mm of accuracy. That cost is
reported rather than buried: `test_perception_pipeline.py` carries a control
that **fails** if the INT8 error is ever recorded as no worse than FP32.

The FP32 IR is not bit-identical to PyTorch, and the reason is attributed by
measurement rather than asserted — the same IR is run twice:

| Execution precision | Max drift vs PyTorch |
|---|---|
| plugin default (`bfloat16` on this host) | **0.4488 mm** |
| forced `INFERENCE_PRECISION_HINT=f32` | **0.00009 mm** |

So the drift is the CPU plugin's own precision choice, not a conversion loss.
Conversion itself is host-independent; latency and throughput are not.

## 7. Intel hardware mapping

The track requires the final demonstration and the benchmark to run on an
**Intel Core Ultra Series 2/3** system. **This machine has no Intel silicon**,
so that measurement does not exist here and is not claimed anywhere in this
repository.

`scripts/bench_openvino.py` enumerates every device OpenVINO reports on the host
and, per device and per exported precision, records single-stream latency
(mean, p50, p90, p99, min, max under the LATENCY hint), async throughput at the
plugin's own `optimal_number_of_infer_requests` under the THROUGHPUT hint, the
execution precision the plugin actually chose, and the model's task quality in
millimetres on the same ten seeds the task result is quoted on.

Intended mapping on a Core Ultra part, stated as a plan because it has not been
run: **NPU** for the steady-state perception tick at INT8, **Intel iGPU** for
FP16 when the NPU is busy or absent, **CPU** as the always-available fallback.
The script needs no edit to do this — it takes whatever `available_devices`
returns.

Run on *this* host it produces **9** device/precision rows and stamps the
report `NOT_THE_REQUIRED_MEASUREMENT`:

```
CPU/FP32   0.436 ms p50   5220 fps async   1.81 mm   (bfloat16)
host: AMD EPYC 9654 96-Core Processor
```

**Those are not Intel Core Ultra numbers and are not offered as any.** The
script decides that itself: `required_hardware_verdict()` reads the host CPU
name and returns `MEASURED_ON_REQUIRED_HARDWARE` only for a Core Ultra part.
Because that branch cannot fire here, it is driven directly by
`test_perception_pipeline.py` against three real Core Ultra model strings and
five non-Core-Ultra ones — an accept path that is never run is how a gate ships
broken.

To obtain the figure the track scores, run `python3 scripts/bench_openvino.py`
unchanged on a Core Ultra Series 2/3 machine. No build and no training are
needed; the IRs are committed.

## 8. What is not built

Stated here in one place so no reader has to infer it:

1. **No VLA, no VLM, no learned policy, no language conditioning.** T2 scores zero.
2. **Perception is not in the control loop.** The scored result is produced by a
   controller reading privileged `MjData` poses.
3. **No Intel Core Ultra measurement.** The benchmark script exists and runs;
   the required silicon does not exist here.
4. **The task has never been completed.** Task success **0 / 10**; three of five
   sub-goals have never fired; no object hand-off has ever occurred.
5. The plate is dragged, not carried.
6. Mid-rollout perception, where the arms occlude the table, is unmeasured.

## 9. Reproducing every number in this document

```bash
pip install -r requirements.txt
python3 scripts/build_scene.py                              # envs/dinner_table.xml
python3 scripts/verify_scene.py                             # 16/16
python3 scripts/test_task_predicates.py                     # 11/11
python3 scripts/eval_seeds.py --seeds 10                    # control  -> evidence/eval_seeds.json
python3 scripts/eval_seeds.py --seeds 10 --policy scripted  # scored   -> evidence/eval_seeds_scripted.json
python3 scripts/record_demo.py --seeds 10                   # the demo video
python3 scripts/make_perception_dataset.py --split train --compiles 500 --per-compile 8
python3 scripts/make_perception_dataset.py --split val   --compiles 60  --per-compile 4
python3 scripts/make_perception_dataset.py --split eval10
python3 scripts/train_perception.py --epochs 80             # evidence/perception_train.json
python3 scripts/export_openvino.py                          # evidence/openvino_export.json
python3 scripts/bench_openvino.py                           # evidence/openvino_bench_*.json
python3 scripts/test_perception_pipeline.py                 # 13/13

python3 scripts/test_technical_summary.py                   # this document vs the evidence
```

The last line is the one that keeps this document honest. It re-derives every
figure quoted above from the JSON in `evidence/`, requires all seven Deliverable
5 topics to be present, requires the absence ledger in section 8 to name every
absence the evidence records, and re-derives the demonstration video's own
"what this does not show" sentence from the rollouts it filmed. Every mechanism
is driven from both sides: corrupting any one of the 29 figures, deleting any
one of the seven sections, inflating the task-success figure, deleting the
absence statement, restoring the stale sentence the video record actually
shipped, and moving a number in `evidence/` while leaving this document alone
must each make it fail — and each is checked. It needs only the Python standard
library, so a judge can run it on a fresh clone before installing anything.
