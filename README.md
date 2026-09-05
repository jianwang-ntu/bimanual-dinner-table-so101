# Bimanual dinner-table manipulation — dual SO-101 in MuJoCo

Simulation package for the **Bimanual VLA Manipulation with Multi-Modal
Reasoning** online track (Intel Physical AI Online Challenge, AI Infra Summit
hackathon, lablab.ai). Target scenario: *Setting Up a Dinner Table*.

## What is in here, and what is not

This repository contains **the environment, the scorer, and a scripted
controller that solves two sub-goals of five on 6 seeds of 10, and never the
whole task. There is no learned policy.** Read that before reading anything
else.

| Piece | State | Evidence |
|---|---|---|
| Dual SO-101 MuJoCo scene, drawer, plate, mug, bottle, cutlery | built, verified | `evidence/scene_verification.json` (16/16) |
| Seeded domain randomization — geometry, mass, friction, lighting, background, placement | built, verified over 10 seeds | `evidence/eval_seeds.json` |
| Task definition and success predicates | built, controls both ways | `evidence/task_predicate_controls.json` (11/11) |
| Scripted bimanual controller | built, **16 sub-goals of 50 over 10 seeds** | `evidence/eval_seeds_scripted.json` |
| Demo video, 10 randomized seeds, captioned from the simulator | recorded, **shows a partial rollout** | `evidence/demo_scripted_10seeds.mp4` + `.json` |
| Scene-state perception CNN, trained on rendered frames | built, **1.3 mm val / 1.8 mm on the evaluation seeds** | `evidence/perception_train.json` |
| OpenVINO conversion — FP32, FP16, INT8 (NNCF) | built, accuracy of each measured | `evidence/openvino_export.json` |
| Bench-test script — latency, throughput, device, precision | built, **run here on AMD, not on Intel** | `evidence/openvino_bench_*.json` |
| Controls for all of the above, accept and reject | 13/13 | `evidence/perception_pipeline_controls.json` |
| Technical summary / architecture — Required Deliverable 5 | written, claims machine-checked | [`TECHNICAL_SUMMARY.md`](TECHNICAL_SUMMARY.md), `evidence/technical_summary_controls.json` (11/11) |
| Submission cover image, 1920×1080 PNG rendered from the simulator | rendered, **captioned from `evidence/`, not typed** | `evidence/cover_image.png` + `.json`, `evidence/cover_image_controls.json` (14/14) |
| Slide presentation, 13 slides, 16:9 PDF | generated from `evidence/`, **no figure typed by hand** | `evidence/slides_presentation.pdf` + `.json`, `evidence/slides_controls.json` (12/12) |
| VLA / imitation policy | **not started** | — |
| Perception in the control loop | **not started** — the controller still reads privileged state | — |
| Intel Core Ultra Series 2/3 benchmark numbers | **not measured, and cannot be measured here** | see *Hardware* below |

### Three defects found by measurement, and what they cost

The first version of this controller scored 9/50 and no placement at all. The
three reasons were measured, not guessed, and two of them are fixed here.

1. **The stationary jaw was being planned inside the object.** Only one of the
   SO-101's two jaws moves, so aiming the *meeting point* of the jaws at an
   object puts the fixed jaw on top of it. On seed 0 that put the fixed jaw
   7.1 mm inside the mug wall, 4.9 mm from the fork handle's centre against its
   6 mm half-width, and 0.4 mm inside the plate rim. Every one of those three
   objects was touched **only by the fixed jaw**, on every seed, and shoved
   away instead of gripped. `plan_pose(..., standoff=)` now backs the meeting
   point off along the live jaw axis so the fixed jaw clears the surface.
2. **The arms were being driven through the cabinet.** Waypoints are reached by
   ramping joint targets in a straight line, and from the home pose that line
   sweeps the carcass: of 25 probe waypoints at table height, **18 were missed
   by 40–300 mm**, most of them parked against the cabinet at x≈0.10, z≈0.94.
   One via-point east of the carcass recovered them — 274 mm of error became
   13 mm on the same target. The plate phase now starts from home and routes
   around the corner.
3. **The servos saturate at about 0.30 m of horizontal reach.** At a waypoint
   0.297 m out, `shoulder_pan`, `shoulder_lift` and `elbow_flex` were all at
   their ±2.94 N·m limit with **no contact anywhere on the arm**. That is not a
   kinematic limit — random joint sampling reaches 0.46 m — so IK reports a
   solved pose and the arm simply never gets there. It is why the plate is
   dragged rather than carried, and it is unfixed: it is a property of the
   robot, not of the code.

### What the controller actually does, measured

Same environment, same scorer, 10 seeds, the only difference the controller:

| Run | Sub-goals | Task success | Evidence |
|---|---|---|---|
| `--policy none` (arms hold the home pose) | **0 / 50** | 0 / 10 | `evidence/eval_seeds.json` |
| `--policy scripted` | **16 / 50** | 0 / 10 | `evidence/eval_seeds_scripted.json` |

Broken out: `drawer_open` **10/10**, `plate_placed` **6/10**, and
`fork_placed`, `spoon_placed`, `mug_placed` **0/10 each**. Read the rest of the
row before reading anything into it:

- **The task has never been completed.** `task_success` is 0/10 and
  `in_order_prefix` is 1 on every seed — the drawer, and then the first gap.
  The plate is scored out of order, so it adds a point and no sequencing.
- **Three of the four placements have never fired.** The fork and the spoon are
  never picked out of the drawer and the mug is never lifted.
- The plate is not *carried*. It is hooked by its rim and dragged flat across
  the table (see below). The scorer asks for the plate on the mat, upright and
  resting, and does not ask how it got there — but a drag is not a pick and
  place, and the write-up says so.
- `bimanual`, meaning both arms touched a manipulable object, is true on
  **7 seeds of 10**; on 2 only the right arm touches one and on 1 only the left.
- `handoff_occurred` is true on 10/10 and that number is misleading: every
  recorded hand-off is on the **drawer**, the two arms taking its handle in
  turn during the end-of-episode re-check. **No object hand-off has been
  achieved on any seed.** The script asks for two; it gets none.
- Nothing is dropped on any seed, and every episode is numerically stable.
- The worst single planning residual across the ten episodes is 295 mm: one
  waypoint in a plate correction cycle that the IK could not reach at all.
  That move is a no-op rather than a failure, but it is not a solved plan.

The demo video is captioned with the live predicate state frame by frame, so
it shows those failures rather than hiding them.

The no-policy run is the negative control for that number: any sub-goal that
fired with the arms held still would mean the scorer, not the controller,
produced it. It fires none.

## Quick start

```bash
pip install -r requirements.txt
python3 scripts/build_scene.py            # writes envs/dinner_table.xml
python3 scripts/verify_scene.py           # 16 structural + physical checks
python3 scripts/test_task_predicates.py   # 11 accept/reject controls on the scorer
python3 scripts/eval_seeds.py --seeds 10                    # control: no policy
python3 scripts/eval_seeds.py --seeds 10 --policy scripted  # the controller
python3 scripts/record_demo.py --seeds 10                   # the demo video

# perception + OpenVINO
python3 scripts/make_perception_dataset.py --split train --compiles 500 --per-compile 8
python3 scripts/make_perception_dataset.py --split val   --compiles 60  --per-compile 4
python3 scripts/make_perception_dataset.py --split eval10
python3 scripts/train_perception.py --epochs 80     # ~4 min on one L40S
python3 scripts/export_openvino.py                  # ONNX -> IR at FP32/FP16/INT8
python3 scripts/bench_openvino.py                   # Required Deliverable 3
python3 scripts/test_perception_pipeline.py         # 13 accept/reject controls

# the architecture summary, and the controls that keep its numbers true
python3 scripts/test_technical_summary.py          # stdlib only -- runs before pip install

# the submission cover image, and the controls that keep it honest
python3 scripts/render_cover.py                    # 16:9 PNG from a live rollout
python3 scripts/test_cover_image.py                # 14 controls, stdlib only

# the mandatory slide deck, and the controls that read the shipped PDF back
python3 scripts/make_slides.py                     # 13-slide 16:9 PDF from evidence/
python3 scripts/test_slides.py                     # 12 controls on the PDF bytes
```

Headless rendering uses EGL (`MUJOCO_GL=egl`, set by the scripts). On a machine
without a GPU, `MUJOCO_GL=osmesa` works too.

## The scene

Two Robot Studio SO-101 arms — 5 positioning joints and a parallel jaw each,
12 actuators in total — are mounted 0.44 m apart on a shared table, yawed 36°
inward so their neutral pose faces the workspace. In front of them sits a
cabinet with a prismatic drawer holding a spoon and a fork, plus a plate, a mug
and a bottle on the table.

Two things about the cabinet are set by what a gripper can physically do, and
both were measured rather than guessed. Its carcass is tall enough to leave
177 mm of clearance above the open drawer rim: with a low top panel the only
route to the cutlery is a 25 mm slot between the drawer face and the overhang,
which no SO-101 gripper fits through, and the drawer becomes decorative. And
the cutlery lies *across* the drawer rather than along it, because lengthwise
it has to sit near the drawer face, where the SO-101's wrist camera mount fouls
the face on the way down and the arm stalls with its servos saturated.

- `envs/dinner_table.py` — the scene, as code. `build(dims=...)` returns an
  uncompiled `MjSpec` so object geometry can be varied per episode.
- `envs/dinner_table.xml` — the nominal scene, generated by
  `scripts/build_scene.py`. 48 DoF, 12 actuators, 133 geoms, 5 cameras
  (`scene_cam`, `front_cam`, `top_cam` and one wrist camera per arm).
- `envs/ik.py` — damped-least-squares site IK, used to solve the `home`
  keyframe and available to the controller.
- `envs/randomize.py` — the randomizer.
- `envs/task.py` — the instruction, the sub-goals and the scorer.
- `envs/controller.py` — the scripted bimanual controller: a waypoint state
  machine over position IK. It holds no learned parameters and reads no
  camera; what it does read from the simulator is the world pose of the object
  it is about to touch, which is what a perception stack would supply. Three
  pieces do the work — a jaw-tip calibration swept out of the model, a
  `wrist_roll` alignment that squares the jaws onto what they are gripping,
  and an IK loop that targets the point where the jaws MEET rather than the
  wrist frame (they differ by ~41 mm, and that difference is why an
  unmodified position-IK grasp brushes past everything it reaches for), with a
  standoff so the one jaw that does NOT move ends up clear of the object rather
  than inside it.

Verified properties (`scripts/verify_scene.py`, all 16 pass):
the saved XML loads standalone; both arms expose their full 6-actuator set and
a wrist camera; the `home` keyframe holds against gravity to within
4×10⁻⁴ rad; the scene is quiescent after 3 s (max |qvel| 5.6×10⁻⁴); the drawer
travels its full 90 mm and carries the cutlery with it; every manipulable
object lies within 0.40 m of at least one arm base; the hand-off site is
0.229 m from both; and all five cameras render offscreen.

## Randomization

Three stages, because only the first needs a recompile. Ranges are in
`envs/randomize.py::RANGES` and every draw is logged per episode.

| Stage | What varies |
|---|---|
| geometry | plate radius, mug radius and height, bottle radius and height, cutlery length |
| model | per-object mass (0.6–1.6×), sliding friction (0.6–1.4×), key-light position and intensity, table and floor lightness |
| state | object x, y and yaw, initial drawer opening |

Placements are rejection-sampled against three conditions — inside an arm's
reach, not overlapping another object, not blocking the drawer — so a failed
episode is a policy failure rather than an impossible scene. Across seeds 0–9
every episode was on-table, reachable, free of initial interpenetration, and
numerically stable for the full episode, in both the no-policy control (4 s)
and the scripted run (111–158 s).

## Task and scoring

> Open the top drawer, pick up the fork and the spoon and lay them either side
> of the place setting, put the plate on the mat, then set the mug down to the
> right of the plate.

Five sub-goals: `drawer_open`, `fork_placed`, `spoon_placed`, `plate_placed`,
`mug_placed`. Placement tolerances are 45–50 mm; the plate and mug must also
still be upright within 25°; an object that leaves the table is marked dropped
and cannot score. `TaskMonitor` additionally records sequencing (the longest
completed prefix of the intended order), which arms touched an object, and any
hand-off — one arm taking an object the other was holding.

The scorer is deliberately independent of any controller, so the same numbers
mean the same thing for a scripted rollout, a learned policy, or a human
teleoperating the arms.

Both directions are tested. `scripts/test_task_predicates.py` drives an
ACCEPT control that puts the world in the solved state and requires
`task_success` to be true, then a REJECT control per predicate: each object
displaced 100 mm, the drawer opened only 40 mm, the mug at the target but
rotated 90°, the plate at the target x,y but on the floor, and a single-armed
grasp that must not be reported as a hand-off.

## The plate: a hook and a drag, not a grasp

The plate is 92–116 mm across after randomization and the jaws span 101 mm
open, so there is no opening at which both jaws clear it and then close on it:
whichever way the meeting point is aimed, one jaw sweeps across the plate's own
face. What the rim is good for is a hook. Its boxes stand 5 mm proud of the
top face, so a jaw dropped inside them catches the inner wall and the plate can
be dragged flat instead of lifted — which also keeps it upright and resting,
which is what the scorer asks of it, and keeps the arm inside the 0.30 m
envelope that carrying it at arm's length would leave.

Two parts of that are closed loops rather than fixed waypoints, and both are
there because an open-loop version was measured failing:

- **The descent searches for contact.** The rim is 8 mm tall and the arm tracks
  a commanded pose to 12–20 mm, so a single aim at the middle of that window
  misses more often than it hits. The controller steps down 4 mm at a time
  until a jaw actually touches the plate.
- **The drag re-aims from where the plate is.** Each waypoint is the jaws'
  current position plus a share of the plate's *own* remaining offset, read at
  the moment the move starts. Nothing about the grasp is cached, so a slip
  changes the next waypoint instead of being carried forward. If the plate is
  still short of the mat at the end, the whole cycle repeats, twice.

It works on 6 seeds of 10. On the other 4 the hook never engages and the plate
is left between 80 mm and 300 mm from the mat.

## Robustness and recovery

The controller re-checks the drawer at the end of the episode and re-opens it
from the home pose if a later reach has nudged it shut. That is not cosmetic:
before the re-check the drawer reached its full 90 mm travel on 10 seeds of 10
and was then knocked closed again on 5 of them, so the sub-goal the robot had
genuinely performed was scored as unearned. The re-check is a plain
if-then-redo, logged in the rollout trace as `recheck_fired`.

Two details of it were forced by measurement. The re-check runs **twice**,
because once left 3 of 4 knocked-shut seeds still shut. And the drawer is
released with the jaws **narrow**, not wide: opening them fully swings the
moving jaw through 80 mm and that arc catches the drawer front and shoves it
back — 92 mm of travel collapsing to 45 mm across the release alone, under the
60 mm the scorer wants. With both, `drawer_open` is 10/10.

Raising the drawer's slide friction so it would not drift was tried first and
made things worse — 2/50, because the arm could no longer pull it fully open.
That change was reverted; the friction in the scene is the original 0.35.

## Perception and OpenVINO

The scripted controller reads object positions out of `MjData`. That is
privileged information a real robot does not have, and it is also why this
project had no inference cost to optimise. `envs/perception.py` is the first
piece that replaces it.

**What it is.** A 0.52 M-parameter CNN that takes one 128×224 `top_cam` frame
and regresses seven numbers: the planar centres of the plate, mug and bottle,
and how far the drawer is out. Four stride-2 conv blocks, then a **spatial
softmax** — a global average pool cannot do coordinate regression, because
averaging over space discards the position being asked for; a spatial softmax
turns each channel into a soft keypoint and keeps it.

**What it is not.** It is not a policy, it emits scene state rather than
actions, and **it is not in the control loop**. The 16/50 sub-goal result is
still produced by the privileged scripted controller and is unchanged by
anything here.

**Data.** Rendered from the simulator, labelled by the simulator: 4,000 training
frames from compile seeds 1000–1499, 240 validation frames from 2000–2059, and
the ten evaluation seeds at exactly their scored initial state. The splits are
disjoint *by compile seed*, not by shuffling.

| | worst object centre | drawer travel |
|---|---|---|
| model, validation (unseen seeds) | **1.3 mm** | 0.2 mm |
| model, the ten evaluation seeds | **1.8 mm** | 0.15 mm |
| no-vision baseline (train-set mean layout) | 34.3 mm | 22.1 mm |
| the same model on random-noise images | 306 mm | — |

The last two rows are the point: a metric that cannot fail proves nothing.
`scripts/test_perception_pipeline.py` adds a causal control — move the mug
60 mm in the scene, re-render, re-predict: the mug prediction moves 59.0 mm and
the untouched plate prediction moves 0.4 mm.

### Conversion and precision

`scripts/export_openvino.py` goes PyTorch → ONNX (opset 17) → OpenVINO IR at
three precisions, and checks each one against the PyTorch model it came from,
in millimetres of table position rather than tensor norms:

| precision | weights | error on the evaluation seeds | max drift vs PyTorch |
|---|---|---|---|
| PyTorch FP32 | 2058 KiB | 1.83 mm | — |
| IR FP32 | 2031 KiB | 1.81 mm | 0.45 mm |
| IR FP16 | 1015 KiB | 2.04 mm | 1.12 mm |
| IR INT8 (NNCF PTQ, 300 calibration frames) | **515 KiB** | 3.98 mm | 5.87 mm |

INT8 is 4× smaller and costs 2.2 mm of accuracy. That cost is reported, not
buried: `test_perception_pipeline.py` has a control that **fails** if the INT8
error is ever recorded as no worse than FP32.

The FP32 IR is not bit-identical to PyTorch, and the reason is measured rather
than asserted — the same IR is run twice, once at the plugin default and once
with `INFERENCE_PRECISION_HINT=f32`:

| execution precision | max drift vs PyTorch |
|---|---|
| plugin default (`bfloat16` on this host) | 0.4488 mm |
| forced `f32` | 0.00009 mm |

The drift is the CPU plugin's own precision choice, not a conversion loss.

## Hardware — an open gap

The track brief requires the final demonstration and benchmark to run on an
**Intel Core Ultra Series 2/3** system with OpenVINO. The machine this was
developed on is AMD EPYC 9654 with NVIDIA L40S GPUs and has no Intel silicon.

`scripts/bench_openvino.py` — Required Deliverable 3 — is written and runs. It
enumerates every OpenVINO device on the host and, for each device and each
exported precision, reports single-stream latency (mean, p50, p90, p99),
async throughput at the plugin's own optimal request count, the execution
precision the plugin chose, and the model's task quality in millimetres on the
same ten seeds the task result is quoted on.

Run on this machine it produces nine device/precision rows and stamps the
report `NOT_THE_REQUIRED_MEASUREMENT`:

```
CPU/FP32   0.436 ms p50   5220 fps async   1.81 mm   (bfloat16)
CPU/INT8   0.555 ms p50   5424 fps async   3.98 mm   (bfloat16)
GPU.0/FP16 0.604 ms p50   4792 fps async   1.82 mm   (float16)
…                       host: AMD EPYC 9654 96-Core Processor
```

**Those are not Intel Core Ultra numbers and are not offered as any.** The
script decides that itself: `required_hardware_verdict()` reads the host CPU
name and returns `MEASURED_ON_REQUIRED_HARDWARE` only for a Core Ultra part.
Because that branch cannot fire on this machine, it is exercised directly by
`test_perception_pipeline.py` against three real Core Ultra model strings and
five non-Core-Ultra ones — an accept path that is never run is how a gate ships
broken.

To get the figure the track scores, run `python3 scripts/bench_openvino.py`
unchanged on a Core Ultra Series 2/3 machine. It will pick up `NPU` and the
Intel iGPU automatically if their drivers are present, and the report it writes
will say `MEASURED_ON_REQUIRED_HARDWARE`.

## Licence

MIT, in `LICENSE`. The SO-101 model under `third_party/` is Apache-2.0 from
`mujoco_menagerie`; see `NOTICE`.
