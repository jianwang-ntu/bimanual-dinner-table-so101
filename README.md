# Bimanual dinner-table manipulation — dual SO-101 in MuJoCo

Simulation package for the **Bimanual VLA Manipulation with Multi-Modal
Reasoning** online track (Intel Physical AI Online Challenge, AI Infra Summit
hackathon, lablab.ai). Target scenario: *Setting Up a Dinner Table*.

## What is in here, and what is not

This repository contains **the environment, the scorer, a scripted
controller that solves two sub-goals of five on 5 seeds of 10 and never the
whole task, and — since 2026-09-05 — a learned ACT policy that is worse than
that scripted controller.** Read that before reading anything else.

The learned policy is real and it is not the headline. A LeRobot ACT
(action-chunking transformer) trained by behaviour cloning on 35 rollouts of the
scripted controller scores **3 sub-goals of 50** over the same ten evaluation
seeds, against the scripted controller's **15 of 50** — with *more* simulator
time per episode, not less. Every number this README quotes is the scripted
controller's unless the row says `ACT`.

Since 2026-09-06 the controller can also be run with **no privileged object
poses at all** — see *Perception in the control loop* below. That run scores
**12 / 50**, three sub-goals below the privileged one, and the privileged
number is what this README quotes unless a row says otherwise.

| Piece | State | Evidence |
|---|---|---|
| Dual SO-101 MuJoCo scene, drawer, plate, mug, bottle, cutlery | built, verified | `evidence/scene_verification.json` (16/16) |
| Seeded domain randomization — geometry, mass, friction, lighting, background, placement | built, verified over 10 seeds; **placement covers 3 of the 5 graspables** — the fork and the spoon are fixed | `evidence/eval_seeds.json` + `evidence/randomization_coverage_controls.json` |
| Task definition and success predicates | built, controls both ways | `evidence/task_predicate_controls.json` (11/11) |
| Scripted bimanual controller | built, **15 sub-goals of 50 over 10 seeds** | `evidence/eval_seeds_scripted.json` |
| Demo video, 10 randomized seeds, captioned from the simulator | recorded, **shows a partial rollout** | `evidence/demo_scripted_10seeds.mp4` + `.json` |
| Scene-state perception CNN, trained on rendered frames | built, **2.09 mm on unoccluded / 24.94 mm on mid-rollout frames** | `evidence/perception_train.json` |
| OpenVINO conversion — FP32, FP16, INT8 (NNCF) | built, accuracy of each measured | `evidence/openvino_export.json` |
| Bench-test script — latency, throughput, device, precision | built, **run here on AMD, not on Intel** | `evidence/openvino_bench_*.json` |
| Controls for all of the above, accept and reject | 18/18 | `evidence/perception_pipeline_controls.json` |
| Technical summary / architecture — Required Deliverable 5 | written, claims machine-checked | [`TECHNICAL_SUMMARY.md`](TECHNICAL_SUMMARY.md), `evidence/technical_summary_controls.json` (11/11) |
| Submission cover image, 1920×1080 PNG rendered from the simulator | rendered, **captioned from `evidence/`, not typed** | `evidence/cover_image.png` + `.json`, `evidence/cover_image_controls.json` (14/14) |
| Slide presentation, 13 slides, 16:9 PDF | generated from `evidence/`, **no figure typed by hand** | `evidence/slides_presentation.pdf` + `.json`, `evidence/slides_controls.json` (12/12) |
| Video presentation, 4:36, 1280×720 MP4 | generated from `evidence/`, **real footage pasted unscaled from the demo MP4** | `evidence/video_presentation.mp4` + `.json`, `evidence/video_controls.json` (38/38) |
| Perception in the control loop | built, **12 / 50 perceived vs 15 / 50 privileged vs 9 / 50 blind** | `evidence/eval_seeds_scripted_perceived.json`, `evidence/scene_source_controls.json` (15/15) |
| VLA / imitation policy — LeRobot ACT, behaviour-cloned | built, **3 / 50 sub-goals — five times worse than its own demonstrator** | `evidence/act_train.json`, `evidence/eval_seeds_act.json`, `evidence/act_policy_controls.json` (19/19) |
| Demonstrations the policy was trained on, 40 episodes on seeds 3000–3039 | recorded and committed, 8.4 MB | `data/demos/`, `scripts/collect_demos.py` |
| Language conditioning, multi-step task context, re-planning | **not started** — three of T2's four demands | — |
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
| `--policy scripted` | **15 / 50** | 0 / 10 | `evidence/eval_seeds_scripted.json` |

Broken out: `drawer_open` **10/10**, `plate_placed` **4/10**,
`mug_placed` **1/10**, and `fork_placed`, `spoon_placed` **0/10 each**. Read the
rest of the row before reading anything into it:

- **The task has never been completed.** `task_success` is 0/10 and
  `in_order_prefix` is 1 on every seed — the drawer, and then the first gap.
  The plate is scored out of order, so it adds a point and no sequencing.
- **Two of the four placements have never fired.** The fork and the spoon are
  never picked out of the drawer. The mug is gripped and lifted on 8 seeds of
  10 and reaches its mat on **1** — the only object other than the plate this
  entry has ever placed, and it arrives lying on its side (see below).
- The plate is not *carried*. It is hooked by its rim and dragged flat across
  the table (see below). The scorer asks for the plate on the mat, upright and
  resting, and does not ask how it got there — but a drag is not a pick and
  place, and the write-up says so.
- `bimanual`, meaning both arms touched a manipulable object, is true on
  **10 seeds of 10**.
- `handoff_occurred` is true on 10/10, and most of what it counts is still the
  **drawer** — the two arms taking its handle in turn during the end-of-episode
  re-check. But it is no longer only that: the **mug** is passed from the left
  arm to the right on **5 seeds of 10** (0, 2, 6, 8, 9). The script asks for
  two object hand-offs; it gets one, on half the seeds.
- **The total went down, not up.** Squaring the jaws onto the object (see
  `plan_pose_squared`) is what earned the mug; it also cost the plate on seeds
  2 and 8, where the arms now cross the table carrying the mug after the plate
  is already down and nudge it outside its 50 mm tolerance. Sub-goals went
  **16/50 to 15/50**, mean 1.6 to **1.5**. The trade is stated in both
  directions because it is a trade: the first object hand-off and the first
  non-plate placement, bought with two plates.
- **The mug that is placed is lying on its side.** On seed 8 it finishes 9.8 mm
  from its mat — well inside tolerance — with an upright cosine of 0.000
  against the 0.906 bar the scorer wants, so it scores the placement and not
  the pose. It is gripped 5 mm below the rim because that is where the scene
  puts `mug_grasp`, and a 57 mm mug dragged from its top rim tips. Gripping at
  the base, at mid-height, and lifting clear were all measured and all worse.
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

# the learned ACT policy: demonstrations, training, closed-loop evaluation
python3 scripts/collect_demos.py --seed-start 3000 --count 40 --out data/demos/demos_3000.npz
python3 scripts/train_act.py --steps 6000 --out models/act_policy.pt
python3 scripts/eval_seeds.py --seeds 10 --policy act --scene privileged --no-render
python3 scripts/test_act_policy.py

# the mandatory video presentation, and the controls that decode it back
python3 scripts/make_video.py                      # 4:36 1280x720 MP4 from evidence/
python3 scripts/test_video.py                      # 38 controls on the DECODED PIXELS
```

`scripts/make_video.py` needs `ffmpeg` on the path as well as the Python
packages; `scripts/test_video.py` rebuilds the video and requires the result to
be **byte-identical** to the shipped file, so a hand-edited MP4 fails.

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
| state | object x, y and yaw for the plate, the mug and the bottle; initial drawer opening |

**The fork and the spoon are not placement-randomized.** `NOMINAL_XY` names
three bodies and `GRASPABLES` names five: the cutlery starts in the drawer at
the same x, y and yaw on every seed, 0.347 m and 0.387 m from the nearest arm
base on all ten. Its *length* and its *mass* are randomized — 6.5 mm and 0.85×
of spread across the run — so this is a gap in placement only, and it is a gap
against the track's own wording ("randomized object placement"). Closing it
means jittering the cutlery inside the drawer and re-scoring; that has not been
done, and until it is, `fork_placed` and `spoon_placed` are 0/10 against one
fixed layout rather than ten. `scripts/test_randomization_coverage.py` reads
both tables out of the randomizer and pins this.

Placements are rejection-sampled against three conditions — inside an arm's
reach, not overlapping another object, not blocking the drawer — so a failed
episode is a policy failure rather than an impossible scene. The reach
condition is a 0.40 m planar envelope; `scripts/measure_reach.py` drives the
right arm out along the bearing of the fork and it puts its jaws at 0.417 m of
planar radius, so the envelope is not optimistic and the cutlery's fixed
placement is not an out-of-reach one. Across seeds 0–9 every episode was
on-table, reachable, free of initial interpenetration, and numerically stable
for the full episode, in both the no-policy control (4 s) and the scripted run
(111–158 s).

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

It works on 4 seeds of 10. On the other 6 the hook never engages and the plate
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

**What it is not.** It is not a policy — it emits scene state, not actions —
and it does not read or produce language. It *can* now be put in the control
loop; see the next section for what that costs.

**Data.** Rendered from the simulator, labelled by the simulator: 6,990 training
frames and 840 validation frames, over two regimes. The `home` splits are the
original ones — both arms at the home keyframe, nothing over the table. The
`rollout` splits are frames taken every 1,000 physics steps *during a scripted
episode*, so the arm poses, the occlusions and the mid-manipulation object
positions are the ones the control loop actually asks about. The splits are
disjoint *by compile seed*, not by shuffling.

| | worst object centre | drawer travel |
|---|---|---|
| model, unoccluded validation frames | **2.09 mm** | 0.25 mm |
| model, the ten evaluation seeds | **2.39 mm** | 0.21 mm |
| model, **mid-rollout** validation frames | **24.94 mm** | 0.87 mm |
| no-vision baseline, unoccluded / mid-rollout | 40.5 mm / 101.9 mm | 28.9 mm |
| the same predictions against shuffled labels | 48.6 mm / 130.1 mm | — |
| the same model on random-noise images | 314 mm | — |

The bottom three rows are the point: a metric that cannot fail proves nothing.
Read them together and the honest summary is that the model is 19× better than
no vision where the object is visible and **4.1× better where an arm is over
it** — it is reading the image in both regimes (the shuffled null is 5.2× worse
even on the hard one), it is just not reading it well when there is nothing to
read. `scripts/test_perception_pipeline.py` checks every one of those ratios
per split rather than pooling them, and adds a causal control: move the mug
60 mm in the scene, re-render, re-predict, and see whether the mug prediction
moves and the untouched plate prediction does not.

## Perception in the control loop

`envs/scene_source.py` is the seam. Every object position the controller reads
goes through it, and three sources can be installed:

| `--scene` | what the controller reads | sub-goals |
|---|---|---|
| `privileged` | `MjData`, as it always did | **15 / 50** |
| `perceived` | one `top_cam` frame per planning instant → OpenVINO IR → seven numbers | **12 / 50** |
| `blind` | the nominal, un-randomized layout from `envs/randomize.py` | **9 / 50** |

```bash
python3 scripts/eval_seeds.py --seeds 10 --policy scripted --scene perceived
python3 scripts/eval_seeds.py --seeds 10 --policy scripted --scene blind
python3 scripts/test_scene_source.py     # 15/15 on the seam itself
```

The `blind` row is the negative control and it is why the other two mean
anything: if the controller ignored what it was handed, all three rows would be
identical. They are not.

**What is and is not replaced.** Under `perceived`, the planar centres of the
plate, mug and bottle and the drawer opening come from the network, and a site
welded to one of those bodies — `plate_grasp`, `mug_grasp`,
`drawer_handle_site` — moves with its parent's estimate. Object **height**,
object **yaw**, object **dimensions**, and the **spoon and fork** are still
read from the simulator: the network has no output for any of them. The
scorer in `envs/task.py` is never routed through the seam, and
`test_scene_source.py` checks that by displacing every object 90 mm and
confirming the score does not move.

**Where the 3 sub-goals go.** Not evenly. The drawer is never occluded, is
estimated to 1.8 mm, and `drawer_open` survives at 9/10. The plate spends most
of the episode underneath the arm dragging it, is estimated to 53 mm, and
`plate_placed` drops from 4/10 to 3/10 with the mug lost entirely. The cost
lands exactly where the model cannot see.

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

## A learned policy: ACT, and it loses to the script

Track Objective 4 asks for a policy *trained* in MuJoCo — "using Hugging Face
LeRobot or compatible tooling … SmolVLA, Pi0.5, ACT, or another appropriate
VLA / imitation-learning policy". Until now this repository had none, and said
so. It now has one, and the honest headline is that **it is worse than the
scripted controller it was cloned from.**

What it is: `lerobot.policies.act.modeling_act.ACTPolicy` — the real LeRobot
class, version 0.6.1, not a re-implementation — with 40,158,924 parameters, a
chunk size of 50 and 25 actions executed per network query. Its inputs are
twelve actuated joint positions and the seven scene numbers
`envs/perception.py` regresses; its output is twelve actuator position targets.

How it was trained: `scripts/collect_demos.py` replays the scripted controller
on **seeds 3000–3039** — never an evaluation seed — and logs
`(state, env_state, action)` at 20 Hz, 40 episodes and 186,670 samples, 8.4 MB,
committed under `data/demos/`. `scripts/train_act.py` then behaviour-clones
6,000 steps at batch 128 in 286 s on one L40S. Held-out L1 on five episodes the
optimiser never saw is **0.167** in normalized action units, against **0.810**
for the same architecture with random weights.

How it does, closed-loop, on the same ten seeds and the same untouched scorer:

| | scripted controller | ACT policy |
|---|---|---|
| sub-goals over 10 seeds | **15 / 50** | **3 / 50** |
| task success | 0 / 10 | 0 / 10 |
| simulator seconds per episode | 206.2 | **242.8** |
| what it ever achieves | drawer 10/10, plate 4/10, mug 1/10 | drawer 2/10 (seeds 5, 7), plate 1/10 (seed 4) |

The policy was given **more** simulator time than its demonstrator, not less —
its horizon is the median demonstration length — so the gap is not a clipped
episode. Behaviour cloning on 35 long-horizon episodes recovers about a fifth of
the demonstrator's sub-goal rate, and the reasons are ordinary and unfixed here:
no image input, no temporal ensembling, one seed's worth of coverage per
episode, and 4,856 open-loop-ish policy steps per episode in which small errors
compound.

`scripts/test_act_policy.py` is 19 controls, each with the reject side that
makes the accept side mean something — that the class really is LeRobot's, that
inference assembles the observation in exactly the byte order
`collect_demos.py` recorded it, that no evaluation seed is in the training set,
that the trained weights beat a random-init policy of the same shape, that
moving the seven scene numbers moves the action while the same observation twice
does not, and that the figures above re-derive from the shipped episode list.

The checkpoint itself is **160.8 MB and is not in the repository** — over
GitHub's file limit. The demonstrations are, so
`python3 scripts/train_act.py` rebuilds it from a clean clone.

**Not claimed.** This is imitation of a script. It consumes no language: the
instruction string in `envs/task.py` reaches nothing, and the policy is
configured with exactly two inputs, neither of which is an image or a token
stream. It does not make the entry's demo video, cover image, slide deck or
demo page — all of those still show the scripted controller, because that is
still the better of the two.

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
