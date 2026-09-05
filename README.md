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
| VLA / imitation policy | **not started** | — |
| OpenVINO conversion and quantization | **not started** | — |
| Intel Core Ultra Series 2/3 benchmark | **not started, and cannot be run here** | see *Hardware* below |

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

## Hardware — an open gap

The track brief requires the final demonstration and benchmark to run on an
**Intel Core Ultra Series 2/3** system with OpenVINO. The machine this was
developed on is AMD EPYC with NVIDIA L40S GPUs and has no Intel silicon, so no
Core Ultra latency, throughput or NPU/iGPU utilisation figure has been
measured, and none is reported anywhere in this repository. Publishing AMD
numbers under an Intel heading would be false, so the row is left empty.

## Licence

MIT, in `LICENSE`. The SO-101 model under `third_party/` is Apache-2.0 from
`mujoco_menagerie`; see `NOTICE`.
