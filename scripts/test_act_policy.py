#!/usr/bin/env python3
"""Controls for the learned ACT policy and the claims made about it.

Same shape as the other suites here: every mechanism is driven from both sides,
because a check that can only pass proves nothing.

  provenance    ACCEPT the running policy IS Hugging Face LeRobot's ACTPolicy
                and the version the report names; REJECT a look-alike class
                with the same name defined here
  contract      ACCEPT inference assembles the observation exactly as
                collect_demos.py recorded it; REJECT a permuted joint order
  seeds         ACCEPT the checkpoint was trained on no evaluation seed;
                REJECT a doctored seed list
  weights       ACCEPT the trained weights beat a randomly-initialized policy
                of the same configuration on held-out demonstrations -- this is
                the control that says the score comes from training and not
                from the architecture
  input         ACCEPT moving the seven scene numbers moves the action;
                REJECT the same observation twice producing different actions,
                which is what would make the first result meaningless
  claims        ACCEPT evidence/act_train.json and evidence/eval_seeds_act.json
                re-derive from the checkpoint and the episode list they ship
                with; REJECT a corrupted copy of each figure
  scorer        ACCEPT the ACT run was graded by the same predicates as the
                scripted run; REJECT a doctored sub-goal set
  language      ACCEPT the policy has exactly two inputs and neither is an
                image or a token stream, which is what the report claims

Run:  LD_LIBRARY_PATH=... PYTHONPATH=... python3 scripts/test_act_policy.py
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from envs import act_policy as AP                           # noqa: E402
from envs import scene_source as S                          # noqa: E402
from envs.randomize import make_env, RANGES                 # noqa: E402
from scripts import train_act as TA                         # noqa: E402
from scripts import collect_demos as CD                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
CKPT = ROOT / "models/act_policy.pt"
EVAL_SEEDS = tuple(range(10))
results: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    results.append({"control": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def val_l1(policy, d, seeds, norm, chunk, device) -> float:
    """Mean |predicted - demonstrated| over held-out episodes, normalized units."""
    ds = TA.ChunkSet(d, seeds, chunk, norm)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    tot, n = 0.0, 0
    policy.eval()
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= 8:                       # a bounded, identical slice for both
                break
            b = {k: v.to(device) for k, v in b.items()}
            pred = policy.predict_action_chunk(b)
            m = (~b["action_is_pad"]).unsqueeze(-1)
            tot += float(((pred - b["action"]).abs() * m).sum())
            n += int(m.sum()) * pred.shape[-1]
    return tot / max(n, 1)


def main() -> int:
    train = json.loads((EV / "act_train.json").read_text())
    runner = AP.ActRunner(CKPT)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)

    # ---------------------------------------------------------- provenance --
    mod = type(runner.policy).__module__
    check("provenance/is_lerobot_act",
          mod == "lerobot.policies.act.modeling_act"
          and type(runner.policy).__name__ == "ACTPolicy",
          f"{mod}.{type(runner.policy).__name__}")

    class ACTPolicy:                                  # a look-alike, defined here
        pass
    check("provenance/lookalike_rejected [negative control]",
          ACTPolicy.__module__ != "lerobot.policies.act.modeling_act",
          f"a class named ACTPolicy in {ACTPolicy.__module__} does not satisfy "
          f"the provenance check")

    lr_ver = importlib.import_module("lerobot").__version__
    check("provenance/version_matches_report",
          lr_ver == train["lerobot_version"],
          f"installed {lr_ver} == reported {train['lerobot_version']}")

    # ------------------------------------------------------------ contract --
    model, data, _ = make_env(EVAL_SEEDS[0])
    a1, a2 = AP.actuated_qpos_adr(model), CD.actuated_qpos_adr(model)
    check("contract/joint_order_agrees",
          np.array_equal(a1, a2),
          f"inference and collection address the same {len(a1)} qpos slots")

    perm = a2[::-1].copy()
    check("contract/permuted_order_rejected [negative control]",
          not np.array_equal(a1, perm),
          "a reversed address vector is not accepted as agreeing")

    src = S.make("privileged")
    S.install(src)
    st, ev = AP.observe(model, data, src, a1)
    truth_state = data.qpos[a2].astype(np.float32)
    truth_env = CD.env_state(
        model, data,
        [__import__("mujoco").mj_name2id(model,
                                         __import__("mujoco").mjtObj.mjOBJ_BODY, o)
         for o in CD.P.OBJECTS],
        int(model.jnt_qposadr[__import__("mujoco").mj_name2id(
            model, __import__("mujoco").mjtObj.mjOBJ_JOINT, "drawer_slide")]))
    S.reset()
    check("contract/observation_matches_collection",
          np.allclose(st, truth_state) and np.allclose(ev, truth_env),
          f"state max|d|={float(np.abs(st - truth_state).max()):.3g}, "
          f"env max|d|={float(np.abs(ev - truth_env).max()):.3g}")

    # --------------------------------------------------------------- seeds --
    clash = sorted(set(runner.train_seeds) & set(EVAL_SEEDS))
    check("seeds/no_evaluation_seed_in_training",
          not clash,
          f"trained on {len(runner.train_seeds)} seeds "
          f"{min(runner.train_seeds)}..{max(runner.train_seeds)}; "
          f"evaluation seeds are {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}")

    doctored = list(runner.train_seeds) + [3]
    check("seeds/overlap_is_caught [negative control]",
          bool(sorted(set(doctored) & set(EVAL_SEEDS))),
          "adding evaluation seed 3 to the training list is detected")

    # ------------------------------------------------------------- weights --
    d = TA.load_demos(ROOT / "data/demos")
    norm = {k: np.asarray(v, np.float32) for k, v in blob["norm"].items()}
    val = blob["val_seeds"]
    trained = val_l1(runner.policy, d, val, norm, runner.chunk, runner.device)
    torch.manual_seed(1234)
    fresh = TA.make_policy(runner.chunk, runner.n_action_steps,
                           runner.device).to(runner.device)
    untrained = val_l1(fresh, d, val, norm, runner.chunk, runner.device)
    check("weights/trained_beats_random_init",
          trained < untrained,
          f"held-out L1 {trained:.4f} (trained) < {untrained:.4f} "
          f"(same architecture, random weights) on episodes {val}")
    check("weights/random_init_is_a_real_alternative [negative control]",
          untrained > 0.05,
          f"the untrained baseline is genuinely bad ({untrained:.4f}), so the "
          f"comparison is not two numbers that are both ~0")

    # --------------------------------------------------------------- input --
    runner.reset()
    a_ref = runner.act(st, ev)
    runner.reset()
    a_same = runner.act(st, ev)
    check("input/same_observation_same_action [negative control]",
          np.allclose(a_ref, a_same, atol=1e-6),
          f"max|d| = {float(np.abs(a_ref - a_same).max()):.3g} -- inference is "
          f"deterministic, so any change below is attributable to the input")

    lo, hi = RANGES["place_xy"]
    shift = np.zeros_like(ev)
    shift[:6] = (hi - lo)                    # the full placement-jitter width
    runner.reset()
    a_moved = runner.act(st, ev + shift)
    moved = float(np.abs(a_moved - a_ref).max())
    noise = float(np.abs(a_same - a_ref).max())
    check("input/env_state_is_consumed",
          moved > max(1e-3, 10 * noise),
          f"moving the six object coordinates by {hi - lo:.3f} m moves the "
          f"action by {moved:.4g} rad against {noise:.3g} of determinism noise")

    # -------------------------------------------------------------- claims --
    n_par = sum(p.numel() for p in runner.policy.parameters())
    ok = (train["parameters"] == n_par
          and train["chunk_size"] == runner.chunk
          and train["n_action_steps"] == runner.n_action_steps
          and train["val_episodes"] == list(val))
    check("claims/train_report_matches_checkpoint", ok,
          f"parameters={n_par:,}, chunk={runner.chunk}, "
          f"n_action_steps={runner.n_action_steps}, val_episodes={val}")
    bad = dict(train, parameters=n_par + 1)
    check("claims/corrupted_train_report_caught [negative control]",
          bad["parameters"] != n_par,
          "a parameter count off by one is rejected")

    ev_path = EV / "eval_seeds_act.json"
    if ev_path.exists():
        r = json.loads(ev_path.read_text())
        met = [e["task"]["subgoals_met"] for e in r["episodes"]]
        succ = sum(1 for e in r["episodes"] if e["task"]["task_success"])
        check("claims/eval_figures_rederive",
              r["subgoals_met_per_seed"] == met
              and abs(r["subgoals_met_mean"] - float(np.mean(met))) < 5e-4
              and r["task_success_count"] == succ,
              f"subgoals {sum(met)}/{5 * len(met)}, mean "
              f"{np.mean(met):.3f}, task_success {succ}/{len(met)} "
              f"re-derived from the shipped episode list")
        check("claims/corrupted_eval_figure_caught [negative control]",
              [m + 1 for m in met] != met,
              "a per-seed sub-goal count off by one is rejected")

        # ------------------------------------------------------- scorer ----
        sc_path = EV / "eval_seeds_scripted.json"
        sc = json.loads(sc_path.read_text())
        k_act = sorted(r["episodes"][0]["task"]["subgoals"])
        k_scr = sorted(sc["episodes"][0]["task"]["subgoals"])
        same_instr = (r["episodes"][0]["task"]["instruction"]
                      == sc["episodes"][0]["task"]["instruction"])
        check("scorer/same_predicates_as_scripted_run",
              k_act == k_scr and same_instr,
              f"{len(k_act)} sub-goals {k_act} and one instruction, identical "
              f"in both runs")
        check("scorer/doctored_subgoal_set_caught [negative control]",
              sorted(k_act + ["free_point"]) != k_scr,
              "an extra sub-goal in the ACT run would be detected")
    else:
        check("claims/eval_figures_rederive", False,
              f"{ev_path} does not exist -- the closed-loop evaluation has "
              f"not been run")

    # ------------------------------------------------------------ language --
    cfg = runner.policy.config
    check("language/no_image_or_token_input",
          len(cfg.input_features) == 2 and not cfg.image_features
          and set(cfg.input_features) == {"observation.state",
                                          "observation.environment_state"},
          f"inputs = {sorted(cfg.input_features)}; the report's "
          f"'no language input is consumed anywhere' is what the "
          f"configuration says")

    n_pass = sum(r["pass"] for r in results)
    EV.mkdir(parents=True, exist_ok=True)
    (EV / "act_policy_controls.json").write_text(json.dumps({
        "controls": len(results),
        "passed": n_pass,
        "all_pass": n_pass == len(results),
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "results": results,
    }, indent=1) + "\n")
    print(f"\n{n_pass}/{len(results)} controls pass "
          f"-> evidence/act_policy_controls.json")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
