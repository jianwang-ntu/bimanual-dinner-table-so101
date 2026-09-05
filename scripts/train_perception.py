#!/usr/bin/env python3
"""Train the scene-state regressor and report its error against two controls.

The number that matters is not the loss, it is millimetres of position error on
scenes the network never saw, next to a baseline that has no vision at all:

  constant baseline   always predicts the training-set mean layout.  Any model
                      that does not beat this has learned nothing from pixels.
  shuffled control    the same val predictions re-scored against permuted
                      labels.  Pairing two independent draws, it must land at
                      or above the constant baseline; if it ever landed near
                      the model's own error the metric would not be reading the
                      image at all.

Both are computed here, on the same tensors, by the same code path.

Run:  python3 scripts/train_perception.py --epochs 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
from torch import nn                                    # noqa: E402

from envs import perception as P                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load(split: str):
    f = DATA / f"perception_{split}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run scripts/make_perception_dataset.py "
                         f"--split {split} first")
    d = np.load(f)
    return d["images"], d["labels"].astype(np.float32), d["seeds"]


def to_batches(images, labels, device, batch, shuffle, generator=None):
    n = len(images)
    order = (torch.randperm(n, generator=generator).numpy() if shuffle
             else np.arange(n))
    for i in range(0, n, batch):
        idx = order[i:i + batch]
        x = torch.from_numpy(P.preprocess(images[idx])).to(device)
        y = torch.from_numpy(labels[idx]).to(device)
        yield x, y


@torch.no_grad()
def predict(model, images, device, batch=128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(P.preprocess(images[i:i + batch])).to(device)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    xtr, ytr, str_ = load("train")
    xva, yva, sva = load("val")
    xev, yev, sev = load("eval10")
    assert not (set(str_.tolist()) & set(sva.tolist())), "train/val seed overlap"
    assert not (set(str_.tolist()) & set(sev.tolist())), "train/eval seed overlap"

    model = P.build_model().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, (len(xtr) + args.batch - 1) // args.batch))
    lossf = nn.SmoothL1Loss(beta=0.02)
    gen = torch.Generator().manual_seed(args.seed)

    history, best, best_state, t0 = [], float("inf"), None, time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in to_batches(xtr, ytr, device, args.batch, True, gen):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        val_err = P.position_error_mm(predict(model, xva, device), yva)
        history.append({"epoch": ep, "train_loss": round(tot / nb, 6),
                        "val_worst_centre_mm": round(val_err["worst_centre_mm"], 2),
                        "val_drawer_mm": round(val_err["drawer_q"], 2)})
        score = val_err["worst_centre_mm"] + val_err["drawer_q"]
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  loss {tot/nb:.5f}  "
                  f"val worst-centre {val_err['worst_centre_mm']:6.1f} mm  "
                  f"drawer {val_err['drawer_q']:5.1f} mm", flush=True)

    model.load_state_dict(best_state)
    MODELS.mkdir(exist_ok=True)
    ckpt = MODELS / "scene_state_cnn.pt"
    torch.save({"state_dict": best_state,
                "outputs": list(P.OUT_NAMES),
                "in_shape": list(P.IN_SHAPE),
                "camera": P.CAMERA,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": vars(args)}, ckpt)

    # ------------------------------------------------------------- the report
    pred_va = predict(model, xva, device)
    pred_ev = predict(model, xev, device)
    const = ytr.mean(0, keepdims=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(yva))
    while np.any(perm == np.arange(len(yva))):           # no fixed points
        perm = rng.permutation(len(yva))

    report = {
        "schema": "perception_train/v1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"class": "SceneStateCNN", "parameters": int(n_par),
                  "input": list(P.IN_SHAPE), "outputs": list(P.OUT_NAMES),
                  "head": "spatial softmax over a 8x14 feature map -> MLP",
                  "checkpoint": str(ckpt.relative_to(ROOT))},
        "data": {"train": int(len(xtr)), "val": int(len(xva)),
                 "eval10": int(len(xev)),
                 "split_rule": "disjoint compile seeds -- train 1000..1499, "
                               "val 2000..2059, eval10 0..9",
                 "eval10_note": "the ten seeds scripts/eval_seeds.py scores, at "
                                "their evaluation initial state"},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                     "optimiser": "AdamW + OneCycle", "loss": "SmoothL1(beta=0.02)",
                     "seed": args.seed, "device": str(device),
                     "device_name": (torch.cuda.get_device_name(0)
                                     if device.type == "cuda" else platform.processor()),
                     "seconds": round(time.time() - t0, 1),
                     "selection": "best val worst-centre + drawer error"},
        "error_mm": {
            "val": P.position_error_mm(pred_va, yva),
            "eval10": P.position_error_mm(pred_ev, yev),
        },
        "controls": {
            "constant_baseline_val": P.position_error_mm(
                np.repeat(const, len(yva), axis=0), yva),
            "constant_baseline_eval10": P.position_error_mm(
                np.repeat(const, len(yev), axis=0), yev),
            "shuffled_labels_val": P.position_error_mm(pred_va, yva[perm]),
            "what_the_controls_mean":
                "constant_baseline is the no-vision predictor: the train-set mean "
                "layout, scored the same way -- a model that does not beat it has "
                "learned nothing from pixels. shuffled_labels_val re-scores the "
                "SAME predictions against permuted labels; pairing two independent "
                "draws it lands at or above the constant baseline, and if it ever "
                "landed near the model's own error the metric would not be reading "
                "the image at all.",
        },
        "history": history,
        "not_claimed": [
            "This is a perception model, not a policy. It outputs scene state, "
            "not actions, and it is not in the control loop -- the 16/50 sub-goal "
            "result in evidence/eval_seeds_scripted.json is unchanged by it.",
            "Trained and evaluated on initial states with both arms at the home "
            "keyframe. Mid-rollout frames, where the arms occlude the table, are "
            "not measured here.",
            "The fork and the spoon start inside the drawer and are not regressed.",
        ],
    }
    out = ROOT / "evidence" / "perception_train.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    e = report["error_mm"]
    c = report["controls"]
    print("\nval    worst object centre %6.1f mm   (constant baseline %6.1f mm)"
          % (e["val"]["worst_centre_mm"], c["constant_baseline_val"]["worst_centre_mm"]))
    print("eval10 worst object centre %6.1f mm   (constant baseline %6.1f mm)"
          % (e["eval10"]["worst_centre_mm"],
             c["constant_baseline_eval10"]["worst_centre_mm"]))
    print("val    drawer travel        %6.1f mm   (constant baseline %6.1f mm)"
          % (e["val"]["drawer_q"], c["constant_baseline_val"]["drawer_q"]))
    print("shuffled-label control     %6.1f mm   (must be >= the baseline)"
          % c["shuffled_labels_val"]["worst_centre_mm"])
    print(f"\nwrote {out.relative_to(ROOT)} and {ckpt.relative_to(ROOT)} "
          f"({n_par} parameters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
