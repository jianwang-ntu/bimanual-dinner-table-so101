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

``--train-splits`` and ``--val-splits`` take more than one dataset because the
distribution the control loop asks about is not the one the first model was
fitted on.  ``perception_train.npz`` holds only home-keyframe frames -- no arm
is ever over the table in it -- and a model trained on that alone drifts by two
orders of magnitude once ``envs/scene_source.py`` starts asking it mid-rollout.
``perception_train_rollout.npz`` is the same scenes with the arms where the
controller actually puts them.  Error is reported per validation split as well
as pooled, so the home-pose number stays visible and cannot be averaged away.

Run:  python3 scripts/train_perception.py --epochs 60
      python3 scripts/train_perception.py --epochs 60 \
          --train-splits train train_rollout --val-splits val val_rollout
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


def _derangement(n: int, rng) -> np.ndarray:
    """A permutation with no fixed point, so nothing is paired with itself."""
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    return perm


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
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--val-splits", nargs="+", default=["val"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    def load_many(splits):
        parts = [load(sp) for sp in splits]
        return (np.concatenate([a for a, _, _ in parts]),
                np.concatenate([b for _, b, _ in parts]),
                np.concatenate([c for _, _, c in parts]),
                {sp: len(a) for sp, (a, _, _) in zip(splits, parts)})

    xtr, ytr, str_, n_tr = load_many(args.train_splits)
    xva, yva, sva, n_va = load_many(args.val_splits)
    xev, yev, sev = load("eval10")
    va_parts = {}                       # split -> slice into the pooled val set
    at = 0
    for sp in args.val_splits:
        va_parts[sp] = slice(at, at + n_va[sp])
        at += n_va[sp]
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
    perm = _derangement(len(yva), rng)

    report = {
        "schema": "perception_train/v1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"class": "SceneStateCNN", "parameters": int(n_par),
                  "input": list(P.IN_SHAPE), "outputs": list(P.OUT_NAMES),
                  "head": "spatial softmax over a 8x14 feature map -> MLP",
                  "checkpoint": str(ckpt.relative_to(ROOT))},
        "data": {"train": int(len(xtr)), "val": int(len(xva)),
                 "eval10": int(len(xev)),
                 "train_splits": {k: int(v) for k, v in n_tr.items()},
                 "val_splits": {k: int(v) for k, v in n_va.items()},
                 "split_rule": "disjoint compile seeds -- train 1000..1499, "
                               "val 2000..2059, eval10 0..9. The _rollout "
                               "splits reuse their own side's seed base "
                               "(train_rollout 1000.., val_rollout 2000..), so "
                               "the train/val/eval separation is unchanged.",
                 "eval10_note": "the ten seeds scripts/eval_seeds.py scores, at "
                                "their evaluation initial state"},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                     "optimiser": "AdamW + OneCycle", "loss": "SmoothL1(beta=0.02)",
                     "seed": args.seed, "device": str(device),
                     "device_name": (torch.cuda.get_device_name(0)
                                     if device.type == "cuda" else platform.processor()),
                     "seconds": round(time.time() - t0, 1),
                     "selection": "best val worst-centre + drawer error"},
        "error_mm": dict(
            {"val": P.position_error_mm(pred_va, yva),
             "eval10": P.position_error_mm(pred_ev, yev)},
            **{f"val::{sp}": P.position_error_mm(pred_va[sl], yva[sl])
               for sp, sl in va_parts.items()}),
        "error_mm_note": "'val' is the pooled validation set. The 'val::<split>' "
                         "rows break it out so a split the model is bad at "
                         "cannot be hidden inside a pooled mean.",
        "controls": dict({
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
            "why_per_split": "A pooled baseline is not a fair bar once the "
                "validation set spans two regimes. The constant predictor is "
                "much worse on mid-rollout frames -- objects have been dragged "
                "away from the nominal layout it predicts -- so pooling flatters "
                "the model on the easy split and punishes it on the hard one. "
                "Each split therefore carries its own baseline and its own "
                "shuffled control, and the suite checks every one of them.",
        },
            **{f"constant_baseline_val::{sp}": P.position_error_mm(
                   np.repeat(const, yva[sl].shape[0], axis=0), yva[sl])
               for sp, sl in va_parts.items()},
            **{f"shuffled_labels_val::{sp}": P.position_error_mm(
                   pred_va[sl], yva[sl][_derangement(yva[sl].shape[0], rng)])
               for sp, sl in va_parts.items()}),
        "history": history,
        "not_claimed": [
            "This is a perception model, not a policy. It outputs scene state, "
            "not actions. envs/scene_source.py can put it inside the control "
            "loop, and evidence/eval_seeds_scripted_perceived.json is what "
            "happens when it is -- that file, not this one, is where the task "
            "result under perception is reported.",
            "Nothing here is a natural-language capability. The network takes "
            "pixels and emits seven numbers; no instruction is parsed anywhere "
            "in this repository.",
            "Position only. There is no output for object yaw, object height or "
            "object size, and the fork and the spoon -- which start inside the "
            "drawer -- are not regressed at all.",
            "The error quoted for a split is only as representative as that "
            "split. Read error_mm['val::<split>'], not the pooled row, before "
            "believing a number covers a regime.",
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
