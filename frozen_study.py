# -*- coding: utf-8 -*-
"""
Step 2 - Controlled frozen-feature study (answers R1.5, R1.8, R2.2, R2.4, R2.5, R3.1, R3.2).

Every configuration uses the same cached features, the same optimiser, the same
training length, the same splits and the same selection rule:
    * Adam, 20 epochs, batch 256, class-weighted CE for ISIC 2019 (as in the original ISIC runs)
    * learning rate chosen per (dataset, backbone, reducer, width, head) on the validation split
      using seed 0 (stage "tune"), then 5 seeds at the chosen rate (stage "final")
    * epoch chosen by best validation Macro F1; test metrics reported at that epoch

Results are appended to results/frozen_results.csv; finished runs are skipped on restart.

Examples:
    python frozen_study.py --preset core      # d=128, all datasets/backbones/reducers/heads
    python frozen_study.py --preset widths    # bottleneck-width sweep on ISIC (both backbones) + CIFAR10-ViT
    python frozen_study.py --datasets ISIC2019 --backbones vit_b_16 --reducers bct linear --heads kan kan_ln
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import common
import heads as H

PRESETS = {
    "core": dict(datasets=common.DATASETS, backbones=common.BACKBONES,
                 reducers=["none", "bct", "blockcirc", "linear", "pca", "random"],
                 heads=["linear", "mlp", "kan", "kan_ln"], widths=[128]),
    "widths": dict(datasets=["ISIC2019", "CIFAR10"], backbones=common.BACKBONES,
                   reducers=["bct", "blockcirc", "linear", "pca", "random"],
                   heads=["linear", "mlp", "kan", "kan_ln"], widths=[32, 64, 256, 384]),
}
KEY = ["dataset", "backbone", "reducer", "width", "head", "lr", "seed", "stage"]


def load_split(ds, bb, split, device):
    z = np.load(common.feature_path(ds, bb, split))
    return torch.from_numpy(z["X"]).to(device), torch.from_numpy(z["y"]).to(device), int(z["num_classes"])


# Evaluation chunk size. cuFFT (CUDA 12.1, torch 2.5.1) returns wrong rFFT/irFFT results for
# length-2048 inputs at batch sizes >= ~4096 (verified against CPU in diagnose_bct_eval.py);
# 1024 is exact on GPU and CPU.
EVAL_CHUNK = 1024


@torch.no_grad()
def evaluate(model, X, y, bs=EVAL_CHUNK):
    model.eval()
    pred = torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).cpu().numpy()
    yt = y.cpu().numpy()
    return 100 * f1_score(yt, pred, average="macro", zero_division=0), 100 * (pred == yt).mean()


def train_one(data, reducer_kind, width, head_kind, lr, seed, args, X_train_np):
    (Xtr, ytr), (Xva, yva), (Xte, yte), C = data
    D = Xtr.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    red, d = H.make_reducer(reducer_kind, D, width, X_train=X_train_np, seed=seed, cache_key=args.feat_key)
    head = H.make_head(head_kind, d, C, mlp_hidden=args.mlp_hidden)
    model = nn.Sequential(red, head).to(Xtr.device)

    weight = None
    if args.class_weighted:
        counts = torch.bincount(ytr, minlength=C).float()
        weight = (len(ytr) / (C * counts.clamp(min=1)))
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)

    best = dict(val_f1=-1.0)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(Xtr.device)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        vf1, vacc = evaluate(model, Xva, yva)
        tf1, tacc = evaluate(model, Xte, yte)
        if vf1 > best["val_f1"]:
            best = dict(val_f1=vf1, val_acc=vacc, test_f1=tf1, test_acc=tacc, best_epoch=ep)
    last = dict(last_test_f1=tf1, last_test_acc=tacc, last_val_f1=vf1, final_train_loss=float(loss))

    with torch.no_grad():  # where do head inputs fall relative to the spline grid?
        z = red(Xva[:2048])
        if isinstance(head, nn.Sequential) and isinstance(head[0], nn.LayerNorm):
            z = head[0](z)                      # kan_ln: measure the input the spline actually sees
        cov = H.spline_grid_coverage(z)
        z_std = z.std().item()
    p_red, p_head = H.count_params(red), H.count_params(head)
    bct = red[-1] if isinstance(red, nn.Sequential) else red
    p_eff = bct.effective_params() + p_head if hasattr(bct, "effective_params") else p_red + p_head
    return dict(**best, **last, params_reducer=p_red, params_head=p_head, params_module=p_red + p_head,
                params_module_effective=p_eff, head_input_std=z_std, head_input_in_grid=cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--backbones", nargs="+")
    ap.add_argument("--reducers", nargs="+")
    ap.add_argument("--heads", nargs="+")
    ap.add_argument("--widths", nargs="+", type=int)
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(common.RESULT_DIR, "frozen_results.csv"))
    args = ap.parse_args()
    common.tee_console("frozen_study")

    cfg = dict(PRESETS[args.preset]) if args.preset else dict(PRESETS["core"])
    for k in ("datasets", "backbones", "reducers", "heads", "widths"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(map(tuple, prev[KEY].astype(str).values))
    print(f">> device={device} out={args.out} finished runs on disk: {len(done)}")

    for ds in cfg["datasets"]:
        args.class_weighted = (ds == "ISIC2019")
        for bb in cfg["backbones"]:
            Xtr, ytr, C = load_split(ds, bb, "train", device)
            Xva, yva, _ = load_split(ds, bb, "val", device)
            Xte, yte, _ = load_split(ds, bb, "test", device)
            data = ((Xtr, ytr), (Xva, yva), (Xte, yte), C)
            Xtr_np = Xtr.cpu().numpy() if "pca" in cfg["reducers"] else None
            args.feat_key = (ds, bb)
            for red in cfg["reducers"]:
                widths = [Xtr.shape[1]] if red == "none" else cfg["widths"]
                for w in widths:
                    for hd in cfg["heads"]:
                        def run(lr, seed, stage):
                            key = (ds, bb, red, str(w), hd, str(lr), str(seed), stage)
                            if tuple(map(str, key)) in done:
                                return None
                            t0 = time.time()
                            r = train_one(data, red, w, hd, lr, seed, args, Xtr_np)
                            row = dict(zip(KEY, [ds, bb, red, w, hd, lr, seed, stage]), **r, seconds=time.time() - t0)
                            pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
                            done.add(tuple(map(str, key)))
                            print(f"   {ds:8s} {bb:8s} {red:9s} d={w:<4d} {hd:6s} lr={lr:.0e} s={seed} {stage:5s} "
                                  f"valF1={r['val_f1']:.2f} testF1={r['test_f1']:.2f} "
                                  f"params={r['params_module']:,} grid={r['head_input_in_grid']:.2f} {row['seconds']:.1f}s")
                            return r

                        # stage 1: LR selection on validation (seed 0)
                        for lr in args.lrs:
                            run(lr, 0, "tune")
                        res = pd.read_csv(args.out)
                        m = (res.dataset == ds) & (res.backbone == bb) & (res.reducer == red) & \
                            (res.width == w) & (res["head"] == hd) & (res.stage == "tune")
                        best_lr = float(res[m].sort_values("val_f1", ascending=False).iloc[0].lr)
                        # stage 2: all seeds at the selected LR
                        for s in args.seeds:
                            run(best_lr, s, "final")
            del Xtr, Xva, Xte
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(">> done")


if __name__ == "__main__":
    main()
