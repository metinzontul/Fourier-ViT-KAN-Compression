# -*- coding: utf-8 -*-
"""
Step 3 - ISIC 2019 ViT fine-tuning on the lesion-grouped train/val/test split (R3.3, R3.4, R1.8).

Training protocol is unchanged from the original ISIC scripts
(isic2019_vit_bct_kan_ablation.py / isic2019_vit_bct_mlp_ablation.py):
    ViT-B/16 (IMAGENET1K_V1), all parameters trained, AdamW lr 5e-5 wd 1e-4, 20 epochs,
    batch 32, random horizontal/vertical flips, class-weighted cross-entropy.
Changes required by the reviewers:
    * lesion-grouped stratified split (no lesion shared between partitions)
    * epoch selected on the validation split (best Macro F1), metrics REPORTED on the test split
    * per-epoch logs, spline-grid coverage of the head input, optional checkpoint saving

Configurations (BCT, KAN and the 768->128 width are the original operators from heads.py):
    vit_kan, vit_bct_kan, vit_linear, vit_bct_linear  (the four configurations of Table 4)
    vit_bct_ln_kan                                    (mechanism test: LayerNorm before the KAN)

Runs are ordered seed-major (all configurations for seed 42, then 43, ...) so that a partial
night still yields balanced comparisons. Finished (config, seed) pairs are skipped on restart.

    python isic_finetune.py                                   # 4 main configs x seeds 42 43 44
    python isic_finetune.py --configs vit_bct_ln_kan          # mechanism test
    python isic_finetune.py --save_ckpt_seed 42               # keep best checkpoints of seed 42
"""
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

import common
import heads as H

CONFIGS = ["vit_kan", "vit_bct_kan", "vit_linear", "vit_bct_linear", "vit_bct_ln_kan"]
MAIN = CONFIGS[:4]


class ViTHead(nn.Module):
    def __init__(self, config, num_classes, width=128):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        self.reducer, d = (nn.Identity(), 768) if "bct" not in config else (H.SpectralBCT(768, width), width)
        self.norm = nn.LayerNorm(d) if "ln" in config else nn.Identity()
        self.head = H.SplineKAN(d, num_classes) if config.endswith("kan") else nn.Linear(d, num_classes)

    def forward(self, x):
        return self.head(self.norm(self.reducer(self.backbone(x))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, crit, amp):
    model.eval()
    captured = []
    hook = model.head.register_forward_hook(lambda m, inp, out: captured.append(inp[0].float().cpu()))
    preds, targets, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        loss_sum += crit(out.float(), y).item() * x.size(0)
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    hook.remove()
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    z = torch.cat(captured)
    return dict(loss=loss_sum / len(t), acc=100 * (p == t).mean(),
                f1=100 * f1_score(t, p, average="macro", zero_division=0),
                prec=100 * precision_score(t, p, average="macro", zero_division=0),
                rec=100 * recall_score(t, p, average="macro", zero_division=0),
                head_in_std=z.std().item(), head_in_grid=H.spline_grid_coverage(z)), p, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=MAIN, choices=CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (off = original fp32 protocol)")
    ap.add_argument("--save_ckpt_seed", type=int, default=None, help="save best-val checkpoints for this seed")
    args = ap.parse_args()
    common.tee_console("isic_finetune")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(common.RESULT_DIR, "isic_finetune")
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "summary.csv")
    epochs_csv = os.path.join(out_dir, "epochs.csv")
    done = set()
    if os.path.exists(summary_csv):
        s = pd.read_csv(summary_csv)
        done = set(zip(s.config, s.seed))

    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    t_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(), transforms.ToTensor(), norm])
    t_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    tr = common._isic_splits(t_train)["train"]
    ev = common._isic_splits(t_eval)
    C = len(common.ISIC_CLASSES)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(ev["val"], batch_size=64, shuffle=False, **kw)
    test_loader = DataLoader(ev["test"], batch_size=64, shuffle=False, **kw)

    counts = np.bincount(tr.labels, minlength=C)
    class_w = torch.tensor(len(tr) / (C * counts), dtype=torch.float32, device=device)  # same formula as original
    crit = nn.CrossEntropyLoss(weight=class_w)
    print(f">> train/val/test = {len(tr)}/{len(ev['val'])}/{len(ev['test'])}  class weights={np.round(class_w.cpu().numpy(), 4)}")
    print(f">> configs={args.configs} seeds={args.seeds} bf16={args.bf16}")

    for seed in args.seeds:
        for cfg in args.configs:
            if (cfg, seed) in done:
                print(f"   skip {cfg} seed {seed} (done)")
                continue
            set_seed(seed)
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, generator=g, **kw)
            model = ViTHead(cfg, C).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            p_module = H.count_params(model.reducer) + H.count_params(model.norm) + H.count_params(model.head)
            p_total = H.count_params(model)
            best, t_start = dict(val_f1=-1), time.time()
            print(f"\n[{cfg} | seed {seed}] module params={p_module:,} total params={p_total:,}")

            for ep in range(1, args.epochs + 1):
                model.train()
                t0, loss_sum, n = time.time(), 0.0, 0
                for x, y in train_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                        loss = crit(model(x).float(), y)
                    loss.backward()
                    opt.step()
                    loss_sum += loss.item() * x.size(0)
                    n += x.size(0)
                vm, _, _ = evaluate(model, val_loader, device, crit, args.bf16)
                tm, tp, tt = evaluate(model, test_loader, device, crit, args.bf16)
                row = dict(config=cfg, seed=seed, epoch=ep, train_loss=loss_sum / n,
                           **{f"val_{k}": v for k, v in vm.items()}, **{f"test_{k}": v for k, v in tm.items()},
                           epoch_seconds=time.time() - t0)
                pd.DataFrame([row]).to_csv(epochs_csv, mode="a", header=not os.path.exists(epochs_csv), index=False)
                flag = ""
                if vm["f1"] > best["val_f1"]:
                    best = dict(val_f1=vm["f1"], val_acc=vm["acc"], best_epoch=ep,
                                **{f"test_{k}": v for k, v in tm.items()})
                    np.save(os.path.join(out_dir, f"cm_{cfg}_seed{seed}.npy"), confusion_matrix(tt, tp, labels=range(C)))
                    if args.save_ckpt_seed == seed:
                        torch.save(model.state_dict(), os.path.join(out_dir, f"best_{cfg}_seed{seed}.pth"))
                    flag = " *"
                print(f"    Epoch [{ep:02d}/{args.epochs}] train_loss {row['train_loss']:.4f} | "
                      f"val F1 {vm['f1']:.2f} acc {vm['acc']:.2f} | test F1 {tm['f1']:.2f} acc {tm['acc']:.2f} | "
                      f"head-in grid {vm['head_in_grid']:.2f} | {row['epoch_seconds']:.0f}s{flag}")

            summ = dict(config=cfg, seed=seed, **best, last_test_f1=tm["f1"], last_test_acc=tm["acc"],
                        params_module=p_module, params_total=p_total, bf16=args.bf16,
                        minutes=(time.time() - t_start) / 60)
            pd.DataFrame([summ]).to_csv(summary_csv, mode="a", header=not os.path.exists(summary_csv), index=False)
            print(f"   => best epoch {best['best_epoch']}: val F1 {best['val_f1']:.2f} | TEST F1 {best['test_f1']:.2f} "
                  f"| {summ['minutes']:.1f} min")
            del model, opt
            torch.cuda.empty_cache()
    print(">> done")


if __name__ == "__main__":
    main()
