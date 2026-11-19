# -*- coding: utf-8 -*-
"""
Diagnostic: ResNet50 + SpectralBCT shows validation F1 ~92 but test F1 15-86 on EuroSAT
(same distribution). Hypothesis: predictions depend on the evaluation chunk size
(test is evaluated in 2 chunks of <=4096, validation in 1), i.e. a numerical problem in the
batched FFT when the DC bin is very large (ResNet features are non-negative).

Trains one BCT+Linear model (seed 0) on cached EuroSAT/ResNet50 features, then evaluates
the SAME model on val/test with different chunk sizes, on GPU and on CPU.

    python diagnose_bct_eval.py
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import common
import heads as H


def load(split, dev):
    z = np.load(common.feature_path("EuroSAT", "resnet50", split))
    return torch.from_numpy(z["X"]).to(dev), torch.from_numpy(z["y"]).to(dev)


@torch.no_grad()
def f1(model, X, y, bs):
    model.eval()
    p = torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).cpu().numpy()
    return 100 * f1_score(y.cpu().numpy(), p, average="macro")


@torch.no_grad()
def max_diff(model, X, bs_a, bs_b):
    a = torch.cat([model(X[i:i + bs_a]) for i in range(0, len(X), bs_a)])
    b = torch.cat([model(X[i:i + bs_b]) for i in range(0, len(X), bs_b)])
    return (a - b).abs().max().item(), a.abs().max().item()


def main():
    dev = torch.device("cuda")
    Xtr, ytr = load("train", dev)
    Xva, yva = load("val", dev)
    Xte, yte = load("test", dev)
    torch.manual_seed(0)
    model = nn.Sequential(H.SpectralBCT(2048, 128), nn.Linear(128, 10)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(0)
    for ep in range(20):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            crit(model(Xtr[b]), ytr[b]).backward()
            opt.step()

    with torch.no_grad():
        z = model[0](Xva[:512])
        print(f"BCT output: mean {z.mean().item():.3g}  std {z.std().item():.3g}  max|z| {z.abs().max().item():.3g}")
    print("\nGPU, macro F1 by evaluation chunk size")
    for bs in (64, 256, 1024, 2700, 4096, 6000):
        print(f"  chunk {bs:5d}:  val {f1(model, Xva, yva, bs):6.2f}   test {f1(model, Xte, yte, bs):6.2f}")
    d, m = max_diff(model, Xte, 4096, 64)
    print(f"  max |logit difference| chunk 4096 vs 64 on test: {d:.3g} (max |logit| {m:.3g})")

    cpu = model.cpu()
    print("\nCPU, macro F1 by evaluation chunk size")
    for bs in (64, 4096, 6000):
        print(f"  chunk {bs:5d}:  val {f1(cpu, Xva.cpu(), yva.cpu(), bs):6.2f}   test {f1(cpu, Xte.cpu(), yte.cpu(), bs):6.2f}")


if __name__ == "__main__":
    main()
