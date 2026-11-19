# -*- coding: utf-8 -*-
"""
Step 6 - Standardised efficiency measurements (R1.1, R3.5).

One protocol for every configuration (backbone x reducer x head, d = 128):
  * parameters  : real-valued trainable count (complex parameters count twice), for the
                  reducer+head module and for the whole model (backbone + module)
  * FLOPs       : backbone measured with torch.utils.flop_counter (matmul/conv); module FLOPs
                  computed analytically (formulas below), because FFTs and the elementwise
                  B-spline basis are not counted by the PyTorch counter
  * latency     : fp32, eval mode, torch.no_grad, 20 warm-up + 100 timed iterations,
                  torch.cuda.synchronize around every iteration; median and IQR reported
                    - module alone on GPU (batch 1 and 64) and on CPU (1 thread, batch 1)
                    - whole model on GPU (batch 1 and 64)
  * peak memory : torch.cuda.max_memory_allocated during a batch-64 forward pass of the whole model
Weights are random: timings and counts do not depend on the trained values.

Analytic module FLOPs per sample (multiply-add = 2 FLOPs; FFT of length N ~ 2.5 N log2 N real FLOPs):
  linear D->C          : 2 D C
  mlp D->h->C          : 2 (D h + h C)
  spline KAN D->C      : base 2 D C + spline 2 D C (G+k) + basis evaluation ~ 6 D (G+2k+1) k
  LayerNorm            : ~ 8 D
  spectral BCT D->d    : rFFT(D) + d complex products (6 d) + irFFT(d) + d
  block-circulant D->d : rFFT per input block + (d/b)(D/b)(b/2+1) complex MACs (8 each) + irFFT per output block
  linear/PCA/random D->d : 2 D d

    python benchmark_efficiency.py
"""
import os
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

import common
import heads as H

REDUCERS = ["none", "bct", "blockcirc", "linear", "pca", "random"]
HEADS = ["linear", "mlp", "kan", "kan_ln"]
G, KORD, B = 5, 3, 16


def fft_flops(n):
    return 2.5 * n * math.log2(n)


def module_flops(red, hd, D, d, C, hidden=128):
    f = 0.0
    if red == "bct":
        f += fft_flops(D) + 6 * d + fft_flops(d) + d
    elif red == "blockcirc":
        f += (D // B) * fft_flops(B) + 8 * (d // B) * (D // B) * (B // 2 + 1) + (d // B) * fft_flops(B) + d
    elif red in ("linear", "pca", "random"):
        f += 2 * D * d
    din = D if red == "none" else d
    if hd == "linear":
        f += 2 * din * C
    elif hd == "mlp":
        f += 2 * (din * hidden + hidden * C)
    else:
        if hd == "kan_ln":
            f += 8 * din
        f += 2 * din * C + 2 * din * C * (G + KORD) + 6 * din * (G + 2 * KORD + 1) * KORD
    return f


def build_backbone(name):
    from torchvision.models import vit_b_16, resnet50
    m = vit_b_16(weights=None) if name == "vit_b_16" else resnet50(weights=None)
    if name == "vit_b_16":
        m.heads = nn.Identity()
    else:
        m.fc = nn.Identity()
    return m


def build_module(red, hd, D, d, C):
    if red in ("pca", "random"):
        # fixed projection of the right shape; values do not affect cost (and avoids numpy/MKL
        # linear algebra, whose OpenMP runtime clashes with PyTorch's on Windows conda installs)
        r, din = H.FixedProjection(torch.randn(D, d) / math.sqrt(D), torch.zeros(D)), d
    else:
        r, din = H.make_reducer(red, D, d, seed=0)
    return nn.Sequential(r, H.make_head(hd, din, C))


@torch.no_grad()
def time_it(fn, device, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 75) - np.percentile(ts, 25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=common.BACKBONES)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    common.tee_console("benchmark_efficiency")
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    torch.set_num_threads(1)              # CPU measurements: single thread
    C, d = args.classes, args.width
    out = os.path.join(common.RESULT_DIR, "efficiency.csv")
    os.makedirs(common.RESULT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.backbone, prev.reducer, prev["head"]))
    print(f">> device={gpu} torch={torch.__version__} "
          f"{torch.cuda.get_device_name(0) if gpu.type == 'cuda' else 'cpu'}  d={d} C={C}")

    for bb in args.backbones:
        backbone = build_backbone(bb).eval().to(gpu)
        D = 768 if bb == "vit_b_16" else 2048
        x1, x64 = torch.randn(1, 3, 224, 224, device=gpu), torch.randn(64, 3, 224, 224, device=gpu)
        # Count in train mode with autograd enabled: in eval/no-grad mode torchvision's ViT uses the fused
        # attention fast path, which the FLOP counter does not see. (ViT dropout is 0; FLOPs are identical.)
        backbone.train()
        with FlopCounterMode(display=False) as fc:
            backbone(x1)
        backbone.eval()
        bb_flops = fc.get_total_flops()
        bb_params = H.count_params(backbone)
        print(f"   {bb}: backbone {bb_params:,} params, {bb_flops / 1e9:.2f} GFLOPs/sample (counter)")

        for red in REDUCERS:
            for hd in HEADS:
                if (bb, red, hd) in done:
                    print(f"   skip {bb} {red} {hd} (done)")
                    continue
                mod = build_module(red, hd, D, d, C).eval()
                p_mod = H.count_params(mod)
                f_mod = module_flops(red, hd, D, d, C)
                mod_g = mod.to(gpu)
                z1, z64 = torch.randn(1, D, device=gpu), torch.randn(64, D, device=gpu)
                lat = {}
                lat["module_gpu_b1"] = time_it(lambda: mod_g(z1), gpu, iters=args.iters)
                lat["module_gpu_b64"] = time_it(lambda: mod_g(z64), gpu, iters=args.iters)
                mod_c = build_module(red, hd, D, d, C).eval().to(cpu)
                zc = torch.randn(1, D)
                lat["module_cpu1t_b1"] = time_it(lambda: mod_c(zc), cpu, iters=args.iters)
                full = nn.Sequential(backbone, mod_g).eval()
                lat["full_gpu_b1"] = time_it(lambda: full(x1), gpu, iters=args.iters)
                if gpu.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                lat["full_gpu_b64"] = time_it(lambda: full(x64), gpu, warmup=5, iters=max(20, args.iters // 5))
                peak = torch.cuda.max_memory_allocated() / 2 ** 20 if gpu.type == "cuda" else float("nan")
                row = dict(backbone=bb, reducer=red, head=hd, width=(D if red == "none" else d), classes=C,
                           params_module=p_mod, params_total=bb_params + p_mod,
                           module_share_of_params_pct=100 * p_mod / (bb_params + p_mod),
                           flops_module=f_mod, flops_backbone=bb_flops, flops_total=bb_flops + f_mod,
                           module_share_of_flops_pct=100 * f_mod / (bb_flops + f_mod), peak_mem_mib_b64=peak)
                for k, (med, iqr) in lat.items():
                    row[f"lat_{k}_ms"] = med
                    row[f"lat_{k}_iqr"] = iqr
                pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"   {bb:8s} {red:9s} {hd:6s} params {p_mod:>8,} ({row['module_share_of_params_pct']:.3f}%)  "
                      f"FLOPs {f_mod / 1e3:8.1f}k ({row['module_share_of_flops_pct']:.4f}%)  "
                      f"module GPU b1 {lat['module_gpu_b1'][0]:.3f} ms  CPU {lat['module_cpu1t_b1'][0]:.3f} ms  "
                      f"full b64 {lat['full_gpu_b64'][0]:.1f} ms  peak {peak:.0f} MiB")
        del backbone
        if gpu.type == "cuda":
            torch.cuda.empty_cache()

    print(">> wrote " + out)


if __name__ == "__main__":
    main()
