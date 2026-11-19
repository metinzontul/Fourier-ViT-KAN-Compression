# -*- coding: utf-8 -*-
"""
Figure: test macro F1 vs. bottleneck width for every reducer (R2.5, R3.2).
Linear head (the simplest head, so the curves show the reducer and not head capacity);
the uncompressed linear probe is the dashed reference line. Error bars: sd over 5 seeds.

Inputs : results/tables/frozen_all_configs.csv (from analyze.py)
Outputs: results/figures/width_sweep.png (600 dpi) and .pdf, 13.9 cm wide

    python plot_width_sweep.py [--head linear]
"""
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
# validated categorical order; markers give a second, colour-independent encoding
RED = [("linear", "Learned linear", "#2a78d6", "o"),
       ("blockcirc", "Block-circulant", "#eb6834", "s"),
       ("pca", "PCA", "#1baf7a", "^"),
       ("random", "Random projection", "#eda100", "D"),
       ("bct", "Spectral BCT", "#e87ba4", "v")]
PANELS = [("ISIC2019", "vit_b_16"), ("ISIC2019", "resnet50"), ("CIFAR10", "vit_b_16"), ("CIFAR10", "resnet50")]
TITLE = {"ISIC2019": "ISIC 2019", "CIFAR10": "CIFAR-10", "vit_b_16": "ViT-B/16", "resnet50": "ResNet50"}
WIDTHS = [32, 64, 128, 256, 384]


def panel_label(ax, k):
    """MDPI panel label (a), (b), ... at the top-left corner of an axes."""
    ax.text(0.0, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, fontweight="bold", color="#1f1f1e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="linear")
    args = ap.parse_args()
    a = pd.read_csv(os.path.join(common.RESULT_DIR, "tables", "frozen_all_configs.csv"))
    a = a[a["head"] == args.head]
    missing = sorted(set(WIDTHS) - set(a[a.reducer != "none"].width))
    if missing:
        print(f"!! WARNING: widths {missing} are missing from tables/frozen_all_configs.csv - "
              f"run 'python analyze.py' first so the tables include the latest results.")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(13.9 / 2.54, 9.6 / 2.54), sharex=True)
    for k, (ax, (ds, bb)) in enumerate(zip(axes.flat, PANELS)):
        panel_label(ax, k)
        g = a[(a.dataset == ds) & (a.backbone == bb)]
        ref = g[g.reducer == "none"].test_mean.iloc[0]
        ax.axhline(ref, color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1, label="Uncompressed")
        for key, label, color, marker in RED:
            s = g[(g.reducer == key) & g.width.isin(WIDTHS)].sort_values("width")
            ax.errorbar(s.width, s.test_mean, yerr=s.test_sd, color=color, marker=marker, ms=4, lw=1.4,
                        elinewidth=0.7, capsize=1.5, markeredgecolor="white", markeredgewidth=0.5,
                        label=label, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
        ax.minorticks_off()
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{TITLE[ds]}, {TITLE[bb]}", fontsize=7.5, color=INK, pad=3)
        ax.set_xlim(WIDTHS[0] / 1.15, WIDTHS[-1] * 1.15)   # same margins on every matplotlib version
    for ax in axes[:, 0]:
        ax.set_ylabel("Test macro F1 (%)", fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("Bottleneck width d", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]                                  # reducers first, reference last
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=3, frameon=False,
               fontsize=6.5, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.0, h_pad=0.8)
    out = os.path.join(common.RESULT_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"width_sweep_{args.head}.{ext}"), dpi=600)
    print(">> wrote " + os.path.join(out, f"width_sweep_{args.head}.png") + " and .pdf (13.9 cm wide)")


if __name__ == "__main__":
    main()
