# -*- coding: utf-8 -*-
"""
Figures for the ISIC 2019 fine-tuning runs (ViT-B/16, lesion-grouped partitions, five seeds).

 (a) isic_curves:    training loss, validation loss, validation macro F1 and validation grid coverage
                     per epoch; line = mean over seeds, band = +/- 1 sd.
 (b) isic_confusion: row-normalised test confusion matrices (recall, %) at the validation-selected
                     epoch, counts summed over the five seeds, for ViT+KAN, ViT+BCT+KAN and ViT+BCT+LN+KAN.

Inputs : results/isic_finetune/epochs.csv, results/isic_finetune/cm_<config>_seed<seed>.npy
Outputs: results/figures/isic_curves.{png,pdf}, results/figures/isic_confusion.{png,pdf}

    python plot_isic_finetune.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import common

INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]  # alphabetical, as in common.py
# validated categorical order; KAN family solid, linear family dashed (colour-independent encoding)
CONFIGS = [("vit_kan", "ViT+KAN", "#2a78d6", "-"),
           ("vit_bct_kan", "ViT+BCT+KAN", "#eb6834", "-"),
           ("vit_bct_ln_kan", "ViT+BCT+LN+KAN", "#1baf7a", "-"),
           ("vit_linear", "ViT+Linear", "#eda100", (0, (4, 1.6))),
           ("vit_bct_linear", "ViT+BCT+Linear", "#e87ba4", (0, (4, 1.6)))]
CM_CONFIGS = [("vit_kan", "ViT+KAN"), ("vit_bct_kan", "ViT+BCT+KAN"), ("vit_bct_ln_kan", "ViT+BCT+LN+KAN")]


def panel_label(ax, k):
    """MDPI panel label (a), (b), ... at the top-left corner of an axes."""
    ax.text(0.0, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, fontweight="bold", color="#1f1f1e")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6})


def curves(ep, out):
    panels = [("train_loss", "Training loss"), ("val_loss", "Validation loss"),
              ("val_f1", "Validation macro F1 (%)"), ("val_head_in_grid", "Validation grid coverage")]
    fig, axes = plt.subplots(2, 2, figsize=(13.9 / 2.54, 9.6 / 2.54), sharex=True)
    for k, (ax, (col, label)) in enumerate(zip(axes.flat, panels)):
        panel_label(ax, k)
        for key, name, color, ls in CONFIGS:
            g = ep[ep.config == key].groupby("epoch")[col].agg(["mean", "std"]).reset_index()
            ax.fill_between(g.epoch, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15, lw=0)
            ax.plot(g.epoch, g["mean"], color=color, ls=ls, lw=1.4, label=name)
        ax.set_ylabel(label, fontsize=7)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(1, ep.epoch.max())
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[1, 1].set_ylim(0, 1.05)
    for ax in axes[1, :]:
        ax.set_xlabel("Epoch", fontsize=7)
    handles = [Line2D([], [], color=c, ls=ls, lw=1.4) for _, _, c, ls in CONFIGS]
    fig.legend(handles, [n for _, n, _, _ in CONFIGS], loc="lower center", ncol=5, frameon=False,
               fontsize=6.5, handlelength=2.4, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2, h_pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"isic_curves.{ext}"), dpi=600)
    plt.close(fig)


def load_cms(cm_dir):
    cms = {}
    for path in sorted(glob.glob(os.path.join(cm_dir, "cm_*_seed*.npy"))):
        cfg = re.match(r"cm_(.+)_seed\d+\.npy", os.path.basename(path)).group(1)
        cms[cfg] = cms.get(cfg, 0) + np.load(path)
    return cms


def confusion(cms, out):
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cfe0f5", "#7aaee8", "#2a78d6", "#123f7a"])
    fig, axes = plt.subplots(1, 3, figsize=(17.0 / 2.54, 6.3 / 2.54))
    for k, (ax, (key, name)) in enumerate(zip(axes, CM_CONFIGS)):
        panel_label(ax, k)
        cm = cms[key].astype(float)
        rec = 100 * cm / cm.sum(1, keepdims=True)
        im = ax.imshow(rec, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = rec[i, j]
                if v >= 0.5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                            color="white" if v >= 55 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_title(name, fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("Predicted class", fontsize=6.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True class", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 0.9, 1), w_pad=0.8)
    cax = fig.add_axes([0.91, 0.27, 0.012, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Recall (%)", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"isic_confusion.{ext}"), dpi=600)
    plt.close(fig)


def main():
    style()
    src = os.path.join(common.RESULT_DIR, "isic_finetune")
    out = os.path.join(common.RESULT_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    curves(pd.read_csv(os.path.join(src, "epochs.csv")), out)
    confusion(load_cms(src), out)
    print(">> wrote isic_curves.{png,pdf} and isic_confusion.{png,pdf} to " + out)


if __name__ == "__main__":
    main()
