# -*- coding: utf-8 -*-
"""
Figure: cumulative feature-axis (latent) power spectra, original vs. randomly permuted channel order.
Replaces the image-domain PSD figure of the original submission (R1.3, R2.3, R3.6).

Inputs : results/latent_spectra.csv, results/dataset_descriptors.csv (from dataset_descriptors.py)
Outputs: results/figures/latent_spectrum.png (600 dpi) and .pdf

    python plot_latent_spectrum.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

ORIG, PERM = "#2a78d6", "#eb6834"          # validated categorical pair (CVD dE 24.7)
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
DS = ["MNIST", "CIFAR10", "EuroSAT", "ISIC2019"]
DS_LABEL = {"MNIST": "MNIST", "CIFAR10": "CIFAR-10", "EuroSAT": "EuroSAT", "ISIC2019": "ISIC 2019"}
BB_LABEL = {"vit_b_16": "ViT-B/16", "resnet50": "ResNet50"}
WIDTH = 128                                  # BCT output width; irfft(n=128) reads bins 0..64


def panel_label(ax, k):
    """MDPI panel label (a), (b), ... at the top-left corner of an axes."""
    ax.text(0.0, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, fontweight="bold", color="#1f1f1e")


def main():
    sp = pd.read_csv(os.path.join(common.RESULT_DIR, "latent_spectra.csv"))
    de = pd.read_csv(os.path.join(common.RESULT_DIR, "dataset_descriptors.csv")).set_index(["dataset", "backbone"])
    # Sized for the MDPI text width (13.9 cm): fonts below are the printed sizes (>= 6.5 pt)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axes = plt.subplots(2, 4, figsize=(13.9 / 2.54, 8.6 / 2.54), sharex=True, sharey=True)

    for i, bb in enumerate(["vit_b_16", "resnet50"]):
        for j, ds in enumerate(DS):
            ax = axes[i, j]
            s = sp[(sp.dataset == ds) & (sp.backbone == bb)].sort_values("bin")
            s = s[s.bin > 0]                       # DC bin excluded: it does not depend on channel order
            D = int(de.loc[(ds, bb), "feature_dim"])
            band = (WIDTH // 2) / D                # highest frequency read by irfft(n=128)
            c, cp = np.cumsum(s.power) / s.power.sum(), np.cumsum(s.power_perm) / s.power_perm.sum()
            ax.axvspan(0, band, color=GRID, lw=0, zorder=0)
            ax.plot([0, 0.5], [0, 1], color=MUTED, lw=0.8, ls=":", zorder=1)
            ax.plot(s.freq, cp, color=PERM, lw=1.8, ls=(0, (3, 1.5)), zorder=2)
            ax.plot(s.freq, c, color=ORIG, lw=1.0, zorder=3)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.25, 0.5], ["0", ".25", ".5"])
            ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
            ax.spines[["top", "right"]].set_visible(False)
            r = de.loc[(ds, bb)]
            txt = f"z = {r.perm_ac_z:+.2f}\np = {r.perm_ac_p:.2f}"
            if bb == "resnet50":
                txt += f"\nDC {r.dc_energy:.2f}"
            ax.text(0.05, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK,
                    linespacing=1.2)
            panel_label(ax, 4 * i + j)
            if i == 0:
                ax.set_title(DS_LABEL[ds], fontsize=7.5, color=INK, pad=3)
            if j == 0:
                ax.set_ylabel(f"{BB_LABEL[bb]}\ncumulative energy", fontsize=7)

    fig.supxlabel("Feature-axis frequency (cycles per channel), DC bin excluded", fontsize=7, y=0.115)
    handles = [plt.Line2D([], [], color=ORIG, lw=1.0), plt.Line2D([], [], color=PERM, lw=1.8, ls=(0, (3, 1.5))),
               plt.Line2D([], [], color=MUTED, lw=0.8, ls=":"), plt.Rectangle((0, 0), 1, 1, color=GRID)]
    fig.legend(handles, ["Original channel order", "Permuted channel order", "White spectrum",
                         "Band kept by spectral BCT (d = 128)"],
               loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.5, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.6, h_pad=0.5)
    out = os.path.join(common.RESULT_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"latent_spectrum.{ext}"), dpi=600)
    print(">> wrote " + os.path.join(out, "latent_spectrum.png") + " and .pdf (13.9 cm wide)")


if __name__ == "__main__":
    main()
