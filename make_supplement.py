"""Build the Supplementary Materials (Tables S1-S11) as a LaTeX document from the final results.

Every number in the supplement is read from the result files written by the other scripts:
    <RESULT_DIR>/tables/*.csv            (analyze.py, make_efficiency_table.py, isic_per_class.py)
    <RESULT_DIR>/dataset_descriptors.csv (dataset_descriptors.py)
    <RESULT_DIR>/isic_finetune/summary.csv
    splits/isic2019_lesion_grouped_split.csv

    python make_supplement.py                      # writes <RESULT_DIR>/supplement/supplementary.tex
    pdflatex supplementary.tex  (twice)            # inside that folder

It also prints the exploratory summary of BCT versus channel-permuted BCT quoted in the paper
(mean difference across cells with a 90% CI, and Welch tests per cell with Holm correction).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.environ.get("REV_RESULT_DIR", os.path.join(HERE, "results"))
TAB = os.path.join(RESULT_DIR, "tables")
OUT = os.path.join(RESULT_DIR, "supplement")
SPLIT = os.environ.get("REV_SPLIT_CSV", os.path.join(HERE, "splits", "isic2019_lesion_grouped_split.csv"))

DS = ["MNIST", "CIFAR10", "EuroSAT", "ISIC2019"]
DSN = {"MNIST": "MNIST", "CIFAR10": "CIFAR-10", "EuroSAT": "EuroSAT", "ISIC2019": "ISIC 2019"}
BB = {"vit_b_16": "ViT-B/16", "resnet50": "ResNet50"}
RED = {"none": "None", "linear": "Learned linear", "blockcirc": "Block-circulant", "pca": "PCA",
       "random": "Random projection", "bct": "BCT", "bct_perm": "BCT-perm"}
HEAD = {"linear": "Linear", "mlp": "MLP", "kan": "KAN", "kan_ln": "LN+KAN",
        "kan_orig": r"KAN$_{\mathrm{orig}}$", "mlp_orig": r"MLP$_{\mathrm{orig}}$"}
RORDER = ["none", "linear", "blockcirc", "pca", "random", "bct", "bct_perm"]
HORDER = ["linear", "mlp", "kan", "kan_ln"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def num(x, d=2, sign=False):
    if pd.isna(x):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def pval(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def small(x, nd=3):
    """Format a non-negative share; values that round to zero are shown as <0.001 (or the matching bound)."""
    if 0 < x < 0.5 * 10 ** (-nd):
        return "$<$0." + "0" * (nd - 1) + "1"
    return f"{x:.{nd}f}"


def iqr(x):
    """Latency IQR with one decimal; a value that rounds to 0.0 (IQR < 0.05 ms) is shown as <0.1."""
    s = f"{x:.1f}"
    return "$<$0.1" if s == "0.0" else s


def longtable(caption, label, cols, header, rows, size=r"\footnotesize", note=None):
    n = len(cols)
    out = [size, r"\setlength{\tabcolsep}{3pt}",
           r"\begin{longtable}{" + cols + "}",
           r"\caption*{\textbf{Table " + label.split(":")[1] + ".} " + caption + r"}\\",
           r"\toprule", header + r" \\", r"\midrule", r"\endfirsthead",
           r"\multicolumn{" + str(n) + r"}{l}{\textit{(continued)}}\\", r"\toprule", header + r" \\",
           r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    out += rows
    out += [r"\end{longtable}"]
    if note:
        out += [r"\vspace{-6pt}{\footnotesize " + note + r"\par}"]
    out += [r"\normalsize", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- S1
def s1():
    # Partition sizes of all four datasets are given in Table 1 of the main text.
    d = pd.read_csv(SPLIT)
    ct = pd.crosstab(d.label, d.split).reindex(columns=["train", "val", "test"])
    rows = [f"{c} & {r.train:,} & {r.val:,} & {r.test:,} & {r.sum():,}" + r" \\" for c, r in ct.iterrows()]
    rows.append(r"\midrule")
    rows.append(f"Total & {ct.train.sum():,} & {ct.val.sum():,} & {ct.test.sum():,} & {ct.values.sum():,}" + r" \\")
    nolesion = d.lesion_id.isna().sum()
    shared = (d.dropna(subset=["lesion_id"]).groupby("lesion_id").split.nunique() > 1).sum()
    note = (f"Images without a lesion identifier: {nolesion:,} (each treated as its own group). "
            f"Lesions shared between partitions: {shared}. AK, actinic keratosis; BCC, basal cell carcinoma; "
            "BKL, benign keratosis-like lesion; DF, dermatofibroma; MEL, melanoma; NV, melanocytic nevus; "
            "SCC, squamous cell carcinoma; VASC, vascular lesion.")
    t2 = longtable("Class distribution of the lesion-grouped ISIC 2019 partitions. Partition sizes of all four datasets are given in Table 1 of the main text.", "tab:S1",
                   "lrrrr", r"\textbf{Class} & \textbf{Train} & \textbf{Validation} & \textbf{Test} & \textbf{Total}", rows, note=note)
    return r"\subsection*{Table S1. ISIC 2019 class distribution}" + "\n" + t2


# ---------------------------------------------------------------- S2
def s2():
    d = pd.read_csv(os.path.join(RESULT_DIR, "dataset_descriptors.csv"))
    rows = []
    for bb in ["vit_b_16", "resnet50"]:
        for ds in DS:
            r = d[(d.dataset == ds) & (d.backbone == bb)].iloc[0]
            rows.append(f"{BB[bb]} & {DSN[ds]} & {r.participation_ratio:.1f} & {int(r.pca_dim_95)} & {r.pca128_var:.3f} & "
                        f"{r.imbalance_ratio:.1f} & {small(r.dc_energy)} & {r.perm_ac_observed:.4f} & {r.perm_ac_null_mean:.4f} & "
                        f"{num(r.perm_ac_z)} & {r.perm_ac_p:.3f}" + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    note = ("PR, participation ratio of the feature covariance; PC95, number of principal components for 95\\% of the variance; "
            "Var128, variance retained by 128 components; Imb., ratio of the largest to the smallest class; DC, share of the energy "
            "in the DC bin of the channel-axis FFT; Band, share of non-DC energy in the bins that influence the BCT output at "
            "$d_{\\mathrm{out}}=128$; Perm., mean of this share over 500 channel permutations (5000 vectors each); $z$ and $p$, "
            "two-sided permutation test.")
    return r"\subsection*{Table S2. Feature-space descriptors}" + "\n" + longtable(
        "Feature-space descriptors of the training features and permutation tests of the latent channel spectrum.", "tab:S2",
        "llrrrrrrrrr", r"\textbf{Backbone} & \textbf{Dataset} & \textbf{PR} & \textbf{PC95} & \textbf{Var128} & \textbf{Imb.} & \textbf{DC} & \textbf{Band} & \textbf{Perm.} & \textbf{$z$} & \textbf{$p$}",
        rows, note=note)


# ---------------------------------------------------------------- S3/S4
def s34(bb, idx):
    a = pd.read_csv(os.path.join(TAB, "frozen_all_configs.csv"))
    a = a[(a.backbone == bb) & ((a.width == 128) | (a.reducer == "none")) & a["head"].isin(HORDER)]
    rows = []
    for r in RORDER:
        for h in HORDER:
            cells = []
            for ds in DS:
                x = a[(a.reducer == r) & (a["head"] == h) & (a.dataset == ds)]
                if len(x) == 0:
                    cells.append("--")
                    continue
                x = x.iloc[0]
                cells.append(f"{x.test_mean:.2f} $\\pm$ {x.test_sd:.2f} ({x.lr:.0e})".replace("e-0", "e-"))
            p10 = a[(a.reducer == r) & (a["head"] == h) & (a.dataset == "CIFAR10")].params
            rows.append(f"{RED[r]} & {HEAD[h]} & {int(p10.iloc[0]):,} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    note = ("Test Macro F1 (\\%), mean $\\pm$ standard deviation over five seeds, with the selected learning rate in parentheses. "
            "Params, trainable module parameters for ten classes (nominal count; the BCT has 126 inert weights). "
            "Uncompressed features have $d=768$ (ViT-B/16) or $d=2048$ (ResNet50); all other rows use $d_{\\mathrm{out}}=128$.")
    return (rf"\subsection*{{Table S{idx}. Frozen-feature results, {BB[bb]}}}" + "\n" + longtable(
        f"Frozen-feature results for {BB[bb]} at $d_{{\\mathrm{{out}}}}=128$.", f"tab:S{idx}", "llrcccc",
        r"\textbf{Reducer} & \textbf{Head} & \textbf{Params} & \textbf{MNIST} & \textbf{CIFAR-10} & \textbf{EuroSAT} & \textbf{ISIC 2019}",
        rows, size=r"\scriptsize", note=note))


# ---------------------------------------------------------------- S5
def bct_perm_tests():
    s = pd.read_csv(os.path.join(TAB, "frozen_selected_runs.csv"))
    s = s[s.width == 128]
    out = []
    for (ds, bb, h), g in s[s.reducer.isin(["bct", "bct_perm"])].groupby(["dataset", "backbone", "head"]):
        x = g[g.reducer == "bct"].sort_values("seed").test_f1.values
        y = g[g.reducer == "bct_perm"].sort_values("seed").test_f1.values
        t = stats.ttest_ind(x, y, equal_var=False)
        out.append(dict(dataset=ds, backbone=bb, head=h, diff=x.mean() - y.mean(), p_welch=t.pvalue))
    o = pd.DataFrame(out)
    o["p_welch_holm"] = holm(o.p_welch)
    d = o["diff"].values
    ci = stats.t.interval(0.90, len(d) - 1, d.mean(), stats.sem(d))
    print(f"BCT - BCT-perm across {len(d)} cells: mean {d.mean():+.3f}, 90% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
          f"|diff|<0.5 in {(np.abs(d) < 0.5).sum()}, Welch-Holm significant in {(o.p_welch_holm < 0.05).sum()}: "
          + ", ".join(f"{r.dataset}/{r.backbone}/{r['head']} {r['diff']:+.2f}" for _, r in o[o.p_welch_holm < 0.05].iterrows()))
    return o


def s5():
    h = pd.read_csv(os.path.join(TAB, "hypothesis_tests_cells.csv"))
    welch = bct_perm_tests().set_index(["dataset", "backbone", "head"])
    blocks = []
    names = {"H1": ("H1: BCT minus random projection", "Head"), "H2": ("H2: BCT minus BCT-perm", "Head"),
             "H3": ("H3: interaction (KAN loss minus linear loss)", "Reducer"), "H4": ("H4: LN+KAN minus KAN", "Reducer")}
    for key in ["H1", "H2", "H3", "H4"]:
        g = h[h.hypothesis.str.startswith(key)]
        rows = []
        for _, r in g.iterrows():
            lab = HEAD.get(r["head"], RED.get(r["head"], r["head"]))
            if key == "H2":
                w = welch.loc[(r.dataset, r.backbone, r["head"])]
                rows.append(f"{BB[r.backbone]} & {DSN[r.dataset]} & {lab} & {num(r['diff'], sign=True)} & "
                            f"{pval(r.p)} & {pval(r.p_holm)} & {pval(w.p_welch)} & {pval(w.p_welch_holm)}" + r" \\")
            else:
                ci = f"[{num(r.ci_lo)}, {num(r.ci_hi)}]"
                rows.append(f"{BB[r.backbone]} & {DSN[r.dataset]} & {lab} & {num(r['diff'], sign=True)} & {ci} & "
                            f"{num(r.hedges_g)} & {pval(r.p)} & {pval(r.p_holm)}" + r" \\")
        title, col = names[key]
        if key == "H2":
            header = (rf"\textbf{{Backbone}} & \textbf{{Dataset}} & \textbf{{{col}}} & \textbf{{Diff.}} & \textbf{{TOST $p$}} & "
                      r"\textbf{TOST $p_{\mathrm{Holm}}$} & \textbf{Welch $p$} & \textbf{Welch $p_{\mathrm{Holm}}$}")
            note = ("Equivalence margin $\\pm0.5$ percentage points. The Welch columns test for a difference (exploratory).")
        else:
            header = (rf"\textbf{{Backbone}} & \textbf{{Dataset}} & \textbf{{{col}}} & \textbf{{Diff.}} & \textbf{{95\% CI}} & "
                      r"\textbf{$g$} & \textbf{$p$} & \textbf{$p_{\mathrm{Holm}}$}")
            note = "Differences in Macro F1 percentage points; $g$, Hedges' $g$ (not defined for the interaction)."
        blocks.append(longtable(title + ".", "tab:S5" + "abcd"[["H1","H2","H3","H4"].index(key)], "lllrcrrr", header, rows, size=r"\scriptsize", note=note))
    return "\n".join(blocks)


# ---------------------------------------------------------------- S6/S7
def s6():
    h = pd.read_csv(os.path.join(TAB, "frozen_head_tests.csv"))
    rows = [f"{BB[r.backbone]} & {DSN[r.dataset]} & {r.comparison.replace('mlp', 'MLP').replace('kan', 'KAN').replace('linear', 'Linear')} & "
            f"{num(r['diff'], sign=True)} & [{num(r.ci_lo)}, {num(r.ci_hi)}] & {num(r.hedges_g)} & {pval(r.p)} & {pval(r.p_holm)}" + r" \\"
            for _, r in h.sort_values(["backbone", "dataset"]).iterrows()]
    return r"\subsection*{Table S6. Head comparisons without reduction}" + "\n" + longtable(
        "Pairwise head comparisons on the uncompressed features (Welch tests, Holm correction over all rows).", "tab:S6",
        "lllrcrrr", r"\textbf{Backbone} & \textbf{Dataset} & \textbf{Comparison} & \textbf{Diff.} & \textbf{95\% CI} & \textbf{$g$} & \textbf{$p$} & \textbf{$p_{\mathrm{Holm}}$}",
        rows, size=r"\scriptsize")


def s7():
    a = pd.read_csv(os.path.join(TAB, "frozen_all_configs.csv"))
    a = a[(a.backbone == "vit_b_16") & (a.reducer == "none")]
    rows = []
    for h in ["kan_orig", "mlp_orig", "linear", "mlp", "kan"]:
        cells = []
        for ds in ["MNIST", "CIFAR10", "EuroSAT"]:
            x = a[(a["head"] == h) & (a.dataset == ds)]
            cells.append("--" if len(x) == 0 else f"{x.iloc[0].test_mean:.2f} $\\pm$ {x.iloc[0].test_sd:.2f}")
        p = a[(a["head"] == h) & (a.dataset == "CIFAR10")].params
        rows.append(f"{HEAD[h]} & {int(p.iloc[0]):,} & " + " & ".join(cells) + r" \\")
    note = ("KAN$_{\\mathrm{orig}}$: bias-free weighted sum over five feature copies (a linear map); MLP$_{\\mathrm{orig}}$: ReLU followed by "
            "a linear layer without hidden units. Control heads used a learning rate of $10^{-3}$ and a batch size of 64; the other rows "
            "repeat the tuned results of Table~S3.")
    return r"\subsection*{Table S7. Control heads}" + "\n" + longtable(
        "Control heads on uncompressed ViT-B/16 features (test Macro F1, \\%).", "tab:S7", "lrccc",
        r"\textbf{Head} & \textbf{Params} & \textbf{MNIST} & \textbf{CIFAR-10} & \textbf{EuroSAT}", rows, note=note)


# ---------------------------------------------------------------- S8
def s8():
    a = pd.read_csv(os.path.join(TAB, "frozen_all_configs.csv"))
    w = a[(a.reducer != "none") & (a.reducer != "bct_perm")]
    rows = []
    for ds in ["ISIC2019", "CIFAR10"]:
        for bb in ["vit_b_16", "resnet50"]:
            for r in ["linear", "blockcirc", "pca", "random", "bct"]:
                for h in HORDER:
                    x = w[(w.dataset == ds) & (w.backbone == bb) & (w.reducer == r) & (w["head"] == h)]
                    if len(x) == 0:
                        continue
                    x = x.set_index("width")
                    cells = [f"{num(x.loc[k, 'test_mean'])} $\\pm$ {x.loc[k, 'test_sd']:.2f}" if k in x.index else "--"
                             for k in [32, 64, 128, 256, 384]]
                    rows.append(f"{DSN[ds]} & {BB[bb]} & {RED[r]} & {HEAD[h]} & " + " & ".join(cells) + r" \\")
            rows.append(r"\midrule")
    rows = rows[:-1]
    return r"\subsection*{Table S8. Bottleneck-width sweep}" + "\n" + longtable(
        "Test Macro F1 (\\%, mean $\\pm$ standard deviation over five seeds) as a function of the bottleneck width $d_{\\mathrm{out}}$.", "tab:S8",
        "llllccccc", r"\textbf{Dataset} & \textbf{Backbone} & \textbf{Reducer} & \textbf{Head} & \textbf{32} & \textbf{64} & \textbf{128} & \textbf{256} & \textbf{384}",
        rows, size=r"\scriptsize")


# ---------------------------------------------------------------- S9/S10
CFG = [("vit_kan", "ViT+KAN"), ("vit_bct_kan", "ViT+BCT+KAN"), ("vit_bct_ln_kan", "ViT+BCT+LN+KAN"),
       ("vit_linear", "ViT+Linear"), ("vit_bct_linear", "ViT+BCT+Linear")]


def s9():
    s = pd.read_csv(os.path.join(RESULT_DIR, "isic_finetune", "summary.csv"))
    r5 = pd.read_csv(os.path.join(TAB, "isic_runs.csv")).set_index(["config", "seed"])
    rows = []
    for k, name in CFG:
        for _, r in s[s.config == k].sort_values("seed").iterrows():
            l5 = r5.loc[(k, r.seed), "last5"]
            rows.append(f"{name} & {r.seed} & {int(r.best_epoch)} & {r.val_f1:.2f} & {r.test_acc:.2f} & {r.test_f1:.2f} & "
                        f"{r.last_test_f1:.2f} & {l5:.2f} & {r.test_head_in_grid:.2f}" + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    t = pd.read_csv(os.path.join(TAB, "isic_tests.csv"))
    note = ("Interaction (BCT loss of KAN minus BCT loss of linear head, Welch--Satterthwaite): "
            + "; ".join(f"{r.metric}: {r.interaction:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}], $p={r.p_interaction:.3f}$"
                        for _, r in t[t.metric.isin(['selected', 'last', 'last5'])].iterrows()) + ".")
    return r"\subsection*{Table S9. ISIC 2019 fine-tuning, seed-level results}" + "\n" + longtable(
        "Seed-level ISIC 2019 fine-tuning results (ViT-B/16, lesion-grouped partitions).", "tab:S9", "lrrrrrrrr",
        r"\textbf{Configuration} & \textbf{Seed} & \textbf{Epoch} & \textbf{Val F1} & \textbf{Test Acc.} & \textbf{Test F1} & \textbf{Last} & \textbf{Last 5} & \textbf{Coverage}",
        rows, size=r"\scriptsize", note=note)


def s10():
    p = pd.read_csv(os.path.join(TAB, "isic_per_class.csv"))
    d = pd.read_csv(os.path.join(TAB, "isic_per_class_drops.csv")).set_index("cls")
    rows = []
    for c in ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]:
        cells = []
        for k, _ in CFG:
            x = p[(p.cls == c) & (p.config == k)].iloc[0]
            cells.append(f"{x.f1_mean:.1f} $\\pm$ {x.f1_sd:.1f}")
        r = d.loc[c]
        rows.append(f"{c} & {int(r.support)} & " + " & ".join(cells) +
                    f" & {num(r.bct_drop_kan, 1, True)} & {num(r.bct_drop_linear, 1, True)} & {num(r.ln_gain_over_bct_kan, 1, True)}" + r" \\")
    note = ("Per-class F1 (\\%) from the test confusion matrices, mean $\\pm$ standard deviation over five seeds. "
            "$\\Delta_{\\mathrm{KAN}}$ and $\\Delta_{\\mathrm{Lin}}$: F1 loss caused by the BCT (positive values are losses); "
            "LN gain: ViT+BCT+LN+KAN minus ViT+BCT+KAN. Exploratory analysis without correction for multiple comparisons.")
    return r"\subsection*{Table S10. ISIC 2019 per-class F1}" + "\n" + longtable(
        "Per-class F1 in ISIC 2019 fine-tuning.", "tab:S10", "lrcccccrrr",
        r"\textbf{Class} & \textbf{$n$} & \textbf{KAN} & \textbf{BCT+KAN} & \textbf{BCT+LN+KAN} & \textbf{Linear} & \textbf{BCT+Lin.} & \textbf{$\Delta_{\mathrm{KAN}}$} & \textbf{$\Delta_{\mathrm{Lin}}$} & \textbf{LN gain}",
        rows, size=r"\scriptsize", note=note)


# ---------------------------------------------------------------- S11
def s11():
    e = pd.read_csv(os.path.join(TAB, "efficiency_full.csv"))
    rows = []
    prev = None
    for _, r in e.iterrows():
        if prev is not None and r.Backbone != prev:
            rows.append(r"\midrule")
        prev = r.Backbone
        red = "None" if r["Reducer (d=128)"] == "-" else r["Reducer (d=128)"].replace("Spectral BCT", "BCT").replace("Random projection", "Random").replace("Block-circulant", "Block-circ.").replace("Learned linear", "Learned lin.")
        fl = r["Module share of FLOPs (%)"]
        rows.append(f"{r.Backbone} & {red} & {r.Head} & {int(r['Module params']):,} & "
                    f"{r['Module share of params (%)']:.3f} & {r['Module FLOPs (k)']:.1f} & {'$<$0.0001' if fl < 0.0001 else f'{fl:.4f}'} & "
                    f"{r['Module CPU-1T b1 (ms)']:.3f} & {r['Model GPU b64 (ms)']:.1f} ({iqr(r['Model GPU b64 IQR'])}) & "
                    f"{r['Peak memory b64 (MiB)']:.0f}" + r" \\")
    note = ("CPU: module alone, one thread, batch 1; Model: whole model on the GPU, batch 64. Block-circ., block-circulant; Learned lin., learned linear. Shares are relative to the whole model (ViT-B/16: 85.80 M parameters, 35.13 GFLOPs; ResNet50: 23.51 M, 8.17 GFLOPs); fixed PCA and random projection matrices are not trainable and are not counted as "
            "parameters, but their operations are counted as FLOPs. Latency: median of 100 iterations after 20 warm-up iterations, fp32; "
            "whole-model interquartile range in parentheses. Hardware: NVIDIA GeForce RTX 4080, Intel Core i9, PyTorch 2.5.1, CUDA 12.1.")
    return r"\subsection*{Table S11. Efficiency measurements}" + "\n" + longtable(
        "Complete efficiency measurements ($d_{\\mathrm{out}}=128$; ten classes).", "tab:S11", "lllrrrrrrr",
        r"\textbf{Backbone} & \textbf{Reducer} & \textbf{Head} & \textbf{Params} & \textbf{Share (\%)} & \textbf{kFLOPs} & \textbf{FLOP sh. (\%)} & \textbf{CPU (ms)} & \textbf{Model (ms)} & \textbf{Mem. (MiB)}",
        rows, size=r"\scriptsize", note=note)


PRE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,amssymb}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\renewcommand{\thetable}{S\arabic{table}}
\setlength{\LTcapwidth}{\textwidth}
\begin{document}
\begin{center}
{\large\bfseries Supplementary Materials}\\[4pt]
Performance--Compression Trade-offs in Vision Classifier Heads: A Controlled Comparison of KAN, MLP, and Linear Configurations\\[2pt]
Metin Zontul\\[2pt]
\textit{Applied Sciences} (MDPI)
\end{center}

\section*{Contents}
Tables S1--S11 report the class distribution of the ISIC 2019 partitions, the descriptors of the feature spaces, the complete frozen-feature results, the per-cell
hypothesis tests, the head comparisons, the control heads, the bottleneck-width sweep, the ISIC 2019 fine-tuning results at the
seed and class level, and the complete efficiency measurements referred to in the main text. All tables were generated by
\texttt{make\_supplement.py} from the result files in the repository; Macro F1 values are test-partition results at the epoch
selected on the validation partition. The sub-tables of Table S5 are numbered S5a--S5d.

\section*{Data and Code Availability}
The repository \url{https://github.com/metinzontul/Fourier-ViT-KAN-Compression} (release v2.0) contains the source code of all
experiments, the partition files (including the lesion-grouped ISIC 2019 partition), the training logs, the result tables, the
scripts that produce the figures and the tables of the article and of this supplement, and the software environment
specification. A \texttt{README} file maps every table and figure to the command that produces it. The scripts and results of
the previous version of the study are kept in a separate folder of the repository with notes on their status.
\clearpage
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [PRE, s1(), s2(), s34("vit_b_16", 3), s34("resnet50", 4), s5(), s6(), s7(), s8(), s9(), s10(), s11(),
             r"\end{document}"]
    tex = "\n".join(parts)
    # longtable numbering: one counter per table; sub-tables keep the S-number of their section
    path = os.path.join(OUT, "supplementary.tex")
    open(path, "w", encoding="utf-8").write(tex)
    print(">> wrote", path)


if __name__ == "__main__":
    main()
