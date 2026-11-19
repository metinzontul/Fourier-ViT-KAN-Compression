# -*- coding: utf-8 -*-
"""
Step 5 - Build every table and statistic of the revision from the result CSVs.

Inputs (whatever exists is used):
    results/frozen_results.csv            controlled frozen-feature study
    results/isic_finetune/epochs.csv      ISIC 2019 fine-tuning, per epoch
    results/isic_finetune/summary.csv     ISIC 2019 fine-tuning, per run
    results/dataset_descriptors.csv       feature-space descriptors
Outputs: results/tables/*.csv and results/tables/report.md

Learning-rate selection (validation data only):
    seed-0 sweep over the LR grid; if more than one LR was later run with all 5 seeds
    (this happens when the grid was extended), the LR with the higher MEAN validation F1
    over the 5 seeds is kept.

    python analyze.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import common

OUT = os.path.join(common.RESULT_DIR, "tables")
K = ["dataset", "backbone", "reducer", "width", "head"]
DS = ["MNIST", "CIFAR10", "EuroSAT", "ISIC2019"]


# ----------------------------------------------------------------------------- helpers
def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else np.nan


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    p = 2 * stats.t.sf(abs(d / se), df)
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h, p


def interaction(kb, kp, lb, lp):
    """(mean kb - mean kp) - (mean lb - mean lp), Welch-Satterthwaite CI and p."""
    groups = [np.asarray(x, float) for x in (kb, kp, lb, lp)]
    d = (groups[0].mean() - groups[1].mean()) - (groups[2].mean() - groups[3].mean())
    v = [g.var(ddof=1) / len(g) for g in groups]
    se = np.sqrt(sum(v))
    df = sum(v) ** 2 / sum(x ** 2 / (len(g) - 1) for x, g in zip(v, groups))
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h, 2 * stats.t.sf(abs(d / se), df)


def tost(a, b, margin):
    """Welch two one-sided tests for equivalence of means within +/- margin. Returns (diff, p_tost)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    p_lower = stats.t.sf((d + margin) / se, df)      # H0: d <= -margin
    p_upper = stats.t.cdf((d - margin) / se, df)     # H0: d >= +margin
    return d, max(p_lower, p_upper)


def sign_test(diffs):
    """Two-sided binomial sign test on the direction of per-cell differences."""
    diffs = np.asarray(diffs, float)
    diffs = diffs[diffs != 0]
    k = int((diffs > 0).sum())
    return k, len(diffs), stats.binomtest(k, len(diffs), 0.5).pvalue if len(diffs) else np.nan


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def md(df, index=True):
    """Minimal markdown table (no 'tabulate' dependency)."""
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                     (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in row) + " |")
    return "\n".join(out)


def fmt(m, s):
    return f"{m:.2f} ± {s:.2f}"


# ----------------------------------------------------------------------------- frozen study
def select_final(d):
    """Final-stage rows at the selected learning rate (validation data only)."""
    tune_best = (d[d.stage == "tune"].sort_values("val_f1", ascending=False)
                 .drop_duplicates(K).set_index(K).lr)
    rows = []
    for key, g in d[d.stage == "final"].groupby(K):
        full = g.groupby("lr").filter(lambda x: x.seed.nunique() >= 5)
        cands = sorted(full.lr.unique())
        if len(cands) > 1:
            lr = full.groupby("lr").val_f1.mean().idxmax()
        elif len(cands) == 1:
            lr = cands[0]
        else:
            lr = tune_best.get(key, g.lr.iloc[0])
        rows.append(g[g.lr == lr].drop_duplicates("seed", keep="last"))
    return pd.concat(rows)


def frozen_tables(path, lines):
    d = pd.read_csv(path)
    f = select_final(d)
    agg = f.groupby(K).agg(test_mean=("test_f1", "mean"), test_sd=("test_f1", "std"), n=("seed", "nunique"),
                           val_mean=("val_f1", "mean"), lr=("lr", "first"), params=("params_module", "first"),
                           params_eff=("params_module_effective", "first"),
                           grid=("head_input_in_grid", "mean")).reset_index()
    agg.to_csv(os.path.join(OUT, "frozen_all_configs.csv"), index=False)
    f.to_csv(os.path.join(OUT, "frozen_selected_runs.csv"), index=False)
    lines.append(f"## Frozen-feature study\n\n{len(agg)} configurations, "
                 f"{int((agg.n < 5).sum())} with fewer than 5 seeds.\n")

    main = agg[(agg.width == 128) | (agg.reducer == "none")]
    main = main[main["head"].isin(["linear", "mlp", "kan", "kan_ln"])]
    for bb in ["vit_b_16", "resnet50"]:
        m = main[main.backbone == bb].copy()
        m["cell"] = [fmt(a, b) for a, b in zip(m.test_mean, m.test_sd)]
        p = m.pivot_table(index=["reducer", "head"], columns="dataset", values="cell", aggfunc="first")
        p = p[[c for c in DS if c in p.columns]]
        p.to_csv(os.path.join(OUT, f"frozen_main_{bb}.csv"))
        lines.append(f"### Test macro F1, d = 128, {bb} (mean ± sd over 5 seeds)\n\n" + md(p) + "\n")

    # compression penalty relative to no reduction, same head
    base = main[main.reducer == "none"].set_index(["dataset", "backbone", "head"]).test_mean
    pen = main[main.reducer != "none"].copy()
    pen["penalty"] = pen.test_mean.values - base.reindex(
        pd.MultiIndex.from_frame(pen[["dataset", "backbone", "head"]])).values
    pt = pen.pivot_table(index=["backbone", "reducer"], columns="dataset", values="penalty", aggfunc="mean").round(2)
    pt = pt[[c for c in DS if c in pt.columns]]
    pt.to_csv(os.path.join(OUT, "frozen_penalty.csv"))
    lines.append("### Compression penalty vs. no reduction (test F1 points, mean over the four heads)\n\n"
                 + md(pt) + "\n")

    # head comparison without reduction, Welch + Holm
    rows = []
    for (ds, bb), g in f[(f.reducer == "none") & f["head"].isin(["linear", "mlp", "kan"])].groupby(["dataset", "backbone"]):
        for a, b in [("mlp", "kan"), ("linear", "kan"), ("mlp", "linear")]:
            x, y = g[g["head"] == a].test_f1, g[g["head"] == b].test_f1
            if len(x) >= 2 and len(y) >= 2:
                dlt, lo, hi, p = welch(x, y)
                rows.append(dict(dataset=ds, backbone=bb, comparison=f"{a} - {b}", diff=dlt, ci_lo=lo, ci_hi=hi,
                                 p=p, hedges_g=hedges_g(x, y)))
    if rows:
        h = pd.DataFrame(rows)
        h["p_holm"] = holm(h.p)
        h.round(3).to_csv(os.path.join(OUT, "frozen_head_tests.csv"), index=False)
        lines.append("### Head comparisons without reduction (Welch, Holm-adjusted over all rows)\n\n"
                     + md(h.round(3), index=False) + "\n")

    # original (published) heads vs real heads
    ctl = d[d["head"].isin(["kan_orig", "mlp_orig"]) & (d.stage == "final")]
    if len(ctl):
        c = ctl.groupby(["dataset", "head"]).test_f1.agg(["mean", "std"]).round(2)
        lines.append("### Control: heads exactly as published (ViT, no reduction)\n\n" + md(c) + "\n")

    # width sweep
    w = agg[(agg.reducer != "none")]
    if w.width.nunique() > 1:
        wt = w.pivot_table(index=["dataset", "backbone", "reducer", "head"], columns="width",
                           values="test_mean", aggfunc="first").round(2)
        wt.to_csv(os.path.join(OUT, "frozen_widths.csv"))
        lines.append("### Bottleneck width sweep (test F1)\n\n" + md(wt) + "\n")
    return agg


# ----------------------------------------------------------------------------- hypothesis tests
EQ_MARGIN = 0.5   # equivalence margin (test-F1 points) for H2b, fixed before the bct_perm runs


def hypothesis_tests(f, lines):
    f = f[(f.width == 128) | (f.reducer == "none")]
    cells = f.groupby(["dataset", "backbone", "reducer", "head"]).test_f1.apply(np.asarray)
    rows, summary = [], []

    def get(ds, bb, red, hd):
        return cells.get((ds, bb, red, hd))

    pairs = [(ds, bb) for ds in DS for bb in ["vit_b_16", "resnet50"]]
    heads4 = ["linear", "mlp", "kan", "kan_ln"]

    # H1: spectral BCT vs random projection (positive diff = BCT better)
    fam = []
    for ds, bb in pairs:
        for hd in heads4:
            a, b = get(ds, bb, "bct", hd), get(ds, bb, "random", hd)
            if a is not None and b is not None:
                d, lo, hi, p = welch(a, b)
                fam.append(dict(hypothesis="H1 BCT - random", dataset=ds, backbone=bb, head=hd, diff=d,
                                ci_lo=lo, ci_hi=hi, p=p, hedges_g=hedges_g(a, b)))
    if fam:
        for r, ph in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = ph
        k, n, p = sign_test([r["diff"] for r in fam])
        summary.append(dict(hypothesis="H1: spectral BCT is not better than random projection (d=128)",
                            cells=n, cells_favouring=f"BCT better in {k}/{n}", sign_test_p=p,
                            sig_after_holm=f"{sum(r['p_holm'] < .05 and r['diff'] < 0 for r in fam)} worse, "
                                           f"{sum(r['p_holm'] < .05 and r['diff'] > 0 for r in fam)} better"))
        rows += fam

    # H2b: channel order does not matter (equivalence, margin EQ_MARGIN)
    fam = []
    for ds, bb in pairs:
        for hd in heads4:
            a, b = get(ds, bb, "bct", hd), get(ds, bb, "bct_perm", hd)
            if a is not None and b is not None:
                d, p = tost(a, b, EQ_MARGIN)
                fam.append(dict(hypothesis=f"H2b BCT - BCT(permuted), TOST +/-{EQ_MARGIN}", dataset=ds, backbone=bb,
                                head=hd, diff=d, p=p))
    if fam:
        for r, ph in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = ph
        summary.append(dict(hypothesis=f"H2b: BCT is equivalent with permuted channel order (margin {EQ_MARGIN})",
                            cells=len(fam), cells_favouring=f"|diff| < {EQ_MARGIN} in "
                            f"{sum(abs(r['diff']) < EQ_MARGIN for r in fam)}/{len(fam)}",
                            sign_test_p=np.nan, sig_after_holm=f"equivalent in {sum(r['p_holm'] < .05 for r in fam)}"))
        rows += fam

    # H3: KAN loses more than Linear under each lossy reducer (positive = KAN loses more)
    fam = []
    for ds, bb in pairs:
        for red in ["bct", "blockcirc", "pca", "random"]:
            g = [get(ds, bb, r, h) for r, h in [("none", "kan"), (red, "kan"), ("none", "linear"), (red, "linear")]]
            if all(x is not None for x in g):
                d, lo, hi, p = interaction(*g)
                fam.append(dict(hypothesis="H3 interaction (KAN drop - Linear drop)", dataset=ds, backbone=bb,
                                head=red, diff=d, ci_lo=lo, ci_hi=hi, p=p))
    if fam:
        for r, ph in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = ph
        k, n, p = sign_test([r["diff"] for r in fam])
        summary.append(dict(hypothesis="H3: KAN loses more than Linear under lossy reduction (frozen)",
                            cells=n, cells_favouring=f"KAN loses more in {k}/{n}", sign_test_p=p,
                            sig_after_holm=f"{sum(r['p_holm'] < .05 and r['diff'] > 0 for r in fam)} KAN-worse, "
                                           f"{sum(r['p_holm'] < .05 and r['diff'] < 0 for r in fam)} KAN-better"))
        rows += fam

    # H4: LayerNorm before the KAN helps under lossy reduction (positive = KAN+LN better)
    fam = []
    for ds, bb in pairs:
        for red in ["bct", "blockcirc", "pca", "random"]:
            a, b = get(ds, bb, red, "kan_ln"), get(ds, bb, red, "kan")
            if a is not None and b is not None:
                d, lo, hi, p = welch(a, b)
                fam.append(dict(hypothesis="H4 KAN+LN - KAN", dataset=ds, backbone=bb, head=red, diff=d,
                                ci_lo=lo, ci_hi=hi, p=p, hedges_g=hedges_g(a, b)))
    if fam:
        for r, ph in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = ph
        for red in ["bct", "blockcirc", "pca", "random"]:
            sub = [r for r in fam if r["head"] == red]
            k, n, p = sign_test([r["diff"] for r in sub])
            summary.append(dict(hypothesis=f"H4: LayerNorm helps the KAN under {red}", cells=n,
                                cells_favouring=f"KAN+LN better in {k}/{n}", sign_test_p=p,
                                sig_after_holm=f"{sum(r['p_holm'] < .05 and r['diff'] > 0 for r in sub)} better, "
                                               f"{sum(r['p_holm'] < .05 and r['diff'] < 0 for r in sub)} worse"))
        rows += fam

    if rows:
        pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "hypothesis_tests_cells.csv"), index=False)
        sm = pd.DataFrame(summary)
        sm.to_csv(os.path.join(OUT, "hypothesis_tests_summary.csv"), index=False)
        lines.append("## Hypothesis tests (frozen study, d = 128; Holm within each family; "
                     "per-cell results in hypothesis_tests_cells.csv)\n\n" + md(sm.round(4), index=False) + "\n")


# ----------------------------------------------------------------------------- ISIC fine-tuning
def isic_tables(ep_path, lines):
    e = pd.read_csv(ep_path)

    def per_run(x):
        b = x.loc[x.val_f1.idxmax()]
        return pd.Series(dict(selected=b.test_f1, best_epoch=b.epoch, last=x[x.epoch == x.epoch.max()].test_f1.iloc[0],
                              last5=x[x.epoch > x.epoch.max() - 5].test_f1.mean(),
                              grid=x.val_head_in_grid.mean(), head_in_std=x.val_head_in_std.mean()))

    r = pd.DataFrame([dict(config=c, seed=sd, **per_run(x)) for (c, sd), x in e.groupby(["config", "seed"])])
    r.to_csv(os.path.join(OUT, "isic_runs.csv"), index=False)
    s = r.groupby("config").agg(n=("seed", "nunique"), selected=("selected", "mean"), selected_sd=("selected", "std"),
                                last=("last", "mean"), last5=("last5", "mean"), grid=("grid", "mean"),
                                head_in_std=("head_in_std", "mean")).round(2)
    s.to_csv(os.path.join(OUT, "isic_summary.csv"))
    lines.append("## ISIC 2019 fine-tuning (lesion-grouped split, test macro F1)\n\n" + md(s) + "\n")

    need = {"vit_kan", "vit_bct_kan", "vit_linear", "vit_bct_linear"}
    if need <= set(r.config):
        g = {c: r[r.config == c] for c in r.config.unique()}
        rows = []
        for metric in ["selected", "last", "last5"]:
            kb, kp, lb, lp = (g[c][metric] for c in ["vit_kan", "vit_bct_kan", "vit_linear", "vit_bct_linear"])
            dk, _, _, pk = welch(kb, kp)
            dl, _, _, pl = welch(lb, lp)
            di, lo, hi, pi = interaction(kb, kp, lb, lp)
            rows.append(dict(metric=metric, drop_kan=dk, p_kan=pk, g_kan=hedges_g(kb, kp), drop_linear=dl, p_linear=pl,
                             g_linear=hedges_g(lb, lp), interaction=di, ci_lo=lo, ci_hi=hi, p_interaction=pi))
        if "vit_bct_ln_kan" in g:
            for metric in ["selected"]:
                d, lo, hi, p = welch(g["vit_bct_ln_kan"][metric], g["vit_bct_kan"][metric])
                rows.append(dict(metric=f"{metric}: bct_ln_kan - bct_kan", interaction=d, ci_lo=lo, ci_hi=hi,
                                 p_interaction=p, g_kan=hedges_g(g["vit_bct_ln_kan"][metric], g["vit_bct_kan"][metric])))
        t = pd.DataFrame(rows).round(3)
        t.to_csv(os.path.join(OUT, "isic_tests.csv"), index=False)
        lines.append("### BCT drop per head and interaction (Welch-Satterthwaite)\n\n" + md(t, index=False) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Revision results (generated by analyze.py)\n"]
    fz = os.path.join(common.RESULT_DIR, "frozen_results.csv")
    if os.path.exists(fz):
        frozen_tables(fz, lines)
        hypothesis_tests(pd.read_csv(os.path.join(OUT, "frozen_selected_runs.csv")), lines)
    ep = os.path.join(common.RESULT_DIR, "isic_finetune", "epochs.csv")
    if os.path.exists(ep):
        isic_tables(ep, lines)
    dd = os.path.join(common.RESULT_DIR, "dataset_descriptors.csv")
    if os.path.exists(dd):
        lines.append("## Feature-space descriptors\n\n" + md(pd.read_csv(dd).round(3), index=False) + "\n")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f">> tables written to {OUT}")


if __name__ == "__main__":
    main()
