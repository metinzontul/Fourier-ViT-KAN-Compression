# -*- coding: utf-8 -*-
"""
Exploratory: relation between the grid coverage of the KAN input and the additional KAN loss under
lossy reduction (the H3 cells: BCT, block-circulant, PCA, random projection; d_out = 128; frozen features).

interaction = (KAN drop) - (Linear drop), drop = uncompressed test macro F1 - reduced test macro F1,
using the seed means of the final runs (positive = KAN loses more). Coverage = mean fraction of KAN
head inputs inside the spline grid (test set, final runs).

Inputs : results/frozen_results.csv
Outputs: results/tables/coverage_vs_interaction.csv          (one row per cell)
         results/tables/coverage_vs_interaction_summary.csv  (correlations and means per reducer)

    python coverage_analysis.py
"""
import os

import pandas as pd
from scipy import stats

import common

LOSSY = ["bct", "blockcirc", "pca", "random"]


def main():
    d = pd.read_csv(os.path.join(common.RESULT_DIR, "frozen_results.csv"))
    d = d[(d.stage == "final") & ((d.width == 128) | (d.reducer == "none"))]
    g = (d.groupby(["dataset", "backbone", "reducer", "head"])
          .agg(f1=("test_f1", "mean"), cov=("head_input_in_grid", "mean")).reset_index())
    f1 = g.pivot_table(index=["dataset", "backbone", "reducer"], columns="head", values="f1")
    cov = g[g["head"] == "kan"].set_index(["dataset", "backbone", "reducer"])["cov"]
    rows = []
    for (ds, bb, red), r in f1.iterrows():
        if red not in LOSSY:
            continue
        base = f1.loc[(ds, bb, "none")]
        inter = (base["kan"] - r["kan"]) - (base["linear"] - r["linear"])
        rows.append(dict(dataset=ds, backbone=bb, reducer=red, kan_coverage=cov[(ds, bb, red)], interaction=inter))
    t = pd.DataFrame(rows)
    out = os.path.join(common.RESULT_DIR, "tables")
    os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "coverage_vs_interaction.csv"), index=False)
    rs, ps = stats.spearmanr(t.kan_coverage, t.interaction)
    rp, pp = stats.pearsonr(t.kan_coverage, t.interaction)
    by_red = t.groupby("reducer")[["kan_coverage", "interaction"]].mean().reset_index()
    summ = pd.concat([
        pd.DataFrame([dict(scope="all H3 cells", n_cells=len(t), spearman_rho=rs, spearman_p=ps,
                           pearson_r=rp, pearson_p=pp, kan_coverage=t.kan_coverage.mean(),
                           interaction=t.interaction.mean())]),
        by_red.rename(columns={"reducer": "scope"}).assign(n_cells=t.groupby("reducer").size().values)],
        ignore_index=True)
    summ.to_csv(os.path.join(out, "coverage_vs_interaction_summary.csv"), index=False)
    print(f"KAN coverage vs interaction over {len(t)} cells: Spearman rho = {rs:+.2f} (p = {ps:.3f}), "
          f"Pearson r = {rp:+.2f} (p = {pp:.3f})")
    print(by_red.set_index("reducer").round(2).to_string())
    print(">> wrote tables/coverage_vs_interaction.csv and tables/coverage_vs_interaction_summary.csv")


if __name__ == "__main__":
    main()
