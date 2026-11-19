# -*- coding: utf-8 -*-
"""
Compression penalty with seed variability (main-text Table 1).

penalty = head-averaged test macro F1 with the reducer (d_out = 128) minus head-averaged test macro F1
without reduction, over the four heads (Linear, MLP, KAN, LN+KAN). For each seed s the difference is
computed between runs with the same seed, giving five seed-wise penalties per cell; the table reports
their mean (identical to tables/frozen_penalty.csv) and standard deviation.

Inputs : results/tables/frozen_selected_runs.csv (from analyze.py)
Outputs: results/tables/frozen_penalty_sd.csv

    python penalty_table.py
"""
import os

import pandas as pd

import common

HEADS = ["linear", "mlp", "kan", "kan_ln"]
DS = ["MNIST", "CIFAR10", "EuroSAT", "ISIC2019"]


def main():
    f = pd.read_csv(os.path.join(common.RESULT_DIR, "tables", "frozen_selected_runs.csv"))
    f = f[f["head"].isin(HEADS) & ((f.width == 128) | (f.reducer == "none"))]
    s = f.groupby(["dataset", "backbone", "reducer", "seed"]).agg(f1=("test_f1", "mean"),
                                                                   nh=("head", "nunique")).reset_index()
    if (s.nh != len(HEADS)).any():
        print("!! WARNING: some seeds lack one of the four heads")
    base = s[s.reducer == "none"].set_index(["dataset", "backbone", "seed"]).f1
    p = s[s.reducer != "none"].copy()
    p["penalty"] = p.f1.values - base.reindex(pd.MultiIndex.from_frame(p[["dataset", "backbone", "seed"]])).values
    out = (p.groupby(["backbone", "reducer", "dataset"]).penalty.agg(mean="mean", sd="std", n="count")
             .reset_index())
    out.to_csv(os.path.join(common.RESULT_DIR, "tables", "frozen_penalty_sd.csv"), index=False)
    t = out.assign(cell=[f"{m:+.2f} ± {sd:.2f}" for m, sd in zip(out["mean"], out.sd)]) \
           .pivot_table(index=["backbone", "reducer"], columns="dataset", values="cell", aggfunc="first")[DS]
    print(t.to_string())
    print(">> wrote tables/frozen_penalty_sd.csv")


if __name__ == "__main__":
    main()
