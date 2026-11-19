"""Per-class F1 for the ISIC 2019 fine-tuning runs, from the saved confusion matrices.

Reads  <RESULT_DIR>/isic_finetune/cm_<config>_seed<seed>.npy  (rows = true class, cols = predicted)
Writes <RESULT_DIR>/tables/isic_per_class.csv        (mean and sd of per-class F1 over seeds)
       <RESULT_DIR>/tables/isic_per_class_drops.csv  (BCT drop per class for KAN and Linear, LN gain)

Descriptive analysis: per-class tests are not corrected for multiple comparisons,
and the two rarest test classes (DF, VASC) have about 50 images each.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

RESULT_DIR = os.environ.get("REV_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
CM_DIR = os.path.join(RESULT_DIR, "isic_finetune")
OUT_DIR = os.path.join(RESULT_DIR, "tables")
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]  # alphabetical, as in common.py
CONFIGS = ["vit_kan", "vit_bct_kan", "vit_bct_ln_kan", "vit_linear", "vit_bct_linear"]


def per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return 100 * f1, 100 * rec, 100 * prec


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(CM_DIR, "cm_*_seed*.npy"))):
        m = re.match(r"cm_(.+)_seed(\d+)\.npy", os.path.basename(path))
        cfg, seed = m.group(1), int(m.group(2))
        cm = np.load(path)
        f1, rec, prec = per_class_f1(cm)
        for k, c in enumerate(CLASSES):
            rows.append(dict(config=cfg, seed=seed, cls=c, support=int(cm[k].sum()),
                             f1=f1[k], recall=rec[k], precision=prec[k]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no confusion matrices found in {CM_DIR}")

    summ = (df.groupby(["cls", "config"])
              .agg(support=("support", "first"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"), n=("seed", "nunique"))
              .reset_index())
    os.makedirs(OUT_DIR, exist_ok=True)
    summ.to_csv(os.path.join(OUT_DIR, "isic_per_class.csv"), index=False)

    def vals(cfg, c):
        return df[(df.config == cfg) & (df.cls == c)].sort_values("seed").f1.values

    drops = []
    for c in CLASSES:
        k, bk = vals("vit_kan", c), vals("vit_bct_kan", c)
        lin, bl = vals("vit_linear", c), vals("vit_bct_linear", c)
        lnk = vals("vit_bct_ln_kan", c)
        drops.append(dict(
            cls=c, support=int(df[df.cls == c].support.iloc[0]),
            bct_drop_kan=(k.mean() - bk.mean()), p_kan=stats.ttest_ind(k, bk, equal_var=False).pvalue,
            bct_drop_linear=(lin.mean() - bl.mean()), p_linear=stats.ttest_ind(lin, bl, equal_var=False).pvalue,
            interaction=(k.mean() - bk.mean()) - (lin.mean() - bl.mean()),
            ln_gain_over_bct_kan=(lnk.mean() - bk.mean()),
        ))
    dd = pd.DataFrame(drops)
    dd.to_csv(os.path.join(OUT_DIR, "isic_per_class_drops.csv"), index=False)

    pd.set_option("display.width", 160)
    print(summ.pivot(index="cls", columns="config", values="f1_mean").reindex(CLASSES)[CONFIGS].round(1))
    print(dd.round(3).to_string(index=False))
    n_loss = int((dd.bct_drop_kan > 0).sum())
    print(f"KAN loses F1 under BCT in {n_loss}/{len(dd)} classes "
          f"(sign test p = {stats.binomtest(n_loss, len(dd)).pvalue:.3f})")
    print(f">> wrote {os.path.join(OUT_DIR, 'isic_per_class.csv')} and isic_per_class_drops.csv")


if __name__ == "__main__":
    main()
