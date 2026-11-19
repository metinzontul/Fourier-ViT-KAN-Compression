# -*- coding: utf-8 -*-
"""Remove ResNet50 + spectral-BCT rows evaluated with the faulty 4096 chunk size
(a backup of the original CSV is kept as frozen_results_before_cufft_fix.csv)."""
import os, shutil
import pandas as pd
import common

f = os.path.join(common.RESULT_DIR, "frozen_results.csv")
backup = os.path.join(common.RESULT_DIR, "frozen_results_before_cufft_fix.csv")
if not os.path.exists(backup):
    shutil.copy(f, backup)
d = pd.read_csv(f)
bad = (d.backbone == "resnet50") & (d.reducer.isin(["bct", "bct_perm"]))
d[~bad].to_csv(f, index=False)
print(f"removed {bad.sum()} rows, kept {(~bad).sum()} (backup: {os.path.basename(backup)})")
