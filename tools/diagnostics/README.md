# Diagnostic scripts (not part of the pipeline)

* `diagnose_bct_eval.py` – showed that, with CUDA 12.1 and PyTorch 2.5.1, batched cuFFT evaluation of the
  ResNet50 spectral BCT in chunks of 4096 samples gave wrong outputs (validation F1 about 92, test F1 15–86 on EuroSAT).
  `frozen_study.py` therefore evaluates in chunks of 1024 (`EVAL_CHUNK`).
* `drop_resnet_bct_rows.py` – removed the affected preliminary ResNet50 BCT rows before they were re-run.

All results in `results_final/` come from the complete re-run with `run_final.bat` after the fix.
Run the scripts from the repository root with `PYTHONPATH=.` (they import `common` and `heads`).
