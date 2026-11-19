# Performance–Compression Trade-offs in Vision Classifier Heads

Code, data partitions, logs and results for the article

> M. Zontul, **"Performance–Compression Trade-offs in Vision Classifier Heads: A Controlled Comparison of KAN, MLP, and Linear Configurations"**, *Applied Sciences* (MDPI), manuscript applsci-4699732 (revised version).

This is **release v2.0**, the version that accompanies the revised manuscript. The scripts and logs of the original submission are kept unchanged in [`archive/v1_original/`](archive/v1_original/); [`archive/v1_original/NOTES.md`](archive/v1_original/NOTES.md) lists the errors found in them and how the revision handles each one. The state of the repository at the time of the first review is preserved by the tags `v1.0.0` and `v1.1-reviewed`.

## What the study does

Frozen ResNet50 and ViT-B/16 features from MNIST, CIFAR-10, EuroSAT and ISIC 2019 are reduced to a common width by one of six settings (none, the Fourier-based banded compression transform **BCT**, a block-circulant layer, a learned linear layer, PCA, a random orthonormal projection) and classified by one of four heads (linear, MLP, B-spline KAN, layer-normalized KAN). All configurations share one optimizer, a learning rate tuned per configuration on the validation partition, and five seeds; results are reported on independent test partitions. ISIC 2019 is partitioned by lesion. A separate experiment fine-tunes ViT-B/16 end to end on ISIC 2019.

## Repository layout

```text
.
├── common.py                 paths, dataset builders, fixed partitions
├── heads.py                  reduction operators (BCT, block-circulant, PCA, random, learned linear) and heads
├── extract_features.py       caches frozen backbone features
├── frozen_study.py           frozen-feature study (main grid, BCT-perm, width sweep, control heads)
├── isic_finetune.py          end-to-end ViT-B/16 fine-tuning on ISIC 2019
├── dataset_descriptors.py    feature-space descriptors and channel-spectrum permutation tests
├── benchmark_efficiency.py   parameters, FLOPs, memory and latency under one protocol
├── analyze.py                learning-rate selection, all tables and hypothesis tests
├── penalty_table.py          compression penalty with seed variability (Table 2)
├── make_efficiency_table.py  efficiency tables (Table 5, Table S11)
├── isic_per_class.py         per-class F1 from the fine-tuning confusion matrices (Table S10)
├── coverage_analysis.py      grid coverage vs. additional KAN loss (Section 4.3)
├── plot_latent_spectrum.py   Figure 3
├── plot_width_sweep.py       Figure 4
├── plot_isic_finetune.py     Figures 5 and 6
├── make_supplement.py        Supplementary Materials (Tables S1–S11)
├── run_final.bat             runs the whole pipeline into results_final/
├── splits/                   fixed partitions used in every experiment
├── results_final/            all result files, logs, tables and figures of the article
├── figures_src/              LaTeX/TikZ sources of the schematic Figures 1 and 2
├── tools/diagnostics/        diagnostic scripts used during the revision (not part of the pipeline)
└── archive/v1_original/      scripts and logs of the original submission, with NOTES.md
```

## Installation

Python 3.10, PyTorch 2.5.1 with CUDA 12.1 (the experiments were run on an NVIDIA GeForce RTX 4080, 16 GB).

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt           # packages and versions used by the scripts
```

`environment.yml` is the complete export of the conda environment on the experiment machine (Windows). It contains further packages that this repository does not use; `requirements.txt` lists only the packages the scripts need, with the versions that produced the results.

## Data

The datasets are not redistributed. Download them from their original sources and arrange them as follows (the root can be changed with the environment variable `REV_DATA_ROOT`; default `../data`):

```text
<DATA_ROOT>/cifar-10-batches-py/
<DATA_ROOT>/MNIST/raw/
<DATA_ROOT>/eurosat/2750/                       (EuroSAT RGB)
<DATA_ROOT>/ISIC2019/train/<class>/*.jpg
<DATA_ROOT>/ISIC2019/isic2019_lesion_grouped_split.csv   (optional; splits/ is used otherwise)
```

The partitions are fixed and stored in [`splits/`](splits/):

| File | Content |
|---|---|
| `MNIST_val_indices.csv`, `CIFAR10_val_indices.csv` | indices of the official training set held out for validation (10%); the official test sets are used for testing |
| `EuroSAT_split.csv` | stratified 70/10/20 partition (seed 42) |
| `isic2019_lesion_grouped_split.csv` | stratified 70/10/20 partition grouped by `lesion_id`; images without an identifier form their own group; no lesion is shared between partitions |

## Reproducing the results

All commands are run from the repository root. Results are written to the folder given by `REV_RESULT_DIR` (default `results/`); `run_final.bat` uses `results_final/`.

```bat
:: 1) cache frozen features once (both backbones, four datasets)
python extract_features.py

:: 2) ISIC 2019 fine-tuning (GPU, about 40 min per run)
python isic_finetune.py --configs vit_kan vit_bct_kan vit_linear vit_bct_linear vit_bct_ln_kan --seeds 42 43 44 45 46

:: 3) efficiency benchmark
python benchmark_efficiency.py

:: 4) frozen-feature study, analysis, tables, figures and supplement (about 10 h on an RTX 4080)
run_final.bat
```

`run_final.bat` expects the fine-tuning results in `results/isic_finetune/` and `results/efficiency.csv` and copies them to `results_final/`. Interrupted runs of `frozen_study.py` can be restarted; finished runs are skipped. Every step after the training runs takes seconds, so all tables and figures can be regenerated from the files in `results_final/` without retraining:

```bat
set REV_RESULT_DIR=%CD%\results_final
python analyze.py & python penalty_table.py & python make_efficiency_table.py & python isic_per_class.py
python coverage_analysis.py & python plot_latent_spectrum.py & python plot_width_sweep.py
python plot_isic_finetune.py & python make_supplement.py
```

## Where each table and figure comes from

| Article | Script | Output in `results_final/` |
|---|---|---|
| Table 1 (datasets and partitions) | `splits/`, `dataset_descriptors.py` | `splits/*.csv`, `dataset_descriptors.csv` |
| Table 2 (compression penalty, mean ± SD) | `analyze.py`, `penalty_table.py` | `tables/frozen_penalty.csv`, `tables/frozen_penalty_sd.csv` |
| Table 3 (hypotheses H1–H4) | `analyze.py` | `tables/hypothesis_tests_summary.csv`, `tables/hypothesis_tests_cells.csv` |
| Table 4 (ISIC 2019 fine-tuning) | `isic_finetune.py`, `analyze.py` | `isic_finetune/summary.csv`, `tables/isic_summary.csv`, `tables/isic_tests.csv` |
| Table 5 (efficiency) | `benchmark_efficiency.py`, `make_efficiency_table.py` | `efficiency.csv`, `tables/efficiency_main.csv` |
| Figure 1 (heads, KAN edge function) | `figures_src/figure_heads.tex` (pdflatex) | – |
| Figure 2 (experimental design) | `figures_src/figure_design.tex` (pdflatex) | – |
| Figure 3 (channel-axis spectrum) | `dataset_descriptors.py`, `plot_latent_spectrum.py` | `latent_spectra.csv`, `figures/latent_spectrum.pdf` |
| Figure 4 (width sweep) | `plot_width_sweep.py` | `figures/width_sweep_linear.pdf` |
| Figure 5 (training curves) | `plot_isic_finetune.py` | `figures/isic_curves.pdf` |
| Figure 6 (confusion matrices) | `plot_isic_finetune.py` | `figures/isic_confusion.pdf` |
| Section 4.3 (coverage vs. KAN loss) | `coverage_analysis.py` | `tables/coverage_vs_interaction*.csv` |
| Tables S1–S11 | `make_supplement.py` (with `isic_per_class.py` for S10) | `supplement/supplementary.tex`, `supplement/supplementary.pdf` |
| Numbers quoted in the text | `analyze.py` | `tables/report.md` |

## Result files

* `frozen_results.csv` – one row per frozen-feature run (tuning and final runs, all seeds), with validation and test metrics, parameter counts, the standard deviation of the head input and its spline-grid coverage.
* `tables/frozen_selected_runs.csv` – the runs at the selected learning rate (five seeds per configuration).
* `isic_finetune/epochs.csv`, `summary.csv`, `cm_<config>_seed<seed>.npy` – per-epoch metrics, one row per fine-tuning run, and the test confusion matrices at the validation-selected epoch.
* `logs/` – console logs of the frozen-feature runs.

Model checkpoints and cached features are not included because of their size.

## Notes on the implementation

* Parameter counts are real-valued: complex weights count twice. `params_module_effective` excludes BCT weights that never receive gradients.
* `head_input_in_grid` is the fraction of head inputs inside the support of the B-spline basis (about [−2.2, 2.2] for G = 5, s = 3).
* The scripts in `tools/diagnostics/` document a numerical issue found during the revision (batched cuFFT evaluation of the ResNet50 BCT in large chunks) and the removal of the affected preliminary rows. The final results in `results_final/` were produced after the fix by a complete re-run with `run_final.bat`.

## Use of generative AI

During the revision, Claude Opus 5.5 (Anthropic) was used to assist in writing and reviewing the experiment and analysis code, including the plotting scripts and the TikZ sources of Figures 1 and 2. The author ran all experiments, checked the code and its outputs, and takes full responsibility for the content of this repository.

## License

MIT License (see [LICENSE](LICENSE)).

## Citation

Please cite the article once it is published. Until then:

```bibtex
@unpublished{zontul2026heads,
  author = {Metin Zontul},
  title  = {Performance--Compression Trade-offs in Vision Classifier Heads: A Controlled Comparison of {KAN}, {MLP}, and Linear Configurations},
  note   = {Manuscript applsci-4699732, Applied Sciences (MDPI), under review},
  year   = {2026}
}
```
