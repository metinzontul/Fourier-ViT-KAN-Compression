# Original submission (v1) – archived files and known errors

This folder contains the scripts, logs and README of the original submission of manuscript applsci-4699732, moved here unchanged in release v2.0. They are kept so that the results reviewed in the first round can be traced. **They are not used to produce any result of the revised article.** The revised experiments, partitions, logs and results are in the repository root (see the main [README](../../README.md)).

The state of the repository during the first review is also preserved by the tags `v1.0.0` and `v1.1-reviewed`. The model weights referred to in `weights/` are attached to release v1.0.0.

## Errors found in the code audit

The audit carried out for the revision found the following problems. The same list is given in the appendix of the response to the reviewers.

| Issue in the original submission | Affected results | Action in the revision |
|---|---|---|
| ResNet50 "KAN" and "BCT" were placeholder linear layers (MNIST, EuroSAT, ISIC 2019) | ResNet50 rows of the original Tables 1–4; parameter counts 20,490 / 16,392 / 263,562 / 263,304 | Removed; replaced by the frozen-feature study |
| ResNet50 ISIC 2019 loaded the wrong folders and trained on 2 classes | ResNet50 ISIC 2019 results | Removed; ResNet50 on ISIC 2019 re-evaluated with 8 classes on the lesion-grouped partition |
| ISIC 2019 split at the image level; about 60% of validation images shared a lesion with training images | All ISIC 2019 results (e.g. macro F1 78.9) | New lesion-grouped 70/10/20 partition; all ISIC runs repeated; test partition reported |
| Model selected and reported on the same validation set | ISIC 2019 tables | Selection on validation, reporting on an independent test partition |
| ResNet50 CIFAR-10: best epoch chosen on the test set; p = 0.0226 from an unpaired test on accuracy | ResNet50 CIFAR-10 comparison | Removed (re-analysis at the final epoch: difference +0.10 pp, paired p = 0.79); replaced by the frozen-feature study |
| ViT "KAN" head was a bias-free weighted sum; ViT "MLP" was ReLU → Linear | ViT KAN/MLP rows | Real B-spline KAN and hidden-layer MLP used; the original variants are kept only as labelled control heads (KAN_orig, MLP_orig) |
| ResNet50 CIFAR-10 "block-circulant" was the spectral BCT; the optimizer was AdamW, not Adam; seeds 42–46 | Methods text | Methods rewritten; every operator defined as implemented |
| The spectral BCT used 65 of its 128 frequency weights (irfft with n = d) | BCT parameter counts | Nominal and effective parameter counts reported |
| Parameter counts mixed the frozen backbone and the trainable modules; complex weights counted once | Original Tables 1–4 | Module and whole-model counts reported separately; complex weights counted as two real parameters |
| Latency measured on randomly initialised models at batch 1 with varying protocols | Latency columns | One benchmarking protocol for all configurations (`benchmark_efficiency.py`) |
| EuroSAT split without a fixed seed | EuroSAT results | Fixed, saved stratified 70/10/20 partition (`splits/EuroSAT_split.csv`) |
| README with a wrong title and journal, files listed but missing (`kan_layer.py`, `vit_linear.py`), script error (undefined `KANLayer`), citation artefacts in a cleaned log | Reproducibility | Release v2.0 with new scripts, partitions, logs, environment file and a README that maps every table and figure to a command |

In addition, the manuscript described a LayerNorm, a skip path and a SiLU activation in the block-circulant layer that the code did not contain, and the auxiliary confusion matrices and SHAP maps (former Figures 10 and 11) came from models that differed from the main ablation. These figures and the attribution analysis were removed from the revised article.

## File inventory

| Path | Original purpose | Status |
|---|---|---|
| `README.md` | README of the original submission | Superseded by the main README |
| `requirements.txt` | package list of the original submission (includes SHAP, OpenCV, seaborn) | Superseded by `environment.yml` and `requirements.txt` in the root |
| `models/bct_module.py` | spectral BCT, block-circulant transition and a minimal "KAN" layer | Superseded by `heads.py` (operators re-implemented and documented; the spectral BCT kept verbatim for comparability) |
| `models/vit_kan.py`, `models/vit_mlp.py`, `models/resnet_kan.py` | model wrappers of the original experiments | Superseded by `heads.py`, `frozen_study.py` and `isic_finetune.py` |
| `ablation_scripts/vit_comprehensive_ablation.py` | ViT ablation on CIFAR-10, MNIST, EuroSAT | Superseded by the frozen-feature study (`frozen_study.py`) |
| `ablation_scripts/resnet50_comprehensive_ablation.py` | ResNet50 ablation on MNIST, EuroSAT, CIFAR-10 | Superseded by the frozen-feature study |
| `ablation_scripts/isic2019_vit_bct_kan_ablation.py`, `isic2019_vit_bct_linear_ablation.py` | ISIC 2019 ViT fine-tuning with an image-level split | Superseded by `isic_finetune.py` on the lesion-grouped partition |
| `ablation_scripts/SHAP_analysis_isic2019.py` | SHAP maps (former Figure 11) | Removed from the article; not replaced |
| `visualization/plot_PSD.py` | image-space power spectral density (former Figure 1) | Replaced by the channel-axis spectrum analysis (`dataset_descriptors.py`, `plot_latent_spectrum.py`; Figure 3) |
| `visualization/generate_figure4_cifar10.py`, `generate_figure8_isic_dichotomy.py`, `generate_figure9_isic2019_learning_curve.py`, `generate_figure10_isic_cm.py` | figures of the original submission | Replaced by `plot_width_sweep.py` and `plot_isic_finetune.py` (Figures 4–6) |
| `logs/*.txt`, `logs/original_raw_logs/*.txt` | cleaned and raw logs of the original runs | Kept for traceability; the revised logs are in `results_final/` |
| `weights/` | download script for the original ISIC 2019 checkpoints (release v1.0.0) | Kept for traceability; the checkpoints belong to the image-level split |
