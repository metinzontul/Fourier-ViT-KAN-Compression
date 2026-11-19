# -*- coding: utf-8 -*-
"""
Step 4 - Measurable descriptors of each (dataset, backbone) feature space
(answers R1.3, R2.3, R3.6 and supports the dataset-dependence discussion, R2.4).

Computed on the cached TRAINING features only (CPU, a few minutes, can run while the GPU is busy):

  Latent (feature-axis) spectrum - what the spectral BCT actually operates on
    * bct_kept_energy      : share of feature-axis energy in the bins the BCT keeps
                             (rFFT bins 0..d//2, i.e. what irfft(n=d) reads; d = 128)
    * dc_energy            : share of energy in the DC bin alone
    * bct_kept_energy_perm : the same after a random permutation of the channels (mean of 5 permutations);
                             if the two are equal, the channel order carries no spectral structure
    * spectral_flatness    : geometric / arithmetic mean of the power spectrum (1 = white)
  Feature geometry
    * pca_dim_90, pca_dim_95 : principal components needed for 90 % / 95 % of the variance
    * participation_ratio    : (sum lambda)^2 / sum lambda^2, a smooth intrinsic-dimension estimate
    * pca128_var             : variance retained by the best possible 128-dim linear projection
    * nonneg_share           : share of non-negative feature values (post-ReLU features -> ~1)
  Permutation test (500 random channel orders, 5,000-row subsample)
    * perm_*    : BCT-kept energy share vs. its null distribution (z-score, two-sided empirical p)
    * perm_ac_* : the same without the DC bin (the DC bin is invariant to channel order)
  Label statistics
    * num_classes, imbalance_ratio (largest / smallest class)

Outputs: results/dataset_descriptors.csv and results/latent_spectra.csv (mean power spectrum
per dataset/backbone, original and permuted channel order) for plotting.

    python dataset_descriptors.py
"""
import os
import argparse
import numpy as np
import pandas as pd

import common


def power_spectrum(X, chunk=4096):
    """Mean |FFT|^2 along the feature axis (full-length FFT, so Parseval holds exactly)."""
    D = X.shape[1]
    acc = np.zeros(D, dtype=np.float64)
    for i in range(0, len(X), chunk):
        F = np.fft.fft(X[i:i + chunk].astype(np.float64), axis=1)
        acc += (np.abs(F) ** 2).sum(0)
    return acc / len(X)


def kept_share(P, d):
    """Energy share of the rFFT bins 0..d//2 (and their conjugate mirror) that irfft(n=d) reads."""
    D, k = len(P), d // 2
    kept = P[0] + P[1:k + 1].sum() + P[D - k:].sum()
    return kept / P.sum()


def permutation_test(X, d, n_perm=500, n_sub=5000, seed=0):
    """Null distribution of the BCT-kept energy share under random channel orders.
    Uses a fixed random subsample of n_sub rows; returns observed share, null mean/sd,
    z-score and a two-sided empirical p-value, plus the same quantities without the DC bin."""
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]

    def shares(Z):
        P = power_spectrum(Z)
        k, D = d // 2, len(P)
        kept = P[0] + P[1:k + 1].sum() + P[D - k:].sum()
        return kept / P.sum(), (kept - P[0]) / (P.sum() - P[0])

    obs, obs_ac = shares(Xs)
    null = np.array([shares(Xs[:, rng.permutation(Xs.shape[1])]) for _ in range(n_perm)])
    out = {}
    for name, o, nl in (("perm", obs, null[:, 0]), ("perm_ac", obs_ac, null[:, 1])):
        dev = np.abs(nl - nl.mean())
        out.update({f"{name}_observed": o, f"{name}_null_mean": nl.mean(), f"{name}_null_sd": nl.std(ddof=1),
                    f"{name}_z": (o - nl.mean()) / nl.std(ddof=1),
                    f"{name}_p": (1 + np.sum(dev >= abs(o - nl.mean()))) / (n_perm + 1)})
    out["perm_n"] = n_perm
    out["perm_subsample"] = len(Xs)
    return out


def describe(X, y, d, n_perm=5, seed=0):
    P = power_spectrum(X)
    rng = np.random.default_rng(seed)
    perm_shares, P_perm = [], np.zeros_like(P)
    for _ in range(n_perm):
        Pp = power_spectrum(X[:, rng.permutation(X.shape[1])])
        perm_shares.append(kept_share(Pp, d))
        P_perm += Pp / n_perm

    Xc = X - X.mean(0)
    lam = np.clip(np.linalg.eigvalsh(np.cov(Xc.astype(np.float64), rowvar=False))[::-1], 0, None)
    cum = np.cumsum(lam) / lam.sum()
    counts = np.bincount(y)
    counts = counts[counts > 0]
    Ppos = P[1:len(P) // 2 + 1]  # one-sided spectrum without DC, for flatness
    return dict(
        bct_kept_energy=kept_share(P, d),
        dc_energy=P[0] / P.sum(),
        bct_kept_energy_perm=float(np.mean(perm_shares)),
        bct_kept_energy_perm_sd=float(np.std(perm_shares)),
        spectral_flatness=float(np.exp(np.mean(np.log(Ppos + 1e-12))) / np.mean(Ppos)),
        pca_dim_90=int(np.searchsorted(cum, 0.90) + 1),
        pca_dim_95=int(np.searchsorted(cum, 0.95) + 1),
        participation_ratio=float(lam.sum() ** 2 / (lam ** 2).sum()),
        pca128_var=float(cum[d - 1]),
        nonneg_share=float((X >= 0).mean()),
        num_classes=int(len(counts)),
        imbalance_ratio=float(counts.max() / counts.min()),
        n_train=int(len(X)), feature_dim=int(X.shape[1]),
    ), P, P_perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=common.DATASETS)
    ap.add_argument("--backbones", nargs="+", default=common.BACKBONES)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n_perm", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(common.RESULT_DIR, exist_ok=True)

    rows, spectra = [], []
    for bb in args.backbones:
        for ds in args.datasets:
            z = np.load(common.feature_path(ds, bb, "train"))
            X, y = z["X"].astype(np.float32), z["y"]
            r, P, Pp = describe(X, y, args.width)
            r.update(permutation_test(X, args.width, n_perm=args.n_perm))
            rows.append(dict(dataset=ds, backbone=bb, **r))
            D = len(P)
            for k in range(D // 2 + 1):
                spectra.append(dict(dataset=ds, backbone=bb, bin=k, freq=k / D,
                                    power=P[k] / P.sum(), power_perm=Pp[k] / Pp.sum()))
            print(f"   {ds:9s} {bb:9s} kept={r['bct_kept_energy']:.3f} (perm {r['bct_kept_energy_perm']:.3f}) "
                  f"DC={r['dc_energy']:.3f} pca95={r['pca_dim_95']} PR={r['participation_ratio']:.1f} "
                  f"pca128={r['pca128_var']:.3f} nonneg={r['nonneg_share']:.2f} | "
                  f"perm test (no DC): z={r['perm_ac_z']:+.2f} p={r['perm_ac_p']:.3f}")

    out = os.path.join(common.RESULT_DIR, "dataset_descriptors.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(spectra).to_csv(os.path.join(common.RESULT_DIR, "latent_spectra.csv"), index=False)
    print(f">> wrote {out}")


if __name__ == "__main__":
    main()
