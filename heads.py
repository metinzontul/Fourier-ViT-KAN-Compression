# -*- coding: utf-8 -*-
"""
Reducers (D -> d) and classifier heads (d -> C) for the controlled frozen-feature study.

The two Fourier reducers and the spline KAN are copied verbatim from the original
experiment scripts so that the revision evaluates exactly the published operators:
  * SpectralBCT          <- isic2019_vit_bct_kan_ablation.py / resnet50_cifar10_ablation.py (class BCT)
  * BlockCirculant       <- vit_*_ablation_advanced_metrics.py (class BlockCirculantTransition)
  * SplineKAN            <- isic2019_vit_bct_kan_ablation.py (class KAN)
Non-Fourier reducers of the same output width (R2.5, R3.2): learned linear, PCA, random projection.
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- reducers
class SpectralBCT(nn.Module):
    """Original latent BCT: rFFT along the feature axis, keep the first d bins,
    complex re-weighting, inverse rFFT of length d. NOTE: irfft(n=d) reads only the
    first d//2+1 bins, so the remaining weights never receive gradients (reported)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.freq_weight = nn.Parameter(torch.view_as_real(torch.randn(out_features, dtype=torch.cfloat)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=1)
        x_freq_compressed = x_freq[:, :self.out_features]
        weight_complex = torch.view_as_complex(self.freq_weight)
        x_freq_compressed = x_freq_compressed * weight_complex
        out = torch.fft.irfft(x_freq_compressed, n=self.out_features, dim=1)
        return out + self.bias

    def effective_params(self):
        k = self.out_features // 2 + 1          # bins actually read by irfft
        return 2 * k + self.out_features         # complex weights used + bias (upper bound)


class BlockCirculant(nn.Module):
    """Original ViT block-circulant transition (block size 16, frequency-domain weights)."""
    def __init__(self, in_features, out_features, block_size=16):
        super().__init__()
        assert in_features % block_size == 0 and out_features % block_size == 0
        self.block_size = block_size
        fft_size = (block_size // 2) + 1
        weight_shape = (out_features // block_size, in_features // block_size, fft_size)
        self.freq_weights = nn.Parameter(torch.randn(weight_shape, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        b = x.size(0)
        x_fft = torch.fft.rfft(x.view(b, -1, self.block_size), dim=-1)
        out_fft = torch.einsum('oci,bci->boi', self.freq_weights, x_fft)
        return torch.fft.irfft(out_fft, n=self.block_size, dim=-1).reshape(b, -1) + self.bias


class ChannelPermutation(nn.Module):
    """Fixed random re-ordering of the feature channels (control for the spectral BCT)."""
    def __init__(self, D, seed):
        super().__init__()
        g = torch.Generator().manual_seed(5000 + seed)
        self.register_buffer("perm", torch.randperm(D, generator=g))

    def forward(self, x):
        return x[:, self.perm]


class FixedProjection(nn.Module):
    """Non-trainable projection x -> (x - mu) W, used for PCA and random projection."""
    def __init__(self, W, mu):
        super().__init__()
        self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mu) @ self.W


_PCA_CACHE = {}


def make_reducer(kind, D, d, X_train=None, seed=0, cache_key=None):
    if kind == "none":
        return nn.Identity(), D
    if kind == "bct":
        return SpectralBCT(D, d), d
    if kind == "bct_perm":                    # same BCT after a random channel permutation
        return nn.Sequential(ChannelPermutation(D, seed), SpectralBCT(D, d)), d
    if kind == "blockcirc":
        return BlockCirculant(D, d), d
    if kind == "linear":
        return nn.Linear(D, d), d
    if kind == "pca":
        # top-d principal directions of the training features (SVD computed once per feature set)
        if cache_key not in _PCA_CACHE or cache_key is None:
            mu = X_train.mean(0)
            cov = np.cov((X_train - mu).astype(np.float64), rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            _PCA_CACHE[cache_key] = (mu, evecs[:, ::-1].astype(np.float32))
        mu, V = _PCA_CACHE[cache_key]
        return FixedProjection(V[:, :d], mu), d
    if kind == "random":
        rng = np.random.default_rng(1000 + seed)
        Q, _ = np.linalg.qr(rng.standard_normal((D, d)))   # orthonormal columns
        return FixedProjection(Q, np.zeros(D)), d
    raise ValueError(kind)


# ----------------------------------------------------------------------------- heads
class SplineKAN(nn.Module):
    """Original B-spline KAN head (grid 5, order 3, base SiLU path); grid support ~[-2.2, 2.2]."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.grid_size, self.spline_order = grid_size, spline_order
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
        h = 2 / grid_size
        grid = torch.arange(-1 - spline_order * h, 1 + spline_order * h + h / 2, h)
        self.register_buffer('grid', grid.view(1, 1, -1))

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :, :-1]) & (x < self.grid[:, :, 1:])).float()
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[:, :, :-(k + 1)]) / (self.grid[:, :, k:-1] - self.grid[:, :, :-(k + 1)] + 1e-8) * bases[:, :, :-1]
            right = (self.grid[:, :, k + 1:] - x) / (self.grid[:, :, k + 1:] - self.grid[:, :, 1:-k] + 1e-8) * bases[:, :, 1:]
            bases = left + right
        return bases

    def forward(self, x):
        base = nn.functional.linear(nn.functional.silu(x), self.base_weight)
        spline = torch.einsum('bie,oie->bo', self.b_splines(x), self.spline_weight)
        return base + spline


class OrigViTKAN(nn.Module):
    """Head published as 'KAN' in the original ViT MNIST/CIFAR-10/EuroSAT scripts
    (vit_*_ablation_baseline.py, class KANLayer): sum_g w[o,i,g] * x_i. This is a bias-free
    LINEAR map with an un-scaled N(0,1) initialisation. Kept only as a control."""
    def __init__(self, in_dim, out_dim, grid_size=5):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_dim, in_dim, grid_size))

    def forward(self, x):
        return torch.einsum('bi,oig->bo', x, self.weights)


def make_head(kind, d, C, mlp_hidden=128):
    if kind == "kan_orig":                    # original ViT 'KAN' (linear, see OrigViTKAN)
        return OrigViTKAN(d, C)
    if kind == "mlp_orig":                    # original ViT 'MLP': ReLU -> Linear, no hidden layer
        return nn.Sequential(nn.ReLU(), nn.Linear(d, C))
    if kind == "linear":
        return nn.Linear(d, C)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(d, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, C))
    if kind == "kan":
        return SplineKAN(d, C)
    if kind == "kan_ln":                      # same KAN, inputs normalised onto the spline grid
        return nn.Sequential(nn.LayerNorm(d), SplineKAN(d, C))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- accounting
def count_params(module):
    """Real-valued trainable parameter count (complex parameters count as 2)."""
    n = 0
    for p in module.parameters():
        if p.requires_grad:
            n += p.numel() * (2 if p.is_complex() else 1)
    return n


def spline_grid_coverage(x, lo=-2.2, hi=2.2):
    """Fraction of head inputs that fall inside the spline grid support."""
    return ((x > lo) & (x < hi)).float().mean().item()
