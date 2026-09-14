# -*- coding: utf-8 -*-
"""
Core modules for Frequency-Domain Compression and Kolmogorov-Arnold Networks.

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: Contains the Banded Compression Transform (BCT) modules and 
the Minimal KAN layer utilized across all hybrid architectures.
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import math

class BCT(nn.Module):
    """
    Banded Compression Transform (BCT) Layer (Optimized for Medical Imaging).
    Aggressively truncates high-frequency spatial components in the Fourier domain
    and projects them via trainable complex weights to strictly reduce dimensionality.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable complex weights for the pruned frequencies (Projection)
        self.freq_weight = nn.Parameter(torch.view_as_real(torch.randn(out_features, dtype=torch.cfloat)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 1. Spatial to Frequency Domain (FFT)
        x_freq = torch.fft.rfft(x, dim=1)
        
        # 2. Direct High-Frequency Truncation (Spectral Truncation)
        x_freq_compressed = x_freq[:, :self.out_features]
        
        # 3. Frequency-domain weighting (Complex Projection)
        weight_complex = torch.view_as_complex(self.freq_weight)
        x_freq_compressed = x_freq_compressed * weight_complex
        
        # 4. Inverse FFT directly back to the compressed spatial dimension
        out = torch.fft.irfft(x_freq_compressed, n=self.out_features, dim=1)
        return out + self.bias


class BlockCirculantTransition(nn.Module):
    """
    Block-Circulant Transition Layer (Optimized for Macroscopic Vision Tasks).
    Reduces parameters efficiently using block-wise cross-frequency interactions.
    """
    def __init__(self, in_features=1280, out_features=128, block_size=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.p = in_features // block_size  
        self.q = out_features // block_size 
        
        self.weight = nn.Parameter(torch.randn(self.q, self.p, self.block_size) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.channel_scale = nn.Parameter(torch.ones(out_features))
        
        self.pre_norm = nn.LayerNorm(in_features)
        self.post_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        batch_size = x.size(0)
        
        x_reshaped = x_norm.view(batch_size, self.p, self.block_size)
        x_fft = torch.fft.rfft(x_reshaped, dim=-1)
        w_fft = torch.fft.rfft(self.weight, dim=-1)
        
        y_fft = torch.einsum('bpf,qpf->bqf', x_fft, w_fft)
        y_reshaped = torch.fft.irfft(y_fft, n=self.block_size, dim=-1)
        
        y = y_reshaped.reshape(batch_size, self.out_features)
        y = self.post_norm(y)
        y = (y * self.channel_scale) + self.bias
        
        # Parameter-free skip connection
        x_skip = x_norm.view(batch_size, self.out_features, -1).mean(dim=2)
        return F.silu(y + x_skip)


class MinimalKANLayer(nn.Module):
    """
    Minimal Kolmogorov-Arnold Network (KAN) Layer.
    Utilizes B-Spline-like symbolic parameterization on the edges.
    """
    def __init__(self, in_features, num_classes, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_weights = nn.Parameter(torch.randn(num_classes, in_features, grid_size) / math.sqrt(in_features))

    def forward(self, x):
        bases = torch.stack([torch.sin((i+1) * x) for i in range(self.grid_size)], dim=-1)
        return torch.einsum('big,oig->bo', bases, self.spline_weights)