# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) Architectures with MLP Classification Heads.

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: Contains both the Baseline and Proposed BCT-compressed ViT+MLP architectures.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .bct_module import BCT, BlockCirculantTransition

class ViT_MLP_Baseline(nn.Module):
    """Uncompressed Baseline Model (ViT + Standard MLP Head)"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity() 
        
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.mlp_head(self.backbone(x))

class ViT_BCT_MLP_Proposed(nn.Module):
    """Proposed Model (ViT + BCT + Standard MLP Head)"""
    def __init__(self, num_classes, compressed_dim=128, use_block_circulant=False):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity() 
        
        if use_block_circulant:
            self.bct = BlockCirculantTransition(in_features=in_features, out_features=compressed_dim)
        else:
            self.bct = BCT(in_features=in_features, out_features=compressed_dim)
            
        self.mlp_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(compressed_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.bct(x)
        return self.mlp_head(x)