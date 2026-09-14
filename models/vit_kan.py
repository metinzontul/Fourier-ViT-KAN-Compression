# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) Architectures with KAN Classification Heads.

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: Contains both the Baseline and Proposed BCT-compressed ViT+KAN architectures.
"""

import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .bct_module import BCT, BlockCirculantTransition, MinimalKANLayer

class ViT_KAN_Baseline(nn.Module):
    """Uncompressed Baseline Model (ViT + KAN Head)"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = self.backbone.heads.head.in_features
        
        # Remove original MLP head
        self.backbone.heads.head = nn.Identity() 
        self.kan_head = MinimalKANLayer(in_features=in_features, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.kan_head(x)

class ViT_BCT_KAN_Proposed(nn.Module):
    """Proposed Model (ViT + BCT + KAN Head) utilizing Spectral Truncation"""
    def __init__(self, num_classes, compressed_dim=128, use_block_circulant=False):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity() 
        
        # Select compression strategy based on dataset complexity
        if use_block_circulant:
            self.bct = BlockCirculantTransition(in_features=in_features, out_features=compressed_dim)
        else:
            self.bct = BCT(in_features=in_features, out_features=compressed_dim)
            
        self.kan_head = MinimalKANLayer(in_features=compressed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bct(x)
        return self.kan_head(x)