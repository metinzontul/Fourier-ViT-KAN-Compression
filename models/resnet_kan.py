# -*- coding: utf-8 -*-
"""
ResNet-50 Architectures with KAN Classification Heads.

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: Contains both the Baseline and Proposed BCT-compressed ResNet50+KAN architectures.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from .bct_module import BlockCirculantTransition, MinimalKANLayer

class ResNet50_KAN_Baseline(nn.Module):
    """Uncompressed Baseline Model (ResNet50 + KAN Head)"""
    def __init__(self, num_classes=10):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Outputs (Batch, 2048, 1, 1)
        
        # Direct ingestion of the massive 2048-dim vector into KAN
        self.head = MinimalKANLayer(in_features=2048, num_classes=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1) 
        return self.head(features)

class ResNet50_BCT_KAN_Proposed(nn.Module):
    """Proposed Model (ResNet50 + BCT + KAN Head) utilizing Block-Circulant Truncation"""
    def __init__(self, num_classes=10, compressed_dim=128):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Architecture Pipeline: 2048 -> BCT -> 128 -> KAN -> Classes
        self.bct = BlockCirculantTransition(in_features=2048, out_features=compressed_dim) 
        self.kan = MinimalKANLayer(in_features=compressed_dim, num_classes=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        compressed = self.bct(features)
        return self.kan(compressed)