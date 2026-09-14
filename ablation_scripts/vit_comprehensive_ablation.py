# -*- coding: utf-8 -*-
"""
Comprehensive Vision Transformer (ViT) Ablation Study
Comparing Topological Vulnerability: ViT+BCT+KAN vs. ViT+BCT+MLP

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Evaluates non-linear KAN topologies against standard MLPs under extreme spectral pruning.
- Supports MNIST, CIFAR-10, and EuroSAT.
"""

import os
import time
import sys
import argparse
import math
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# =============================================================================
# 1. CORE ARCHITECTURAL MODULES (BCT & KAN)
# =============================================================================

class BlockCirculantTransition(nn.Module):
    """Banded Compression Transform (BCT) Module."""
    def __init__(self, in_features=768, out_features=128, block_size=16):
        super().__init__()
        self.block_size = block_size
        fft_size = (block_size // 2) + 1 
        weight_shape = (out_features // block_size, in_features // block_size, fft_size)
        
        self.freq_weights = nn.Parameter(torch.randn(weight_shape, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, self.block_size)
        x_fft = torch.fft.rfft(x_reshaped, dim=-1)
        out_fft = torch.einsum('oci,bci->boi', self.freq_weights, x_fft)
        x_out = torch.fft.irfft(out_fft, n=self.block_size, dim=-1)
        return x_out.reshape(batch_size, -1) + self.bias

class MinimalKANLayer(nn.Module):
    """Kolmogorov-Arnold Network (KAN) Layer."""
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features, self.out_features, self.grid_size = in_features, out_features, grid_size
        self.spline_weights = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        
    def forward(self, x):
        x_expanded = x.unsqueeze(1).unsqueeze(-1).expand(-1, self.out_features, -1, self.grid_size)
        return torch.sum(x_expanded * self.spline_weights, dim=(2, 3))

# =============================================================================
# 2. CLASSIFIER HEADS
# =============================================================================

class ViT_BCT_MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.backbone.heads = nn.Identity() 
        self.bct = BlockCirculantTransition(in_features=768, out_features=128)
        self.mlp_head = nn.Sequential(nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.mlp_head(self.bct(self.backbone(x)))

class ViT_BCT_KAN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.backbone.heads = nn.Identity()
        self.bct = BlockCirculantTransition(in_features=768, out_features=128)
        self.kan_head = KANLayer(in_features=128, out_features=num_classes)

    def forward(self, x):
        return self.kan_head(self.bct(self.backbone(x)))

# =============================================================================
# 3. TRAINING & UTILS
# =============================================================================

def get_loaders(dataset_name, batch_size=64):
    print(f">> Initializing dataloaders for {dataset_name}...")
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if dataset_name == "MNIST":
        transform_list.insert(0, transforms.Grayscale(num_output_channels=3))
    
    transform = transforms.Compose(transform_list)
    
    if dataset_name == "CIFAR-10":
        train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "MNIST":
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "EuroSAT":
        full_ds = datasets.EuroSAT(root='./data', download=True, transform=transform)
        train_size = int(0.8 * len(full_ds))
        test_size = len(full_ds) - train_size
        train_ds, test_ds = random_split(full_ds, [train_size, test_size])
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, 10

def train_and_evaluate(model, train_loader, test_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.bct.parameters()) + 
                           (list(model.kan_head.parameters()) if hasattr(model, 'kan_head') else list(model.mlp_head.parameters())), 
                           lr=1e-3)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / total_train
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        start_time = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        end_time = time.time()
        val_loss = val_loss / len(all_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        latency_ms = ((end_time - start_time) / len(all_labels)) * 1000
        
        print(f"    Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | F1: {f1:.4f}")
        
    return val_acc, precision, recall, f1, latency_ms, param_count

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not found. Exiting.")
        sys.exit(1)
    
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda")

    parser = argparse.ArgumentParser(description="ViT Classifier Head Ablation Study")
    parser.add_argument("--dataset", type=str, default="CIFAR-10", choices=["CIFAR-10", "MNIST", "EuroSAT"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    print(f"=== [{args.dataset}] TOPOLOGICAL VULNERABILITY ABLATION STUDY ===")
    train_loader, test_loader, num_classes = get_loaders(args.dataset, args.batch_size)
    
    kan_metrics, mlp_metrics = [], []
    
    # 1. ViT + BCT + KAN
    for run in range(args.runs):
        print(f"\n[{args.dataset}] Training ViT + BCT + KAN (Run {run+1}/{args.runs})")
        torch.manual_seed(run)
        model = ViT_BCT_KAN(num_classes)
        acc, prec, rec, f1, lat, p_count = train_and_evaluate(model, train_loader, test_loader, DEVICE, args.epochs)
        kan_metrics.append([acc, prec, rec, f1, lat, p_count])
        del model; torch.cuda.empty_cache()

    # 2. ViT + BCT + MLP
    for run in range(args.runs):
        print(f"\n[{args.dataset}] Training ViT + BCT + MLP (Run {run+1}/{args.runs})")
        torch.manual_seed(run)
        model = ViT_BCT_MLP(num_classes)
        acc, prec, rec, f1, lat, p_count = train_and_evaluate(model, train_loader, test_loader, DEVICE, args.epochs)
        mlp_metrics.append([acc, prec, rec, f1, lat, p_count])
        del model; torch.cuda.empty_cache()

    kan_arr = np.array(kan_metrics)
    mlp_arr = np.array(mlp_metrics)
    
    print("\n" + "="*100)
    print(f"[{args.dataset}] FINAL EVALUATION RESULTS")
    print(f"{'Model':<15} | {'Acc (%)':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Latency(ms)':<12} | {'Params'}")
    print("-" * 100)
    print(f"{'ViT+BCT+KAN':<15} | {np.mean(kan_arr[:,0]):.2f}±{np.std(kan_arr[:,0]):.2f} | {np.mean(kan_arr[:,3]):.4f}   | {np.mean(kan_arr[:,1]):.4f}   | {np.mean(kan_arr[:,2]):.4f}   | {np.mean(kan_arr[:,4]):.2f} ms     | {kan_arr[0,5]:.0f}")
    print(f"{'ViT+BCT+MLP':<15} | {np.mean(mlp_arr[:,0]):.2f}±{np.std(mlp_arr[:,0]):.2f} | {np.mean(mlp_arr[:,3]):.4f}   | {np.mean(mlp_arr[:,1]):.4f}   | {np.mean(mlp_arr[:,2]):.4f}   | {np.mean(mlp_arr[:,4]):.2f} ms     | {mlp_arr[0,5]:.0f}")
    
    t_stat, p_value = stats.ttest_ind(kan_arr[:,0], mlp_arr[:,0])
    print("-" * 100)
    print(f">> Statistical Validation: T-Statistic = {t_stat:.4f} | P-Value = {p_value:.4e}")
    if p_value < 0.05:
        print(">> Conclusion: The diagnostic disparity between KAN and MLP is statistically SIGNIFICANT (p < 0.05).")
    else:
        print(">> Conclusion: The difference between KAN and MLP is statistically INSIGNIFICANT.")
    print("="*100)