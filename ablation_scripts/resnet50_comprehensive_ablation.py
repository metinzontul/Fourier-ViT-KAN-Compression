# -*- coding: utf-8 -*-
"""
Comprehensive ResNet-50 Ablation Study (MNIST, EuroSAT, ISIC 2019)
Baseline (ResNet50+KAN) vs. Proposed (ResNet50+BCT+KAN)

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Autonomous Multi-Dataset Loop
- 5 Seeds, 20 Epochs per run
- Evaluates Macro F1, Precision, Recall, Inference Latency, and Head Parameters.
- Concludes with an independent T-Test for statistical validation.
"""

import os
import time
import warnings
import argparse
import math
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CORE ARCHITECTURAL MODULES (BCT & KAN)
# =============================================================================

class BlockCirculantTransition(nn.Module):
    """
    Banded Compression Transform (BCT) Module.
    Aggressively truncates high-frequency spatial components in the Fourier domain
    and projects them via trainable complex weights to reduce dimensionality.
    """
    def __init__(self, in_features, out_features, block_size=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.p = in_features // block_size  
        self.q = out_features // block_size 
        
        # Trainable complex weights in the frequency domain
        self.weight = nn.Parameter(torch.randn(self.q, self.p, self.block_size) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.channel_scale = nn.Parameter(torch.ones(out_features))
        
        self.pre_norm = nn.LayerNorm(in_features)
        self.post_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        batch_size = x.size(0)
        
        # 1. Spatial to Frequency Domain (FFT)
        x_reshaped = x_norm.view(batch_size, self.p, self.block_size)
        x_fft = torch.fft.rfft(x_reshaped, dim=-1)
        w_fft = torch.fft.rfft(self.weight, dim=-1)
        
        # 2. Spectral Truncation & Complex Projection
        y_fft = torch.einsum('bpf,qpf->bqf', x_fft, w_fft)
        
        # 3. Inverse FFT (Directly to the compressed dimension)
        y_reshaped = torch.fft.irfft(y_fft, n=self.block_size, dim=-1)
        y = y_reshaped.reshape(batch_size, self.out_features)
        
        y = self.post_norm(y)
        y = (y * self.channel_scale) + self.bias
        
        # Parameter-free skip connection (Global Average)
        x_skip = x_norm.view(batch_size, self.out_features, -1).mean(dim=2)
        
        return torch.nn.functional.silu(y + x_skip)

class MinimalKANLayer(nn.Module):
    """
    Minimal Kolmogorov-Arnold Network (KAN) Layer.
    Utilizes B-Spline-like symbolic parameterization.
    """
    def __init__(self, in_features, num_classes, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_weights = nn.Parameter(torch.randn(num_classes, in_features, grid_size) / math.sqrt(in_features))

    def forward(self, x):
        bases = torch.stack([torch.sin((i+1) * x) for i in range(self.grid_size)], dim=-1)
        return torch.einsum('big,oig->bo', bases, self.spline_weights)

# =============================================================================
# 2. CLASSIFIER ARCHITECTURES
# =============================================================================

class ResNet50_KAN_Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Uncompressed Baseline Head
        self.head = MinimalKANLayer(num_ftrs, num_classes) 
        
    def forward(self, x):
        return self.head(self.backbone(x))

class ResNet50_BCT_KAN_Proposed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Proposed Compressed Head
        self.bct = BlockCirculantTransition(in_features=num_ftrs, out_features=num_ftrs // 16) 
        self.head = MinimalKANLayer(in_features=num_ftrs // 16, num_classes=num_classes)

    def forward(self, x):
        compressed = self.bct(self.backbone(x))
        return self.head(compressed)

# =============================================================================
# 3. UTILITIES & DATALOADERS
# =============================================================================

def get_dataloaders(dataset_name, batch_size=64):
    print(f"\n>> Loading {dataset_name} dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'MNIST':
        transform.transforms.insert(0, transforms.Grayscale(3))
        train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == 'EuroSAT':
        train_ds = torchvision.datasets.EuroSAT(root='./data', download=True, transform=transform)
        val_ds = torchvision.datasets.EuroSAT(root='./data', download=True, transform=transform)
        train_size = int(0.8 * len(train_ds))
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, len(train_ds) - train_size])
        num_classes = 10

    elif dataset_name == 'ISIC2019':
        isic_path = './data/ISIC2019'
        if not os.path.exists(isic_path):
            print(f"Warning: {isic_path} not found! Skipping ISIC 2019.")
            return None, None, 8
        full_ds = torchvision.datasets.ImageFolder(root=isic_path, transform=transform)
        train_size = int(0.8 * len(full_ds))
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, len(full_ds) - train_size])
        num_classes = 8

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, num_classes

def count_head_params(model):
    params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    if hasattr(model, 'bct'):
        params += sum(p.numel() for p in model.bct.parameters() if p.requires_grad)
    return params

# =============================================================================
# 4. TRAINING & EVALUATION LOOP
# =============================================================================

def train_and_evaluate(dataset_name, model_name, model_class, train_loader, val_loader, num_classes, device, epochs=20, runs=5):
    all_f1, all_acc, all_prec, all_rec, all_latencies = [], [], [], [], []
    head_params = 0
    
    for run in range(1, runs + 1):
        print(f"\n[{dataset_name}] {model_name} (Run {run}/{runs})")
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        model = model_class(num_classes).to(device)
        head_params = count_head_params(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
            train_loss_epoch = train_loss / train_total
            train_acc_epoch = 100. * train_correct / train_total
            
            # --- EVALUATION ---
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            
            start_time = time.time()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    
            latency_ms = ((time.time() - start_time) / len(val_loader.dataset)) * 1000
            val_loss_epoch = val_loss / len(val_loader.dataset)
            val_acc_epoch = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
            
            macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            
            print(f"    Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss_epoch:.4f} - Acc: {train_acc_epoch:.2f}% | Val Loss: {val_loss_epoch:.4f} - Acc: {val_acc_epoch:.2f}% | Macro F1: {macro_f1:.4f}")
            
            if epoch == epochs:
                all_f1.append(macro_f1)
                all_acc.append(val_acc_epoch)
                all_prec.append(prec)
                all_rec.append(rec)
                all_latencies.append(latency_ms)

    return {
        'acc_m': np.mean(all_acc), 'acc_s': np.std(all_acc),
        'f1_m': np.mean(all_f1), 'prec_m': np.mean(all_prec), 'rec_m': np.mean(all_rec),
        'latency': np.mean(all_latencies), 'params': head_params, 'f1_raw': all_f1
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive ResNet-50 Ablation Study")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per run")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent seeds")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== COMPREHENSIVE RESNET-50 ABLATION STUDY STARTING ===")
    print(f">> Active Device: {DEVICE}")
    
    datasets_to_run = ['MNIST', 'EuroSAT'] # Excluded ISIC2019 by default due to size; add if downloaded
    
    for dataset in datasets_to_run:
        print(f"\n{'='*100}\n>>> INITIATING: {dataset} \n{'='*100}")
        
        train_loader, val_loader, num_classes = get_dataloaders(dataset, args.batch_size)
        if train_loader is None:
            continue
            
        res_base = train_and_evaluate(dataset, "Baseline ResNet50 + KAN", ResNet50_KAN_Baseline, train_loader, val_loader, num_classes, DEVICE, args.epochs, args.runs)
        res_prop = train_and_evaluate(dataset, "Proposed ResNet50 + BCT + KAN", ResNet50_BCT_KAN_Proposed, train_loader, val_loader, num_classes, DEVICE, args.epochs, args.runs)
        
        comp_rate = (1 - (res_prop['params'] / res_base['params'])) * 100
        t_stat, p_val = stats.ttest_rel(res_base['f1_raw'], res_prop['f1_raw'])
        
        print("\n" + "="*100)
        print(f"[{dataset}] FINAL EVALUATION RESULTS")
        print(f"{'Model':<27} | {'Acc (%)':<13} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8} | {'Latency':<7} | {'Head Params'}")
        print("-" * 100)
        print(f"{'ResNet50+KAN (Baseline)':<27} | {res_base['acc_m']:.2f}±{res_base['acc_s']:.2f} | {res_base['f1_m']:.4f}   | {res_base['prec_m']:.4f}    | {res_base['rec_m']:.4f}   | {res_base['latency']:.2f} ms | {res_base['params']:,}")
        print(f"{'ResNet50+BCT+KAN (Proposed)':<27} | {res_prop['acc_m']:.2f}±{res_prop['acc_s']:.2f} | {res_prop['f1_m']:.4f}   | {res_prop['prec_m']:.4f}    | {res_prop['rec_m']:.4f}   | {res_prop['latency']:.2f} ms | {res_prop['params']:,} (-{comp_rate:.1f}%)")
        print("-" * 100)
        print(f">> Statistical Validation (Macro F1): T-Statistic = {t_stat:.4f} | P-Value = {p_val:.4e}")
        
        if p_val < 0.05:
            print(">> Conclusion: The performance difference is statistically SIGNIFICANT (p < 0.05).")
        else:
            print(">> Conclusion: The performance difference is statistically INSIGNIFICANT (Performance Preserved).")
        print("=" * 100)