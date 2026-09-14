# -*- coding: utf-8 -*-
"""
Medical Image Ablation Study: ViT + BCT + KAN on ISIC 2019

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Evaluates the topological vulnerability of the KAN classifier head under 
  extreme frequency-domain truncation (BCT) on the ISIC 2019 dataset.
- Automatically handles dataset downloading, extraction, and class balancing.
- Computes Macro F1, Precision, Recall, and Inference Latency across 5 seeds.
"""

import os
import sys
import time
import math
import random
import urllib.request
import zipfile
import shutil
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.datasets import ImageFolder

# =============================================================================
# 1. AUTO-DATASET DOWNLOADER & ORGANIZER 
# =============================================================================

def download_progress_hook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(f"\r... Downloading: {percent}% | {progress_size / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB | {speed} KB/s")
    sys.stdout.flush()

def setup_isic2019(base_dir='./data/ISIC2019'):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f">> ISIC 2019 dataset is already prepared at '{base_dir}'. Skipping download.")
        
        # Cleanup: Remove empty directories to prevent PyTorch DataLoader crashes
        for d in [train_dir, val_dir]:
            if os.path.exists(d):
                for cls_name in os.listdir(d):
                    cls_path = os.path.join(d, cls_name)
                    if os.path.isdir(cls_path) and not os.listdir(cls_path):
                        os.rmdir(cls_path)
        return train_dir, val_dir

    print(f">> Preparing ISIC 2019 Dataset automatically at '{base_dir}'...")
    os.makedirs(base_dir, exist_ok=True)
    img_zip_path = os.path.join(base_dir, 'ISIC_2019_Training_Input.zip')
    csv_path = os.path.join(base_dir, 'ISIC_2019_Training_GroundTruth.csv')
    img_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
    csv_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
    
    if not os.path.exists(csv_path):
        urllib.request.urlretrieve(csv_url, csv_path)
    if not os.path.exists(img_zip_path):
        urllib.request.urlretrieve(img_url, img_zip_path, reporthook=download_progress_hook)

    extraction_dir = os.path.join(base_dir, 'ISIC_2019_Training_Input')
    if not os.path.exists(extraction_dir):
        with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    df = pd.read_csv(csv_path)
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    img_list, label_list = [], []
    for _, row in df.iterrows():
        img_name = row['image'] + '.jpg'
        for cls in classes:
            if row[cls] == 1.0:
                img_list.append(img_name)
                label_list.append(cls)
                break

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_list, label_list, test_size=0.2, random_state=42, stratify=label_list
    )

    def move_files(img_names, labels, target_dir):
        for img, lbl in zip(img_names, labels):
            src = os.path.join(extraction_dir, img)
            dst = os.path.join(target_dir, lbl, img)
            if os.path.exists(src):
                shutil.move(src, dst)

    move_files(train_imgs, train_lbls, train_dir)
    move_files(val_imgs, val_lbls, val_dir)

    shutil.rmtree(extraction_dir)
    os.remove(img_zip_path)
    
    for d in [train_dir, val_dir]:
        for cls_name in os.listdir(d):
            cls_path = os.path.join(d, cls_name)
            if os.path.isdir(cls_path) and not os.listdir(cls_path):
                os.rmdir(cls_path)
                
    return train_dir, val_dir

# =============================================================================
# 2. INTEGRATED CUSTOM MODULES (BCT & KAN)
# =============================================================================

class BCT(nn.Module):
    """
    Banded Compression Transform (BCT) Layer.
    Aggressively truncates high-frequency spatial components in the Fourier domain.
    """
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

class KAN(nn.Module):
    """
    Minimal Kolmogorov-Arnold Network (KAN) Layer.
    Utilizes B-Splines instead of static linear weights.
    """
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
            left_term = (x - self.grid[:, :, :-(k + 1)]) / (self.grid[:, :, k:-1] - self.grid[:, :, :-(k + 1)] + 1e-8) * bases[:, :, :-1]
            right_term = (self.grid[:, :, k + 1:] - x) / (self.grid[:, :, k + 1:] - self.grid[:, :, 1:-k] + 1e-8) * bases[:, :, 1:]
            bases = left_term + right_term
        return bases

    def forward(self, x):
        base_output = nn.functional.linear(nn.functional.silu(x), self.base_weight)
        spline_bases = self.b_splines(x)
        spline_output = torch.einsum('bie,oie->bo', spline_bases, self.spline_weight)
        return base_output + spline_output

# =============================================================================
# 3. MODEL ARCHITECTURES
# =============================================================================

class ViT_Baseline_KAN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        self.head = KAN(in_features=768, out_features=num_classes)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class ViT_Proposed_BCT_KAN(nn.Module):
    def __init__(self, num_classes, compressed_dim=128):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        self.bct = BCT(in_features=768, out_features=compressed_dim) 
        self.kan = KAN(in_features=compressed_dim, out_features=num_classes)
        
    def forward(self, x):
        return self.kan(self.bct(self.backbone(x)))

# =============================================================================
# 4. TRAINING & UTILS
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(dataset):
    class_counts = np.bincount(dataset.targets)
    total_samples = len(dataset)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def measure_latency(model, device, input_size=(1, 3, 224, 224), runs=50):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input) # Warmup
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs): _ = model(dummy_input)
    return ((time.time() - start_time) / runs) * 1000

def count_head_parameters(model, is_proposed=False):
    if is_proposed:
        return sum(p.numel() for p in model.bct.parameters() if p.requires_grad) + \
               sum(p.numel() for p in model.kan.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.head.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, class_weights, device, epochs=20):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    best_acc, best_f1, best_prec, best_rec = 0.0, 0.0, 0.0, 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_acc = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')
        prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        if f1 > best_f1:
            best_acc, best_f1, best_prec, best_rec = val_acc, f1, prec, rec
            
        print(f"    Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Macro F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
    return best_acc, best_f1, best_prec, best_rec

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=== [ISIC 2019] COMPREHENSIVE MEDICAL ABLATION STUDY STARTING ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 43, 44, 45, 46]
    epochs = 20
    
    train_dir, val_dir = setup_isic2019()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    trainset = ImageFolder(root=train_dir, transform=transform)
    valset = ImageFolder(root=val_dir, transform=val_transform)
    num_classes = len(trainset.classes)
    
    class_weights = compute_class_weights(trainset)
    print(f">> Computed Class Weights for {num_classes} classes: {class_weights}")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)

    results = {"Baseline": [], "Proposed": []}
    latencies = {"Baseline": [], "Proposed": []}
    
    for i, seed in enumerate(seeds):
        print(f"\n[ISIC 2019] Baseline ViT + KAN (Run {i+1}/{len(seeds)})")
        set_seed(seed)
        model = ViT_Baseline_KAN(num_classes).to(device)
        acc, f1, prec, rec = train_model(model, trainloader, valloader, class_weights, device, epochs)
        results["Baseline"].append((acc, f1, prec, rec))
        latencies["Baseline"].append(measure_latency(model, device))
        base_params = count_head_parameters(model, is_proposed=False)

    for i, enumerate_seed in enumerate(seeds):
        print(f"\n[ISIC 2019] Proposed ViT + BCT + KAN (Run {i+1}/{len(seeds)})")
        set_seed(enumerate_seed)
        model = ViT_Proposed_BCT_KAN(num_classes, compressed_dim=128).to(device)
        acc, f1, prec, rec = train_model(model, trainloader, valloader, class_weights, device, epochs)
        results["Proposed"].append((acc, f1, prec, rec))
        latencies["Proposed"].append(measure_latency(model, device))
        prop_params = count_head_parameters(model, is_proposed=True)

    base_accs, base_f1s = [x[0] for x in results["Baseline"]], [x[1] for x in results["Baseline"]]
    base_precs, base_recs = [x[2] for x in results["Baseline"]], [x[3] for x in results["Baseline"]]
    
    prop_accs, prop_f1s = [x[0] for x in results["Proposed"]], [x[1] for x in results["Proposed"]]
    prop_precs, prop_recs = [x[2] for x in results["Proposed"]], [x[3] for x in results["Proposed"]]

    t_stat_f1, p_value_f1 = stats.ttest_ind(base_f1s, prop_f1s)
    compression_rate = ((base_params - prop_params) / base_params) * 100
    
    print("\n=========================================================================================================================================")
    print(f"Model                  | Acc (%)       | Macro F1      | Precision     | Recall        | Latency (ms) | Head Params | Compression")
    print("-----------------------------------------------------------------------------------------------------------------------------------------")
    print(f"ViT+KAN (Baseline)     | {np.mean(base_accs):.2f}±{np.std(base_accs):.2f} | {np.mean(base_f1s):.4f}±{np.std(base_f1s):.4f} | {np.mean(base_precs):.4f}±{np.std(base_precs):.4f} | {np.mean(base_recs):.4f}±{np.std(base_recs):.4f} | {np.mean(latencies['Baseline']):.2f} ms     | {base_params:,}     | -")
    print(f"ViT+BCT+KAN (Proposed) | {np.mean(prop_accs):.2f}±{np.std(prop_accs):.2f} | {np.mean(prop_f1s):.4f}±{np.std(prop_f1s):.4f} | {np.mean(prop_precs):.4f}±{np.std(prop_precs):.4f} | {np.mean(prop_recs):.4f}±{np.std(prop_recs):.4f} | {np.mean(latencies['Proposed']):.2f} ms     | {prop_params:,}      | {compression_rate:.1f}%")
    print("-----------------------------------------------------------------------------------------------------------------------------------------")
    print(f">> Statistical Validation (Macro F1): T-Statistic = {t_stat_f1:.4f} | P-Value = {p_value_f1:.4e}")
    print("=========================================================================================================================================")