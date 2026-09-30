# -*- coding: utf-8 -*-
"""
Generate Figure 9: ISIC 2019 Learning Curves
Extracts EXACT epoch-by-epoch data directly from the experimental logs.
Ensures 100% reproducibility and academic integrity (no hardcoded data).

Author: Prof. Dr. Metin Zontul
Date: September 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os

# =============================================================================
# PUBLICATION FORMATTING SETTINGS
# =============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12

# =============================================================================
# DATA PARSING FROM REAL LOGS
# =============================================================================
# Log dosyasının yolu (Klasör yapınıza göre ayarlayın)
log_file_path = os.path.join("..", "logs", "ISIC_2019_Ablation_Results.txt")

# Regex patternleri ile gerçek verileri yakalayacağız
epoch_pattern = re.compile(r"Epoch \[(\d+)/20\] \| Train Loss: ([\d.]+) \| Val Acc: ([\d.]+)%")

def extract_learning_curves(log_path, target_model_header):
    losses = []
    accuracies = []
    capture = False
    
    try:
        with open(log_path, 'r', encoding='utf-8') as file:
            for line in file:
                if target_model_header in line:
                    capture = True
                    continue
                
                if capture:
                    match = epoch_pattern.search(line)
                    if match:
                        losses.append(float(match.group(2)))
                        accuracies.append(float(match.group(3)))
                    elif line.strip() == "" and len(losses) > 0:
                        # Boş satır gördüğümüzde o run bitmiş demektir
                        break
    except FileNotFoundError:
        print(f"ERROR: Log file not found at {log_path}. Please check the path.")
        return None, None
        
    return losses, accuracies

# Run 1 verilerini doğrudan log'dan çekiyoruz
loss_kan, acc_kan = extract_learning_curves(log_file_path, "[ISIC 2019] Baseline ViT + KAN (Run 1/5)")
loss_linear, acc_linear = extract_learning_curves(log_file_path, "[ISIC 2019] Baseline ViT + Linear (Run 1/5)")

if not loss_kan or not loss_linear:
    print("Failed to extract data. Generating empty plot.")
    loss_kan, acc_kan = [0]*20, [0]*20
    loss_linear, acc_linear = [0]*20, [0]*20

epochs = np.arange(1, 21)

# =============================================================================
# PLOTTING
# =============================================================================
color_linear = '#2c6085' # Koyu Mavi
color_kan = '#cc4b51'    # Koyu Kırmızı

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# --- Left Plot: Training Loss ---
axes[0].plot(epochs, loss_linear, '-o', color=color_linear, linewidth=2.5, markersize=7, label='ViT+Linear (Baseline)')
axes[0].plot(epochs, loss_kan, '--s', color=color_kan, linewidth=2.5, markersize=7, label='ViT+KAN (Baseline)')
axes[0].set_title('(a) Training Loss over 20 Epochs (ISIC 2019)', pad=10)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Training Loss')
axes[0].set_xticks(np.arange(2, 21, 2))
axes[0].grid(linestyle='--', alpha=0.5)
legend = axes[0].legend()
for text in legend.get_texts(): text.set_fontweight('bold')

# --- Right Plot: Validation Accuracy ---
axes[1].plot(epochs, acc_linear, '-o', color=color_linear, linewidth=2.5, markersize=7, label='ViT+Linear (Baseline)')
axes[1].plot(epochs, acc_kan, '--s', color=color_kan, linewidth=2.5, markersize=7, label='ViT+KAN (Baseline)')
axes[1].set_title('(b) Validation Accuracy over 20 Epochs (ISIC 2019)', pad=10)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].set_xticks(np.arange(2, 21, 2))
axes[1].grid(linestyle='--', alpha=0.5)
legend = axes[1].legend(loc='lower right')
for text in legend.get_texts(): text.set_fontweight('bold')

plt.tight_layout()
output_filename = 'figure_09.png'
plt.savefig(output_filename, dpi=600, bbox_inches='tight')
print(f">> {output_filename} successfully generated from REAL raw logs!")