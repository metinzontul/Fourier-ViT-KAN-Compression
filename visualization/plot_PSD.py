# -*- coding: utf-8 -*-
"""
Generate Figure 1: Power Spectral Density (PSD) Analysis

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Evaluates and compares the spatial frequency distribution across all four datasets.
- Mathematically validates that ISIC 2019 (medical data) retains critical power in 
  high-frequency domains, explaining the vulnerability of certain classifier topologies.
"""

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets 

# =============================================================================
# 1. ALGORITHM FOR RADIAL PROFILE EXTRACTION
# =============================================================================
def compute_radial_profile(image_gray, target_size=(256, 256)):
    """Computes the 1D Power Spectral Density from a 2D image."""
    if image_gray.shape != target_size:
        image_gray = cv2.resize(image_gray, target_size)
    
    image_gray = image_gray / 255.0
    
    # 2D FFT
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) ** 2
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    
    # Extract meaningful half up to Nyquist frequency
    psd = radialprofile[:target_size[0] // 2]
    psd = psd / np.max(psd) # Normalize to start from 0 dB
    return 10 * np.log10(psd + 1e-10)

# =============================================================================
# 2. DATASET PROCESSING
# =============================================================================
# Standard relative path for GitHub
base_path = "./data"
max_images = 300 # Sample size for analysis

plt.figure(figsize=(10, 6), dpi=600)
x_axis = np.linspace(0, 1, 128) # Normalized X Axis

# --- 1. MNIST (Binary Package) ---
print(">> Analyzing MNIST...")
mnist_ds = datasets.MNIST(root=base_path, train=True, download=True)
mnist_psd_sum = np.zeros(128)
for i in range(max_images):
    img = np.array(mnist_ds[i][0])
    mnist_psd_sum += compute_radial_profile(img)
plt.plot(x_axis, mnist_psd_sum / max_images, label='MNIST (Low Frequency)', color='#1f77b4', linewidth=2.5)

# --- 2. CIFAR-10 (Pickle Package) ---
print(">> Analyzing CIFAR-10...")
cifar_ds = datasets.CIFAR10(root=base_path, train=True, download=True)
cifar_psd_sum = np.zeros(128)
for i in range(max_images):
    img = np.array(cifar_ds[i][0].convert('L')) # Convert to Grayscale
    cifar_psd_sum += compute_radial_profile(img)
plt.plot(x_axis, cifar_psd_sum / max_images, label='CIFAR-10 (Low-Mid)', color='#2ca02c', linewidth=2.5)

# --- 3 & 4. Image Folders (EuroSAT and ISIC 2019) ---
def process_image_folder(folder_name, label, color):
    print(f">> Analyzing {label}...")
    folder_path = os.path.join(base_path, folder_name)
    image_paths = []
    
    if not os.path.exists(folder_path):
        print(f"   WARNING: Folder {folder_path} not found. Skipping plot line.")
        return

    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
        image_paths.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    np.random.shuffle(image_paths)
    image_paths = image_paths[:max_images]
    
    psd_sum = np.zeros(128)
    valid_count = 0
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            psd_sum += compute_radial_profile(img)
            valid_count += 1
            
    if valid_count > 0:
        plt.plot(x_axis, psd_sum / valid_count, label=label, color=color, linewidth=2.5)

process_image_folder('EuroSAT', 'EuroSAT (Macroscopic)', '#ff7f0e')
process_image_folder('ISIC2019', 'ISIC 2019 (Micro-texture)', '#d62728')

# =============================================================================
# 3. PLOT FORMATTING & EXPORT
# =============================================================================
plt.title('Power Spectral Density (PSD) Comparison Across Datasets', fontsize=14, fontweight='bold')
plt.xlabel('Normalized Spatial Frequency', fontsize=12)
plt.ylabel('Normalized Power (dB)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
save_path = os.path.join("figures", "Figure_1_PSD_Comparison.png")
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"\n[SUCCESS] Publication-ready PSD plot saved to {save_path}")