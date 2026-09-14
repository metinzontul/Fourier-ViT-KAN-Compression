#!/bin/bash

# ==============================================================================
# Script: download_weights.sh
# Author: Prof. Dr. Metin Zontul
# Description: Automatically downloads the pre-trained ViT+KAN and ViT+BCT+KAN 
#              model weights from GitHub Releases (v1.0.0) for the ISIC 2019 
#              topological vulnerability and diagnostic collapse experiments.
# ==============================================================================

echo "======================================================================"
echo "Downloading Official Pre-trained Weights (v1.0.0) for ISIC 2019"
echo "======================================================================"

# Define the base URL for the GitHub Release (Update 'YourUsername' below)
BASE_URL="https://github.com/metinzontul/Fourier-ViT-KAN-Compression/releases/download/v1.0.0"

# Target file names to be downloaded
FILE1="best_baseline_vit_kan.pth"
FILE2="best_proposed_vit_bct_kan.pth"

# Function to securely download a file using wget
download_file() {
    local FILE_NAME=$1
    echo ">> Fetching ${FILE_NAME}..."
    wget -q --show-progress -O "${FILE_NAME}" "${BASE_URL}/${FILE_NAME}"
    
    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${FILE_NAME} downloaded successfully."
    else
        echo "[ERROR] Failed to download ${FILE_NAME}. Please check the URL or your internet connection."
    fi
    echo "----------------------------------------------------------------------"
}

# Verify if wget is installed on the host system
if ! command -v wget &> /dev/null; then
    echo "[ERROR] The 'wget' utility could not be found."
    echo "Please install wget or download the files manually from the GitHub Releases page."
    exit 1
fi

# Execute the download sequences
download_file $FILE1
download_file $FILE2

echo "======================================================================"
echo "All automated downloads are completed."
echo "Note: Please move these .pth files to the root repository directory "
echo "if you intend to execute the SHAP analysis or Confusion Matrix scripts "
echo "using their default relative paths."
echo "======================================================================"