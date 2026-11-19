# -*- coding: utf-8 -*-
"""
Shared paths, dataset builders and fixed splits for the revision experiments.

Expected layout (unchanged from the original experiments):
    <DATA_ROOT>/cifar-10-batches-py/
    <DATA_ROOT>/MNIST/raw/
    <DATA_ROOT>/eurosat/2750/
    <DATA_ROOT>/ISIC2019/train/<class>/*.jpg , val/<class>/*.jpg
    <DATA_ROOT>/ISIC2019/isic2019_lesion_grouped_split.csv

All splits are written to revision/splits/ the first time and re-used afterwards,
so every experiment in the revision sees exactly the same partitions.
"""
import os
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("REV_DATA_ROOT", os.path.join(HERE, "..", "data"))
SPLIT_DIR = os.path.join(HERE, "splits")
FEAT_DIR = os.environ.get("REV_FEAT_DIR", os.path.join(HERE, "features"))
RESULT_DIR = os.environ.get("REV_RESULT_DIR", os.path.join(HERE, "results"))

DATASETS = ["MNIST", "CIFAR10", "EuroSAT", "ISIC2019"]
BACKBONES = ["vit_b_16", "resnet50"]
SPLIT_SEED = 42

ISIC_CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]  # ImageFolder (alphabetical) order


def _stratified_three_way(labels, val_frac, test_frac, seed=SPLIT_SEED):
    """Stratified split of indices into train/val/test (test_frac may be 0)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    if test_frac > 0:
        idx_dev, idx_test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=labels)
    else:
        idx_dev, idx_test = idx, np.array([], dtype=int)
    rel_val = val_frac / (1.0 - test_frac)
    idx_tr, idx_val = train_test_split(idx_dev, test_size=rel_val, random_state=seed, stratify=labels[idx_dev])
    return np.sort(idx_tr), np.sort(idx_val), np.sort(idx_test)


def get_dataset_and_splits(name, transform):
    """
    Returns (dataset_dict, num_classes) where dataset_dict maps 'train'/'val'/'test'
    to torch Datasets. Split files are created once under revision/splits/.
    """
    import torch
    from torch.utils.data import Subset
    from torchvision import datasets
    os.makedirs(SPLIT_DIR, exist_ok=True)

    if name in ("MNIST", "CIFAR10"):
        cls = datasets.MNIST if name == "MNIST" else datasets.CIFAR10
        train_full = cls(root=DATA_ROOT, train=True, download=False, transform=transform)
        test = cls(root=DATA_ROOT, train=False, download=False, transform=transform)
        labels = np.array(train_full.targets)
        f = os.path.join(SPLIT_DIR, f"{name}_val_indices.csv")
        if not os.path.exists(f):
            _, idx_val, _ = _stratified_three_way(labels, val_frac=0.10, test_frac=0.0)
            pd.DataFrame({"train_index": idx_val}).to_csv(f, index=False)
        idx_val = pd.read_csv(f)["train_index"].values
        idx_tr = np.setdiff1d(np.arange(len(labels)), idx_val)
        return {"train": Subset(train_full, idx_tr), "val": Subset(train_full, idx_val), "test": test}, 10

    if name == "EuroSAT":
        full = datasets.EuroSAT(root=DATA_ROOT, download=False, transform=transform)
        labels = np.array(full.targets)
        f = os.path.join(SPLIT_DIR, "EuroSAT_split.csv")
        if not os.path.exists(f):
            tr, va, te = _stratified_three_way(labels, val_frac=0.10, test_frac=0.20)
            df = pd.DataFrame({"index": np.arange(len(labels)), "label": labels, "split": ""})
            df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
            df.to_csv(f, index=False)
        df = pd.read_csv(f)
        return {s: Subset(full, df.index[df.split == s].values) for s in ("train", "val", "test")}, 10

    if name == "ISIC2019":
        return _isic_splits(transform), len(ISIC_CLASSES)

    raise ValueError(name)


class ISICDataset:
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform
        self.targets = self.labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def isic_path_index():
    """Map image id -> file path, searching the existing train/ and val/ folders."""
    root = os.path.join(DATA_ROOT, "ISIC2019")
    index = {}
    for p in glob.glob(os.path.join(root, "*", "*", "*.jpg")):
        index[os.path.splitext(os.path.basename(p))[0]] = p
    return index


def _isic_splits(transform):
    csv = os.path.join(DATA_ROOT, "ISIC2019", "isic2019_lesion_grouped_split.csv")
    if not os.path.exists(csv):
        csv = os.path.join(SPLIT_DIR, "isic2019_lesion_grouped_split.csv")
    df = pd.read_csv(csv)
    index = isic_path_index()
    missing = [i for i in df.image if i not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ISIC images not found under {DATA_ROOT}/ISIC2019 (e.g. {missing[:3]})")
    lab = {c: k for k, c in enumerate(ISIC_CLASSES)}
    out = {}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        out[s] = ISICDataset([index[i] for i in d.image], [lab[l] for l in d.label], transform)
    return out


def feature_path(dataset, backbone, split):
    return os.path.join(FEAT_DIR, f"{dataset}__{backbone}__{split}.npz")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def tee_console(name):
    """Mirror everything printed to the console into results/logs/<name>_<timestamp>.txt."""
    import sys
    import time
    log_dir = os.path.join(RESULT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    f = open(path, "a", encoding="utf-8")
    f.write("$ python " + " ".join(sys.argv) + "\n")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print(f">> console log: {path}")
    return path
