# -*- coding: utf-8 -*-
"""
Step 1 - Extract frozen backbone features once and cache them to disk.

Protocol (identical to the original frozen-ViT benchmarks): ImageNet-pretrained
backbone, 224x224 resize, ImageNet normalisation, MNIST converted to 3 channels,
no augmentation. torchvision ViT-B/16 has dropout=0, so features computed in
eval mode are identical to those seen during the original (train-mode) runs.

Usage (Windows, dfm_kan_env, from Hasan\\src\\revision):
    python extract_features.py                       # all datasets x both backbones
    python extract_features.py --datasets ISIC2019 --backbones vit_b_16
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet50, ResNet50_Weights

import common


def build_backbone(name):
    if name == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # same weights as DEFAULT in torchvision 0.20
        m.heads = nn.Identity()
    elif name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
    else:
        raise ValueError(name)
    return m.eval()


def build_transform(dataset):
    t = [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    if dataset == "MNIST":
        t.insert(0, transforms.Grayscale(num_output_channels=3))
    return transforms.Compose(t)


@torch.no_grad()
def extract(model, ds, device, batch_size, workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=common.DATASETS)
    ap.add_argument("--backbones", nargs="+", default=common.BACKBONES)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(common.FEAT_DIR, exist_ok=True)
    print(f">> device={device} torch={torch.__version__} data_root={os.path.abspath(common.DATA_ROOT)}")

    for bb in args.backbones:
        model = build_backbone(bb).to(device)
        for dsname in args.datasets:
            splits, n_cls = common.get_dataset_and_splits(dsname, build_transform(dsname))
            for split, ds in splits.items():
                out = common.feature_path(dsname, bb, split)
                if os.path.exists(out) and not args.overwrite:
                    print(f"   skip (exists) {out}")
                    continue
                t0 = time.time()
                X, y = extract(model, ds, device, args.batch_size, args.workers)
                np.savez_compressed(out, X=X.astype(np.float32), y=y.astype(np.int64), num_classes=n_cls)
                print(f"   {dsname:9s} {bb:9s} {split:5s} X={X.shape} "
                      f"mean|x|={np.abs(X).mean():.3f} {time.time() - t0:6.1f}s -> {os.path.basename(out)}")
        del model
        torch.cuda.empty_cache()
    print(">> done")


if __name__ == "__main__":
    main()
