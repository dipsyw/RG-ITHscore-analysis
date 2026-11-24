from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    "densenet121",
    "efficientnet_b0",
    "googlenet",
    "resnet18",
    "resnet50",
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
]


def create_encoder(name: str, device):
    import torch
    import torch.nn as nn

    if name == "googlenet":
        from torchvision import models

        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True, transform_input=False)
        old = model.conv1.conv
        first = nn.Conv2d(2, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            first.weight.copy_(old.weight[:, :2])
        model.conv1.conv = first
        model.fc = nn.Identity()
    else:
        import timm

        model = timm.create_model(name, pretrained=True, in_chans=2, num_classes=0, global_pool="avg")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def normalize(batch):
    import torch

    mean = torch.tensor([0.485, 0.456], dtype=batch.dtype, device=batch.device).view(1, 2, 1, 1)
    sd = torch.tensor([0.229, 0.224], dtype=batch.dtype, device=batch.device).view(1, 2, 1, 1)
    return (batch - mean) / sd


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen ImageNet-pretrained two-channel imaging representations.")
    parser.add_argument("--inputs", type=Path, required=True, help="N x 2 x 224 x 224 float32 NumPy array")
    parser.add_argument("--ids", type=Path, required=True, help="CSV with one ID column in array order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch

    random.seed(20260604)
    np.random.seed(20260604)
    torch.manual_seed(20260604)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260604)
    values = np.load(args.inputs)
    ids = pd.read_csv(args.ids)
    if list(values.shape[1:]) != [2, 224, 224] or len(values) != len(ids):
        raise ValueError("Inputs must be N x 2 x 224 x 224 and match the ID table")
    if not np.isfinite(values).all():
        raise ValueError("Input array contains non-finite values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(values).float()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        model = create_encoder(name, device)
        batches = []
        with torch.no_grad():
            for start in range(0, len(tensor), args.batch_size):
                embedding = model(normalize(tensor[start : start + args.batch_size].to(device)))
                if isinstance(embedding, (tuple, list)):
                    embedding = embedding[0]
                batches.append(embedding.detach().cpu().reshape(len(embedding), -1).numpy())
        matrix = np.concatenate(batches)
        output = pd.DataFrame(matrix, columns=[f"{name}_feature_{i:04d}" for i in range(matrix.shape[1])])
        output.insert(0, "ID", ids.iloc[:, 0].astype(str).to_numpy())
        output.to_csv(args.output_dir / f"deep_features_{name}.csv", index=False)


if __name__ == "__main__":
    main()
