#!/usr/bin/env python3
"""
Export CompressAI pretrained model weights to a format loadable by the Rust
coldstorage CLI.

Exports all state_dict tensors as individual .npy files in a directory,
along with a manifest JSON mapping parameter names to file paths and shapes.

Usage:
    python3 export_weights.py --model hyperprior --quality 6 --output ./weights/

Requirements:
    pip install compressai torch numpy
"""

import argparse
import json
import os

import numpy as np
import torch

from compressai.zoo import (
    bmshj2018_hyperprior,
    cheng2020_anchor,
    mbt2018,
    mbt2018_mean,
)

MODEL_ZOO = {
    "hyperprior": bmshj2018_hyperprior,
    "mbt2018_mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020": cheng2020_anchor,
}


def export_model(model_name: str, quality: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    model_fn = MODEL_ZOO[model_name]
    model = model_fn(quality=quality, pretrained=True)
    model.eval()
    model.update()

    state_dict = model.state_dict()
    manifest = {}

    for name, tensor in state_dict.items():
        # Convert to numpy
        arr = tensor.detach().cpu().numpy()
        filename = name.replace(".", "_") + ".npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, arr)

        manifest[name] = {
            "file": filename,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "quality": quality,
                "parameters": manifest,
            },
            f,
            indent=2,
        )

    print(f"Exported {len(manifest)} parameters to {output_dir}")
    print(f"Manifest: {manifest_path}")

    # Print parameter names for reference (helps map to Rust module names)
    print("\nParameter mapping reference:")
    for name, info in sorted(manifest.items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        print(f"  {name:<50} [{shape_str}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CompressAI weights for Rust")
    parser.add_argument(
        "--model",
        default="hyperprior",
        choices=list(MODEL_ZOO.keys()),
        help="Model architecture",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=6,
        choices=range(1, 9),
        help="Quality level (1-8)",
    )
    parser.add_argument(
        "--output",
        default="./weights/",
        help="Output directory for exported weights",
    )
    args = parser.parse_args()
    export_model(args.model, args.quality, args.output)
