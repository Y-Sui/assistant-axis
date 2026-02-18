#!/usr/bin/env python3
"""
Compute degeneration axes from per-category vectors.

For each category:
  axis = mean(good) - mean(degen)

Also saves an aggregate axis (mean across categories).
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_vector(vector_file: Path) -> dict:
    return torch.load(vector_file, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(description="Compute degeneration axes from vectors")
    parser.add_argument("--vectors_dir", type=str, required=True, help="Directory with vector .pt files")
    parser.add_argument("--output", type=str, required=True, help="Output axes .pt file path")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vector_files = sorted(vectors_dir.glob("*.pt"))
    print(f"Found {len(vector_files)} vector files")

    axes = {}
    axis_list = []

    for vec_file in tqdm(vector_files, desc="Loading vectors"):
        data = load_vector(vec_file)
        category = data.get("category", vec_file.stem)
        good = data.get("good")
        degen = data.get("degen")
        if good is None or degen is None:
            print(f"Warning: missing good/degen vectors for {category}")
            continue

        axis = good - degen
        axes[category] = axis
        axis_list.append(axis)

    if not axes:
        print("Error: no axes computed")
        sys.exit(1)

    agg_axis = torch.stack(axis_list).mean(dim=0)

    save_data = {
        "axes": axes,
        "aggregate": agg_axis,
        "categories": list(axes.keys()),
    }

    torch.save(save_data, output_path)
    print(f"Saved axes to {output_path}")


if __name__ == "__main__":
    main()
