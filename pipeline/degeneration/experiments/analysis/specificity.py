#!/usr/bin/env python3
"""
Experiment 3: Specificity

For every pair of (data_category, axis_category), compute the mean clean-degen
projection gap. Prints a matrix where rows = data, cols = axis.

Expected result: diagonal entries (data_cat == axis_cat) are largest in each row,
meaning each axis is specific to its own category rather than a generic direction.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def projection_gap(activations, scores, axis, layer):
    v = axis[layer].float()
    v = v / v.norm()
    clean, degen = [], []
    for key, act in activations.items():
        score = scores.get(key)
        if score is None:
            continue
        proj = float(act[layer].float() @ v)
        if key.startswith("clean_"):
            clean.append(proj)
        elif key.startswith("degen_"):
            degen.append(proj)
    if not clean or not degen:
        return None
    return sum(clean) / len(clean) - sum(degen) / len(degen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--axes_file", required=True)
    args = parser.parse_args()

    axes_data = torch.load(args.axes_file, map_location="cpu", weights_only=False)
    axes = axes_data["axes"]
    categories = sorted(axes.keys())

    all_acts, all_scores = {}, {}
    for cat in categories:
        act_file = Path(args.activations_dir) / f"{cat}.pt"
        score_file = Path(args.scores_dir) / f"{cat}.json"
        if act_file.exists() and score_file.exists():
            all_acts[cat]   = torch.load(act_file, map_location="cpu", weights_only=False)
            all_scores[cat] = json.load(open(score_file))

    col_w = 14
    print(f"\n{'':20}" + "".join(f"{c:>{col_w}}" for c in categories))
    print("-" * (20 + col_w * len(categories)))

    for data_cat in categories:
        if data_cat not in all_acts:
            continue
        row = f"{data_cat:<20}"
        for axis_cat in categories:
            axis = axes[axis_cat]
            best_layer = int(axis.norm(dim=1).argmax().item())
            gap = projection_gap(all_acts[data_cat], all_scores[data_cat], axis, best_layer)
            if gap is None:
                row += f"{'N/A':>{col_w}}"
            else:
                marker = "*" if data_cat == axis_cat else " "
                row += f"{gap:>{col_w - 1}.3f}{marker}"
        print(row)

    print("\n* = diagonal (own-category axis)")


if __name__ == "__main__":
    main()
