#!/usr/bin/env python3
"""
Experiment 1: Separability

For each category, project clean and degen activations onto the axis
and check if they're linearly separated. Also computes Spearman r
between projection and judge score.

Expected result: clean_mean > degen_mean, spearman_r > 0 for all categories.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def spearman_r(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--axes_file", required=True)
    args = parser.parse_args()

    axes_data = torch.load(args.axes_file, map_location="cpu", weights_only=False)
    axes = axes_data["axes"]

    header = f"{'category':<20} {'layer':>6} {'clean_mean':>10} {'degen_mean':>10} {'gap':>8} {'spearman_r':>11} {'n':>6}"
    print(header)
    print("-" * len(header))

    for cat, axis in sorted(axes.items()):
        act_file = Path(args.activations_dir) / f"{cat}.pt"
        score_file = Path(args.scores_dir) / f"{cat}.json"
        if not act_file.exists() or not score_file.exists():
            print(f"{cat:<20}  (missing files)")
            continue

        activations = torch.load(act_file, map_location="cpu", weights_only=False)
        scores = json.load(open(score_file))

        best_layer = int(axis.norm(dim=1).argmax().item())
        v = axis[best_layer].float()
        v = v / v.norm()

        clean_projs, degen_projs = [], []
        all_projs, all_scores = [], []

        for key, act in activations.items():
            score = scores.get(key)
            if score is None:
                continue
            proj = float(act[best_layer].float() @ v)
            all_projs.append(proj)
            all_scores.append(score)
            if key.startswith("clean_"):
                clean_projs.append(proj)
            elif key.startswith("degen_"):
                degen_projs.append(proj)

        if not clean_projs or not degen_projs:
            print(f"{cat:<20}  (no samples)")
            continue

        clean_mean = sum(clean_projs) / len(clean_projs)
        degen_mean = sum(degen_projs) / len(degen_projs)
        gap = clean_mean - degen_mean
        r = spearman_r(all_projs, all_scores)
        n = len(clean_projs) + len(degen_projs)

        print(f"{cat:<20} {best_layer:>6} {clean_mean:>10.3f} {degen_mean:>10.3f} {gap:>8.3f} {r:>11.3f} {n:>6}")


if __name__ == "__main__":
    main()
