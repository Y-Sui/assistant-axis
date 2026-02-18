#!/usr/bin/env python3
"""
Compute per-category vectors for degeneration axes.

Uses score thresholds to select good vs degen samples, then computes
mean activation vectors for each category.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_scores(scores_file: Path) -> dict:
    with open(scores_file, 'r') as f:
        return json.load(f)


def load_activations(activations_file: Path) -> dict:
    return torch.load(activations_file, map_location="cpu", weights_only=False)


def compute_mean_vector(activations: dict, scores: dict, label: str, min_score: int, max_score: int, min_count: int) -> torch.Tensor:
    filtered = []
    for key, act in activations.items():
        if not key.startswith(f"{label}_"):
            continue
        score = scores.get(key)
        if score is None:
            continue
        if score < min_score or score > max_score:
            continue
        filtered.append(act)

    if len(filtered) < min_count:
        raise ValueError(f"Only {len(filtered)} samples for label '{label}', need {min_count}")

    stacked = torch.stack(filtered)
    return stacked.mean(dim=0), len(filtered)


def main():
    parser = argparse.ArgumentParser(description="Compute degeneration vectors per category")
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory with activation .pt files")
    parser.add_argument("--scores_dir", type=str, required=True, help="Directory with score JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for vector .pt files")
    parser.add_argument("--good_min_score", type=int, default=3, help="Minimum score for good samples")
    parser.add_argument("--good_max_score", type=int, default=3, help="Maximum score for good samples")
    parser.add_argument("--degen_min_score", type=int, default=0, help="Minimum score for degen samples")
    parser.add_argument("--degen_max_score", type=int, default=1, help="Maximum score for degen samples")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum samples required per label")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = Path(args.activations_dir)
    scores_dir = Path(args.scores_dir)

    activation_files = sorted(activations_dir.glob("*.pt"))
    print(f"Found {len(activation_files)} activation files")

    successful = 0
    skipped = 0
    failed = 0

    for act_file in tqdm(activation_files, desc="Computing vectors"):
        category = act_file.stem
        output_file = output_dir / f"{category}.pt"

        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue

        activations = load_activations(act_file)
        if not activations:
            print(f"Warning: No activations for {category}")
            failed += 1
            continue

        scores_file = scores_dir / f"{category}.json"
        if not scores_file.exists():
            print(f"Warning: No scores file for {category}")
            failed += 1
            continue

        scores = load_scores(scores_file)

        try:
            good_vec, good_count = compute_mean_vector(
                activations,
                scores,
                label="good",
                min_score=args.good_min_score,
                max_score=args.good_max_score,
                min_count=args.min_count,
            )
            degen_vec, degen_count = compute_mean_vector(
                activations,
                scores,
                label="degen",
                min_score=args.degen_min_score,
                max_score=args.degen_max_score,
                min_count=args.min_count,
            )

            save_data = {
                "category": category,
                "good": good_vec,
                "degen": degen_vec,
                "good_count": good_count,
                "degen_count": degen_count,
                "thresholds": {
                    "good": [args.good_min_score, args.good_max_score],
                    "degen": [args.degen_min_score, args.degen_max_score],
                },
            }
            torch.save(save_data, output_file)
            successful += 1
        except ValueError as e:
            print(f"Warning: {category}: {e}")
            failed += 1

    print(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
