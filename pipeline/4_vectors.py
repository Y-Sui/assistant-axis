#!/usr/bin/env python3
"""Step 4: split clean/degenerated samples and compute mean vectors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict

import torch


def load_activations(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_scores(path: Path) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)


def build_vectors(
    activations: Dict[str, torch.Tensor],
    scores: Dict[str, int],
    clean_max: int = 3,
    degen_min: int = 7,
    min_count: int = 30,
) -> dict:
    clean = []
    degen = []
    ignored_count = 0
    missing_scores = 0

    for sample_id, act in activations.items():
        score = scores.get(sample_id)
        if score is None:
            missing_scores += 1
            continue
        if score <= clean_max:
            clean.append(act)
        elif score >= degen_min:
            degen.append(act)
        else:
            ignored_count += 1

    if len(clean) < min_count:
        raise ValueError(f"Only {len(clean)} clean samples, need >= {min_count}")
    if len(degen) < min_count:
        raise ValueError(f"Only {len(degen)} degenerated samples, need >= {min_count}")

    clean_mean = torch.stack(clean).mean(dim=0)
    degen_mean = torch.stack(degen).mean(dim=0)

    return {
        "clean_mean": clean_mean,
        "degen_mean": degen_mean,
        "clean_count": len(clean),
        "degen_count": len(degen),
        "ignored_count": ignored_count,
        "missing_scores": missing_scores,
        "clean_max": clean_max,
        "degen_min": degen_min,
        "min_count": min_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Build clean/degenerated vectors")
    parser.add_argument("--activations_file", required=True)
    parser.add_argument("--scores_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--clean_max", type=int, default=3)
    parser.add_argument("--degen_min", type=int, default=7)
    parser.add_argument("--min_count", type=int, default=30)
    args = parser.parse_args()

    activations = load_activations(Path(args.activations_file))
    scores = load_scores(Path(args.scores_file))
    vectors = build_vectors(
        activations,
        scores,
        clean_max=args.clean_max,
        degen_min=args.degen_min,
        min_count=args.min_count,
    )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, output_file)

    print(
        f"Saved vectors to {output_file} "
        f"(clean={vectors['clean_count']}, degen={vectors['degen_count']}, ignored={vectors['ignored_count']})"
    )


if __name__ == "__main__":
    main()
