"""Drift steering experiment (future).

Use ActivationSteering with capping intervention to mitigate drift
by capping activations' projection onto the drift axis.

Usage:
    python pipeline/5_drift_steering.py [--drift-axis FILE] [--input FILE]
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from src.internals import ProbingModel, ConversationEncoder
from src.steering import ActivationSteering
from src.models import get_config


def compute_early_turn_threshold(
    activations_dir: Path,
    drift_axis: dict,
    early_turns: int = 5,
    n_std: float = 2.0,
) -> dict[int, float]:
    """Compute per-layer capping thresholds from early turns.

    Threshold = mean(projection) + n_std * std(projection) for turns 0..early_turns-1.

    Returns:
        Dict mapping layer_index -> threshold value.
    """
    # Collect early-turn projections per layer
    layer_projections: dict[int, list[float]] = {}

    for pt_file in sorted(activations_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        acts = data["activations"]  # (N, n_layers, hidden_dim)
        meta = data["turn_metadata"]

        for i, m in enumerate(meta):
            if m["turn_index"] >= early_turns:
                break

            for layer_idx, axis_data in drift_axis.items():
                direction = axis_data["direction"]
                scaler_mean = axis_data["scaler_mean"]

                act = acts[i, layer_idx].float()
                centered = act - scaler_mean
                v = direction / (direction.norm() + 1e-8)
                proj = float(centered @ v)

                if layer_idx not in layer_projections:
                    layer_projections[layer_idx] = []
                layer_projections[layer_idx].append(proj)

    thresholds = {}
    for layer_idx, projs in layer_projections.items():
        arr = np.array(projs)
        thresholds[layer_idx] = float(arr.mean() + n_std * arr.std())
        print(
            f"  Layer {layer_idx}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
            f"threshold={thresholds[layer_idx]:.3f} (n={len(projs)})"
        )

    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Drift steering experiment")
    parser.add_argument(
        "--drift-axis",
        type=Path,
        default=Path("data/drift/qwen3-4b/analysis/drift_axis.pt"),
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/activations"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/drift/qwen3-4b/parsed/trajectories.jsonl"),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--early-turns", type=int, default=5)
    parser.add_argument("--n-std", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Load drift axis
    print("Loading drift axis...")
    drift_axis = torch.load(args.drift_axis, map_location="cpu", weights_only=False)

    # Compute thresholds from early turns
    print("Computing capping thresholds from early turns...")
    thresholds = compute_early_turn_threshold(
        args.activations_dir, drift_axis, args.early_turns, args.n_std,
    )

    # Load model (HuggingFace, not vLLM - hooks need direct model access)
    print("Loading model...")
    config = get_config(args.model)
    total_layers = config["total_layers"]
    pm = ProbingModel(args.model)

    # Build steering vectors and layer indices
    # Map from extracted layer indices back to absolute layer indices
    steering_vectors = []
    cap_thresholds = []
    layer_indices = []

    for layer_idx, axis_data in drift_axis.items():
        absolute_layer = total_layers - 10 + layer_idx  # Reverse the extraction offset
        steering_vectors.append(axis_data["direction"])
        cap_thresholds.append(thresholds[layer_idx])
        layer_indices.append(absolute_layer)

    print(f"Steering on {len(steering_vectors)} layers: {layer_indices}")

    # Create steering context
    steerer = ActivationSteering(
        pm.model,
        steering_vectors=steering_vectors,
        layer_indices=layer_indices,
        intervention_type="capping",
        cap_thresholds=cap_thresholds,
        coefficients=[0.0] * len(steering_vectors),
        positions="all",
    )

    print("Drift steering setup complete.")
    print("TODO: Implement trajectory replay with steering and evaluate drift reduction.")


if __name__ == "__main__":
    main()
