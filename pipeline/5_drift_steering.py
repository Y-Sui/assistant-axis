"""Drift steering with contrastive axis.

Uses ActivationSteering with capping intervention to mitigate drift.
Thresholds are computed from good steps (label=0) instead of early turns.

Usage:
    python pipeline/5_drift_steering.py [--drift-axis FILE] [--judgments-dir DIR]
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from src.internals import ProbingModel, ConversationEncoder
from src.steering import ActivationSteering
from src.axis import load_axis
from src.models import get_config


def compute_good_step_threshold(
    activations_dir: Path,
    judgments_dir: Path,
    drift_axis: torch.Tensor,
    metric: str = "hallucination",
    n_std: float = 2.0,
) -> list[float]:
    """Compute per-layer capping thresholds from good steps (label=0).

    Threshold = mean(good_projections) + n_std * std(good_projections).

    Args:
        activations_dir: Directory with .pt activation files.
        judgments_dir: Directory with .judgments.json files.
        drift_axis: Normalized axis tensor, shape (n_layers, hidden_dim).
        metric: Which judgment metric to use for filtering.
        n_std: Number of standard deviations above mean for threshold.

    Returns:
        List of threshold values, one per layer.
    """
    # Load judgments into lookup
    judgment_lookup = {}
    for jf in sorted(judgments_dir.glob("*.judgments.json")):
        with open(jf) as f:
            data = json.load(f)
        instance_id = data["instance_id"]
        run_id = data["run_id"]
        for step in data["steps"]:
            key = (instance_id, run_id, step["step_index"])
            judgment_lookup[key] = step["metrics"]

    # Collect projections of good steps onto the axis
    n_layers = drift_axis.shape[0]
    layer_projections = [[] for _ in range(n_layers)]

    for pt_file in sorted(activations_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        instance_id = data["instance_id"]
        run_id = data.get("run_id", "00")
        acts = data["activations"]  # (n_steps, n_layers, hidden_dim)
        meta = data.get("step_metadata", data.get("turn_metadata", []))

        for i, m in enumerate(meta):
            step_index = m.get("step_index", m.get("turn_index"))
            key = (instance_id, run_id, step_index)

            if key not in judgment_lookup:
                continue

            label = judgment_lookup[key][metric]["label"]
            if label != 0:  # Only good steps
                continue

            for layer_idx in range(n_layers):
                act = acts[i, layer_idx].float()
                v = drift_axis[layer_idx].float()
                proj = float(act @ v)
                layer_projections[layer_idx].append(proj)

    thresholds = []
    for layer_idx in range(n_layers):
        projs = layer_projections[layer_idx]
        if not projs:
            thresholds.append(0.0)
            print(f"  Layer {layer_idx}: no good steps, threshold=0.0")
            continue
        arr = np.array(projs)
        threshold = float(arr.mean() + n_std * arr.std())
        thresholds.append(threshold)
        print(
            f"  Layer {layer_idx}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
            f"threshold={threshold:.3f} (n={len(projs)})"
        )

    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Drift steering with contrastive axis")
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
        "--judgments-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/judgments"),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--metric",
        type=str,
        default="hallucination",
        choices=["hallucination", "dishonesty"],
    )
    parser.add_argument("--n-std", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Load drift axis (new format via save_axis/load_axis)
    print("Loading drift axis...")
    axis_data = torch.load(args.drift_axis, map_location="cpu", weights_only=False)
    drift_axis = axis_data["axis"]  # (n_layers, hidden_dim), normalized per layer
    axis_metadata = axis_data.get("metadata", {})
    print(f"  Axis shape: {drift_axis.shape}")
    print(f"  Metric: {axis_metadata.get('metric', 'unknown')}")
    print(f"  Problems used: {axis_metadata.get('n_problems_used', '?')}")

    # Compute thresholds from good steps
    print("Computing capping thresholds from good steps...")
    thresholds = compute_good_step_threshold(
        args.activations_dir, args.judgments_dir, drift_axis,
        metric=args.metric, n_std=args.n_std,
    )

    # Load model
    print("Loading model...")
    config = get_config(args.model)
    total_layers = config["total_layers"]
    pm = ProbingModel(args.model)

    # Build steering vectors and map to absolute layer indices
    n_extracted = drift_axis.shape[0]
    steering_vectors = []
    cap_thresholds = []
    layer_indices = []

    for i in range(n_extracted):
        absolute_layer = total_layers - n_extracted + i
        steering_vectors.append(drift_axis[i])
        cap_thresholds.append(thresholds[i])
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
