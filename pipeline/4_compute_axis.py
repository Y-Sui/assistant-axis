"""Compute contrastive drift axis from activations and judgments.

Joins per-step activations with judge labels, then computes
mean(bad) - mean(good) per problem, averaged across problems.

Usage:
    python pipeline/4_compute_axis.py [--activations-dir DIR] [--judgments-dir DIR] [--metric METRIC]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go

from src.axis import save_axis
from src.contrastive_axis import compute_contrastive_axis
from src.models import get_config


def load_judgments(judgments_dir: Path) -> dict[tuple[str, str, int], dict]:
    """Load all judgment files into a lookup dict.

    Returns:
        Dict mapping (instance_id, run_id, step_index) -> metrics dict.
    """
    lookup = {}
    for jf in sorted(judgments_dir.glob("*.judgments.json")):
        with open(jf) as f:
            data = json.load(f)
        instance_id = data["instance_id"]
        run_id = data["run_id"]
        for step in data["steps"]:
            key = (instance_id, run_id, step["step_index"])
            lookup[key] = step["metrics"]
    return lookup


def load_activations(activations_dir: Path) -> list[dict]:
    """Load all activation .pt files."""
    results = []
    for pt_file in sorted(activations_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        results.append(data)
    return results


def plot_label_distribution(labels: list[int], output_path: Path):
    """Bar chart of label distribution."""
    counts = {0: 0, 1: 0, 2: 0}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    fig = go.Figure(go.Bar(
        x=["Good (0)", "Bad (1)", "Ambiguous (2)"],
        y=[counts[0], counts[1], counts[2]],
        marker_color=["#2ecc71", "#e74c3c", "#f39c12"],
    ))
    fig.update_layout(title="Label Distribution", yaxis_title="Count", width=600, height=400)
    fig.write_html(output_path)


def plot_axis_norms(norms: list[float], layer_indices: list[int], output_path: Path):
    """Per-layer raw axis norm plot."""
    fig = go.Figure(go.Bar(x=[f"L{i}" for i in layer_indices], y=norms))
    fig.update_layout(
        title="Per-Layer Contrastive Axis Norms (before normalization)",
        xaxis_title="Layer", yaxis_title="L2 Norm",
        width=800, height=400,
    )
    fig.write_html(output_path)


def plot_projection_histograms(
    good_projs: list[float],
    bad_projs: list[float],
    layer_label: str,
    output_path: Path,
):
    """Overlapping histograms of good vs bad projections."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=good_projs, name="Good", opacity=0.6, marker_color="#2ecc71"))
    fig.add_trace(go.Histogram(x=bad_projs, name="Bad", opacity=0.6, marker_color="#e74c3c"))
    fig.update_layout(
        title=f"Projections onto Drift Axis — {layer_label}",
        xaxis_title="Projection", yaxis_title="Count",
        barmode="overlay", width=800, height=400,
    )
    fig.write_html(output_path)


def main():
    parser = argparse.ArgumentParser(description="Compute contrastive drift axis")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/analysis"),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="hallucination",
        choices=["hallucination", "dishonesty"],
        help="Which metric's label to use for good/bad classification",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get model config for layer info
    config = get_config(args.model)
    total_layers = config["total_layers"]
    layer_indices = list(range(total_layers - 10, total_layers))

    # Load data
    print("Loading judgments...")
    judgment_lookup = load_judgments(args.judgments_dir)
    print(f"  {len(judgment_lookup)} step judgments loaded")

    print("Loading activations...")
    all_act_data = load_activations(args.activations_dir)
    print(f"  {len(all_act_data)} activation files loaded")

    # Join activations with labels
    activations = []
    labels = []
    problem_ids = []
    n_matched = 0
    n_unmatched = 0

    for data in all_act_data:
        instance_id = data["instance_id"]
        run_id = data.get("run_id", "00")
        acts = data["activations"]  # (n_steps, n_layers, hidden_dim)
        meta = data.get("step_metadata", data.get("turn_metadata", []))

        for i, m in enumerate(meta):
            step_index = m.get("step_index", m.get("turn_index"))
            key = (instance_id, run_id, step_index)

            if key in judgment_lookup:
                metrics = judgment_lookup[key]
                label = metrics[args.metric]["label"]
                activations.append(acts[i])
                labels.append(label)
                problem_ids.append(instance_id)
                n_matched += 1
            else:
                n_unmatched += 1

    print(f"Matched: {n_matched}, unmatched: {n_unmatched}")
    print(f"Labels: good={labels.count(0)}, bad={labels.count(1)}, ambiguous={labels.count(2)}")

    if n_matched == 0:
        print("No matched steps. Check that activation and judgment files align.")
        return

    # Plot label distribution
    plot_label_distribution(labels, plots_dir / "label_distribution.html")

    # Compute contrastive axis
    print("Computing contrastive drift axis...")
    drift_axis, metadata = compute_contrastive_axis(activations, labels, problem_ids)
    metadata["metric"] = args.metric
    metadata["layer_indices"] = layer_indices

    print(f"  Problems used: {metadata['n_problems_used']}, skipped: {metadata['n_problems_skipped']}")
    print(f"  Good steps: {metadata['n_good_steps']}, bad steps: {metadata['n_bad_steps']}")

    # Save axis
    save_axis(drift_axis, args.output_dir / "drift_axis.pt", metadata=metadata)
    print(f"Saved drift axis to {args.output_dir / 'drift_axis.pt'}")

    # Diagnostic plots
    plot_axis_norms(metadata["per_layer_raw_norms"], layer_indices, plots_dir / "axis_norms.html")

    # Projection histograms for middle extracted layer
    mid_layer = len(layer_indices) // 2
    axis_layer = drift_axis[mid_layer].float()
    good_projs = []
    bad_projs = []
    for act, label in zip(activations, labels):
        proj = float(act[mid_layer].float() @ axis_layer)
        if label == 0:
            good_projs.append(proj)
        elif label == 1:
            bad_projs.append(proj)

    if good_projs and bad_projs:
        plot_projection_histograms(
            good_projs, bad_projs,
            f"Layer {layer_indices[mid_layer]}",
            plots_dir / "projection_histogram.html",
        )

    # Summary
    print("\n" + "=" * 60)
    print("CONTRASTIVE AXIS SUMMARY")
    print("=" * 60)
    print(f"Metric:          {args.metric}")
    print(f"Problems used:   {metadata['n_problems_used']}")
    print(f"Problems skipped: {metadata['n_problems_skipped']}")
    print(f"Good steps:      {metadata['n_good_steps']}")
    print(f"Bad steps:       {metadata['n_bad_steps']}")
    print(f"Axis shape:      {drift_axis.shape}")
    print(f"Per-layer norms: {['%.3f' % n for n in metadata['per_layer_raw_norms']]}")


if __name__ == "__main__":
    main()
