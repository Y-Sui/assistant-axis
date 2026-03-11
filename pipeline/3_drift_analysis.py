"""PCA-based drift analysis of activation trajectories.

Pools activations across trajectories, computes PCA per layer, and
analyzes whether PC1 correlates with turn index (systematic drift).

Usage:
    python pipeline/3_drift_analysis.py [--input-dir DIR] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from src.pca import compute_pca, MeanScaler, plot_variance_explained


def load_all_activations(input_dir: Path) -> list[dict]:
    """Load all activation files from a directory."""
    results = []
    for pt_file in sorted(input_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        results.append(data)
        print(f"  Loaded {data['instance_id']}: {data['activations'].shape}")
    return results


def pool_activations(all_data: list[dict]) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Pool (turn_index, activation) pairs across all trajectories.

    Returns:
        activations: (total_turns, n_layers, hidden_dim)
        turn_indices: (total_turns,) - the turn index within each trajectory
        instance_ids: (total_turns,) - which trajectory each turn came from
    """
    all_acts = []
    all_turns = []
    all_ids = []

    for data in all_data:
        acts = data["activations"]  # (N, n_layers, hidden_dim)
        meta = data["turn_metadata"]
        instance_id = data["instance_id"]

        for i, m in enumerate(meta):
            all_acts.append(acts[i])
            all_turns.append(m["turn_index"])
            all_ids.append(instance_id)

    stacked = torch.stack(all_acts)  # (total, n_layers, hidden_dim)
    turn_indices = np.array(all_turns)
    return stacked, turn_indices, all_ids


def analyze_layer(
    activations_2d: np.ndarray,
    turn_indices: np.ndarray,
    instance_ids: list[str],
    layer_label: str,
) -> dict:
    """Run PCA + drift statistics for a single layer.

    Args:
        activations_2d: (n_samples, hidden_dim)
        turn_indices: (n_samples,)
        instance_ids: per-sample trajectory labels
        layer_label: string label for this layer

    Returns:
        Dict with PCA results, statistics, and drift axis direction.
    """
    scaler = MeanScaler()
    pca_transformed, variance_explained, n_components, pca, fitted_scaler = compute_pca(
        activations_2d, layer=None, scaler=scaler, verbose=False,
    )

    pc1 = pca_transformed[:, 0]
    pc2 = pca_transformed[:, 1] if pca_transformed.shape[1] > 1 else np.zeros_like(pc1)

    # Spearman correlation: turn_index vs PC1
    spearman_r, spearman_p = stats.spearmanr(turn_indices, pc1)

    # Linear regression: PC1 = a * turn_index + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(turn_indices, pc1)

    # Drift axis direction (PC1 component in original space)
    drift_axis = torch.from_numpy(pca.components_[0].copy()).float()

    return {
        "layer_label": layer_label,
        "pca_transformed": pca_transformed,
        "variance_explained": variance_explained,
        "drift_axis": drift_axis,
        "scaler_mean": torch.from_numpy(fitted_scaler.mean.copy()).float(),
        "pc1": pc1,
        "pc2": pc2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
    }


def plot_drift_curve(
    turn_indices: np.ndarray,
    pc1_values: np.ndarray,
    instance_ids: list[str],
    layer_label: str,
    stats_dict: dict,
) -> go.Figure:
    """Plot mean PC1 vs turn index with per-trajectory overlays."""
    fig = go.Figure()

    # Per-trajectory lines
    unique_ids = sorted(set(instance_ids))
    for uid in unique_ids:
        mask = np.array([iid == uid for iid in instance_ids])
        turns = turn_indices[mask]
        vals = pc1_values[mask]
        order = np.argsort(turns)
        fig.add_trace(go.Scatter(
            x=turns[order], y=vals[order],
            mode="lines", opacity=0.3, name=uid,
            line=dict(width=1),
            showlegend=False,
        ))

    # Mean curve
    unique_turns = sorted(set(turn_indices))
    mean_pc1 = [pc1_values[turn_indices == t].mean() for t in unique_turns]
    std_pc1 = [pc1_values[turn_indices == t].std() for t in unique_turns]

    fig.add_trace(go.Scatter(
        x=unique_turns, y=mean_pc1,
        mode="lines+markers", name="Mean PC1",
        line=dict(width=3, color="red"),
        error_y=dict(type="data", array=std_pc1, visible=True),
    ))

    # Regression line
    x_line = np.array([min(unique_turns), max(unique_turns)])
    y_line = stats_dict["slope"] * x_line + stats_dict["intercept"]
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines", name=f"Fit (R\u00b2={stats_dict['r_squared']:.3f})",
        line=dict(dash="dash", color="black", width=2),
    ))

    fig.update_layout(
        title=f"Drift Curve - {layer_label}",
        xaxis_title="Turn Index",
        yaxis_title="PC1 Projection",
        width=900, height=600,
        annotations=[dict(
            text=(
                f"Spearman r={stats_dict['spearman_r']:.3f} (p={stats_dict['spearman_p']:.2e})<br>"
                f"Slope={stats_dict['slope']:.4f}, R\u00b2={stats_dict['r_squared']:.3f}"
            ),
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False, font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
        )],
    )
    return fig


def plot_pca_scatter(
    pc1: np.ndarray,
    pc2: np.ndarray,
    turn_indices: np.ndarray,
    layer_label: str,
) -> go.Figure:
    """PC1 vs PC2 scatter colored by turn index."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pc1, y=pc2,
        mode="markers",
        marker=dict(
            color=turn_indices,
            colorscale="Viridis",
            colorbar=dict(title="Turn"),
            size=5,
            opacity=0.7,
        ),
    ))
    fig.update_layout(
        title=f"PCA Scatter - {layer_label}",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=700, height=600,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Drift analysis via PCA")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/activations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/analysis"),
    )
    args = parser.parse_args()

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    print("Loading activations...")
    all_data = load_all_activations(args.input_dir)
    if not all_data:
        print("No activation files found.")
        return

    # Pool across trajectories
    print("Pooling activations...")
    activations, turn_indices, instance_ids = pool_activations(all_data)
    n_samples, n_layers, hidden_dim = activations.shape
    print(f"Total samples: {n_samples}, layers: {n_layers}, hidden_dim: {hidden_dim}")

    # Analyze each layer
    drift_axes = {}
    summary_rows = []

    for layer_idx in range(n_layers):
        # Layer label (offset from total layers)
        # Assuming last 10 layers of 36: indices 26-35
        actual_layer = activations.shape[1]  # This is n_layers in the extraction
        layer_label = f"Layer {layer_idx} (of extracted)"

        acts_2d = activations[:, layer_idx, :].numpy()

        print(f"\nAnalyzing {layer_label}...")
        result = analyze_layer(acts_2d, turn_indices, instance_ids, layer_label)

        drift_axes[layer_idx] = {
            "direction": result["drift_axis"],
            "scaler_mean": result["scaler_mean"],
        }

        summary_rows.append({
            "layer": layer_idx,
            "spearman_r": result["spearman_r"],
            "spearman_p": result["spearman_p"],
            "slope": result["slope"],
            "r_squared": result["r_squared"],
            "var_pc1": float(result["variance_explained"][0]),
        })

        print(
            f"  Spearman r={result['spearman_r']:.3f} (p={result['spearman_p']:.2e}), "
            f"R\u00b2={result['r_squared']:.3f}, PC1 var={result['variance_explained'][0]:.3f}"
        )

        # Save plots
        fig_drift = plot_drift_curve(
            turn_indices, result["pc1"], instance_ids, layer_label, result,
        )
        fig_drift.write_html(plots_dir / f"drift_curve_layer{layer_idx}.html")

        fig_scatter = plot_pca_scatter(
            result["pc1"], result["pc2"], turn_indices, layer_label,
        )
        fig_scatter.write_html(plots_dir / f"pca_scatter_layer{layer_idx}.html")

        fig_var = plot_variance_explained(
            result["variance_explained"],
            title=f"Variance Explained - {layer_label}",
            max_components=20,
        )
        fig_var.write_html(plots_dir / f"variance_explained_layer{layer_idx}.html")

    # Save drift axes
    torch.save(drift_axes, args.output_dir / "drift_axis.pt")
    print(f"\nDrift axes saved to {args.output_dir / 'drift_axis.pt'}")

    # Print summary table
    print("\n" + "=" * 80)
    print("DRIFT ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Layer':>6} {'Spearman r':>12} {'p-value':>12} {'Slope':>10} {'R²':>8} {'PC1 var%':>10}")
    print("-" * 80)
    for row in summary_rows:
        print(
            f"{row['layer']:>6} {row['spearman_r']:>12.3f} {row['spearman_p']:>12.2e} "
            f"{row['slope']:>10.4f} {row['r_squared']:>8.3f} {row['var_pc1']*100:>9.1f}%"
        )


if __name__ == "__main__":
    main()
