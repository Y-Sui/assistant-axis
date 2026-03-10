"""
Degeneration Axis - Tools for analyzing and mitigating LLM activation drift.

Provides activation extraction, PCA analysis, axis computation, and
inference-time steering for studying how model activations evolve
over long agentic trajectories.

Example:
    from src import get_config, compute_pca, ActivationSteering
    from src.internals import ProbingModel

    pm = ProbingModel("Qwen/Qwen3-4B")
    config = get_config("Qwen/Qwen3-4B")
"""

from .models import get_config, MODEL_CONFIGS
from .axis import (
    compute_axis,
    load_axis,
    save_axis,
    project,
    project_batch,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
    aggregate_role_vectors,
)
from .steering import (
    ActivationSteering,
    create_feature_ablation_steerer,
    create_multi_feature_steerer,
    create_mean_ablation_steerer,
    load_capping_config,
    build_capping_steerer,
)
from .pca import (
    compute_pca,
    plot_variance_explained,
    MeanScaler,
    L2MeanScaler,
)

__all__ = [
    # Models
    "get_config",
    "MODEL_CONFIGS",
    # Axis
    "compute_axis",
    "load_axis",
    "save_axis",
    "project",
    "project_batch",
    "cosine_similarity_per_layer",
    "axis_norm_per_layer",
    "aggregate_role_vectors",
    # Steering
    "ActivationSteering",
    "create_feature_ablation_steerer",
    "create_multi_feature_steerer",
    "create_mean_ablation_steerer",
    "load_capping_config",
    "build_capping_steerer",
    # PCA
    "compute_pca",
    "plot_variance_explained",
    "MeanScaler",
    "L2MeanScaler",
]
