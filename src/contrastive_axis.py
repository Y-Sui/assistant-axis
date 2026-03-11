"""Contrastive drift axis computation.

Computes a drift axis as mean(bad) - mean(good) per problem, then averages
across problems. This controls for problem difficulty by giving each problem
equal weight regardless of how many steps it contributes.
"""

import torch


def compute_contrastive_axis(
    activations: list[torch.Tensor],
    labels: list[int],
    problem_ids: list[str],
) -> tuple[torch.Tensor, dict]:
    """Compute a contrastive drift axis from labeled activations.

    For each unique problem, computes mean(bad) - mean(good), then averages
    the per-problem diffs. Normalizes per layer.

    Args:
        activations: List of tensors, each shape (n_layers, hidden_dim).
        labels: List of integer labels (0=good, 1=bad, 2=ambiguous).
        problem_ids: List of problem identifiers, one per activation.

    Returns:
        Tuple of (axis, metadata):
        - axis: Tensor of shape (n_layers, hidden_dim), normalized per layer.
        - metadata: Dict with counts and diagnostics.
    """
    assert len(activations) == len(labels) == len(problem_ids)

    # Group by problem
    problems: dict[str, dict[str, list]] = {}
    for act, label, pid in zip(activations, labels, problem_ids):
        if pid not in problems:
            problems[pid] = {"good": [], "bad": []}
        if label == 0:
            problems[pid]["good"].append(act)
        elif label == 1:
            problems[pid]["bad"].append(act)
        # label == 2 (ambiguous) is skipped

    # Compute per-problem diffs
    per_problem_diffs = []
    n_problems_skipped = 0
    n_good_total = 0
    n_bad_total = 0

    for pid, groups in problems.items():
        good = groups["good"]
        bad = groups["bad"]

        if not good or not bad:
            n_problems_skipped += 1
            continue

        good_mean = torch.stack(good).float().mean(0)  # (n_layers, hidden_dim)
        bad_mean = torch.stack(bad).float().mean(0)
        diff = bad_mean - good_mean
        per_problem_diffs.append(diff)

        n_good_total += len(good)
        n_bad_total += len(bad)

    if not per_problem_diffs:
        raise ValueError(
            "No problems had both good and bad steps. "
            f"Total problems: {len(problems)}, all skipped."
        )

    # Average across problems
    raw_axis = torch.stack(per_problem_diffs).mean(0)  # (n_layers, hidden_dim)

    # Normalize per layer
    norms = raw_axis.norm(dim=1, keepdim=True) + 1e-8
    drift_axis = raw_axis / norms

    metadata = {
        "n_problems_used": len(per_problem_diffs),
        "n_problems_skipped": n_problems_skipped,
        "n_good_steps": n_good_total,
        "n_bad_steps": n_bad_total,
        "per_layer_raw_norms": raw_axis.norm(dim=1).tolist(),
    }

    return drift_axis, metadata
