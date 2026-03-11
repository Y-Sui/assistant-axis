"""Tests for contrastive axis computation."""

import pytest
import torch

from src.contrastive_axis import compute_contrastive_axis


class TestComputeContrastiveAxis:
    """Tests for compute_contrastive_axis function."""

    def test_basic_separation(self):
        """Axis should point from good toward bad when data is separable."""
        n_layers, hidden_dim = 4, 8
        torch.manual_seed(42)

        # Create clearly separated good and bad activations
        good_base = torch.zeros(n_layers, hidden_dim)
        bad_base = torch.zeros(n_layers, hidden_dim)
        bad_base[:, 0] = 5.0  # Bad activations shifted along dim 0

        activations = []
        labels = []
        problem_ids = []

        # Problem A: 5 good, 5 bad
        for _ in range(5):
            activations.append(good_base + 0.1 * torch.randn(n_layers, hidden_dim))
            labels.append(0)
            problem_ids.append("A")
        for _ in range(5):
            activations.append(bad_base + 0.1 * torch.randn(n_layers, hidden_dim))
            labels.append(1)
            problem_ids.append("A")

        axis, metadata = compute_contrastive_axis(activations, labels, problem_ids)

        assert axis.shape == (n_layers, hidden_dim)
        # Axis should be roughly aligned with dim 0 (the direction of separation)
        for layer in range(n_layers):
            assert abs(axis[layer, 0]) > 0.5  # Dominant component

    def test_output_normalized_per_layer(self):
        """Each layer's axis vector should have unit norm."""
        n_layers, hidden_dim = 4, 16
        torch.manual_seed(0)

        activations = [torch.randn(n_layers, hidden_dim) for _ in range(20)]
        labels = [0] * 10 + [1] * 10
        problem_ids = ["P1"] * 20

        axis, _ = compute_contrastive_axis(activations, labels, problem_ids)

        for layer in range(n_layers):
            norm = axis[layer].norm().item()
            assert abs(norm - 1.0) < 1e-5, f"Layer {layer} norm={norm}"

    def test_metadata_counts(self):
        """Metadata should report correct counts."""
        activations = [torch.randn(2, 4) for _ in range(12)]
        labels = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2]
        problem_ids = ["A"] * 6 + ["B"] * 6

        _, metadata = compute_contrastive_axis(activations, labels, problem_ids)

        assert metadata["n_problems_used"] == 2
        assert metadata["n_problems_skipped"] == 0
        assert metadata["n_good_steps"] == 5  # 3 from A + 2 from B
        assert metadata["n_bad_steps"] == 5   # 3 from A + 2 from B

    def test_ambiguous_skipped(self):
        """Ambiguous labels (2) should not contribute to good or bad."""
        activations = [torch.randn(2, 4) for _ in range(6)]
        labels = [0, 1, 2, 2, 2, 2]
        problem_ids = ["A"] * 6

        axis, metadata = compute_contrastive_axis(activations, labels, problem_ids)

        assert metadata["n_good_steps"] == 1
        assert metadata["n_bad_steps"] == 1

    def test_problem_without_both_classes_skipped(self):
        """Problems with only good or only bad steps should be skipped."""
        activations = [torch.randn(2, 4) for _ in range(8)]
        labels = [0, 0, 0, 0, 0, 1, 1, 1]
        problem_ids = ["A", "A", "A", "A", "B", "B", "B", "B"]
        # A has only good, B has only bad — except we need one of each somewhere
        # Fix: A has good+bad, B has only good
        labels = [0, 0, 1, 1, 0, 0, 0, 0]

        _, metadata = compute_contrastive_axis(activations, labels, problem_ids)

        assert metadata["n_problems_used"] == 1  # Only A
        assert metadata["n_problems_skipped"] == 1  # B skipped

    def test_per_problem_averaging(self):
        """Each problem should contribute equally regardless of step count."""
        n_layers, hidden_dim = 2, 4
        torch.manual_seed(42)

        # Problem A: many steps, problem B: few steps
        activations = []
        labels = []
        problem_ids = []

        # A: 100 good, 100 bad
        good_a = torch.ones(n_layers, hidden_dim)
        bad_a = -torch.ones(n_layers, hidden_dim)
        for _ in range(100):
            activations.append(good_a)
            labels.append(0)
            problem_ids.append("A")
        for _ in range(100):
            activations.append(bad_a)
            labels.append(1)
            problem_ids.append("A")

        # B: 1 good, 1 bad (opposite direction)
        good_b = -torch.ones(n_layers, hidden_dim)
        bad_b = torch.ones(n_layers, hidden_dim)
        activations.append(good_b)
        labels.append(0)
        problem_ids.append("B")
        activations.append(bad_b)
        labels.append(1)
        problem_ids.append("B")

        axis, _ = compute_contrastive_axis(activations, labels, problem_ids)

        # A's diff = bad_a - good_a = -2*ones
        # B's diff = bad_b - good_b = +2*ones
        # Average = 0, so axis should be near zero (before normalization)
        # But after normalization the direction is arbitrary for near-zero vectors
        # Check that the raw norms are small
        for n in _["per_layer_raw_norms"]:
            assert n < 0.1

    def test_raises_on_no_valid_problems(self):
        """Should raise ValueError if no problems have both good and bad."""
        activations = [torch.randn(2, 4) for _ in range(4)]
        labels = [0, 0, 0, 0]  # All good
        problem_ids = ["A"] * 4

        with pytest.raises(ValueError, match="No problems had both good and bad"):
            compute_contrastive_axis(activations, labels, problem_ids)
