"""Tests for hallucination vector split and axis build."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def load_module(rel_path: str, module_name: str):
    root = Path(__file__).resolve().parents[2]
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_thresholds_and_axis_math(tmp_path: Path):
    step4 = load_module("pipeline/4_vectors.py", "step4_vectors")
    step5 = load_module("pipeline/5_axis.py", "step5_axis")

    activations = {
        "a": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "b": torch.tensor([[3.0, 3.0], [3.0, 3.0]]),
        "c": torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        "d": torch.tensor([[12.0, 12.0], [12.0, 12.0]]),
        "m": torch.tensor([[7.0, 7.0], [7.0, 7.0]]),
    }
    scores = {
        "a": 0,
        "b": 3,
        "c": 8,
        "d": 10,
        "m": 5,
    }

    vectors = step4.build_vectors(activations, scores, clean_max=3, degen_min=7, min_count=2)

    assert vectors["clean_count"] == 2
    assert vectors["degen_count"] == 2
    assert vectors["ignored_count"] == 1

    axis_data = step5.build_axis(vectors, model_name="test-model", vectors_file="vectors.pt")
    expected_clean = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    expected_degen = torch.tensor([[11.0, 11.0], [11.0, 11.0]])
    expected_axis = expected_clean - expected_degen

    assert torch.allclose(vectors["clean_mean"], expected_clean)
    assert torch.allclose(vectors["degen_mean"], expected_degen)
    assert torch.allclose(axis_data["axis"], expected_axis)


def test_min_count_guard(tmp_path: Path):
    step4 = load_module("pipeline/4_vectors.py", "step4_vectors_min")

    activations = {"a": torch.zeros(2, 2), "b": torch.ones(2, 2)}
    scores = {"a": 0, "b": 8}

    try:
        step4.build_vectors(activations, scores, clean_max=3, degen_min=7, min_count=2)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "need >= 2" in str(e)


def test_tiny_e2e_artifact(tmp_path: Path):
    step4 = load_module("pipeline/4_vectors.py", "step4_vectors_e2e")
    step5 = load_module("pipeline/5_axis.py", "step5_axis_e2e")

    acts_path = tmp_path / "activations.pt"
    scores_path = tmp_path / "scores.json"
    vectors_path = tmp_path / "vectors.pt"
    axis_path = tmp_path / "axis.pt"

    activations = {
        "s1": torch.full((2, 3), 1.0),
        "s2": torch.full((2, 3), 2.0),
        "s3": torch.full((2, 3), 9.0),
        "s4": torch.full((2, 3), 11.0),
    }
    scores = {"s1": 2, "s2": 3, "s3": 8, "s4": 9}

    torch.save(activations, acts_path)
    with open(scores_path, "w") as f:
        json.dump(scores, f)

    vectors = step4.build_vectors(activations, scores, clean_max=3, degen_min=7, min_count=2)
    torch.save(vectors, vectors_path)

    loaded_vectors = torch.load(vectors_path, map_location="cpu", weights_only=False)
    axis_data = step5.build_axis(loaded_vectors, model_name="tiny", vectors_file=str(vectors_path))
    torch.save(axis_data, axis_path)

    saved = torch.load(axis_path, map_location="cpu", weights_only=False)
    assert "axis" in saved
    assert "metadata" in saved
    assert saved["metadata"]["clean_count"] == 2
    assert saved["metadata"]["degen_count"] == 2
