#!/usr/bin/env python3
"""Step 5: compute hallucination axis from clean/degenerated vectors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timezone

import torch


def load_vectors(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_axis(vectors: dict, model_name: str = "", vectors_file: str = "") -> dict:
    axis = vectors["clean_mean"] - vectors["degen_mean"]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "vectors_file": vectors_file,
        "clean_count": int(vectors["clean_count"]),
        "degen_count": int(vectors["degen_count"]),
        "ignored_count": int(vectors.get("ignored_count", 0)),
        "clean_max": int(vectors["clean_max"]),
        "degen_min": int(vectors["degen_min"]),
    }
    return {"axis": axis, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description="Compute hallucination axis")
    parser.add_argument("--vectors_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    vectors_file = Path(args.vectors_file)
    vectors = load_vectors(vectors_file)
    axis_data = build_axis(vectors, model_name=args.model, vectors_file=str(vectors_file))

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(axis_data, output_file)

    print(
        f"Saved axis to {output_file} "
        f"(clean={axis_data['metadata']['clean_count']}, degen={axis_data['metadata']['degen_count']})"
    )


if __name__ == "__main__":
    main()
