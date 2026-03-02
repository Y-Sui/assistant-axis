#!/usr/bin/env python3
"""Step 1: generate hallucination trajectories turn-by-turn."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict, List

import jsonlines

from assistant_axis.generation import generate_response
from assistant_axis.internals import ProbingModel


def load_trajectories(path: Path) -> List[dict]:
    trajectories = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            trajectories.append(row)
    return trajectories


def generate_trajectory_rows(pm: ProbingModel, trajectory: dict, args) -> List[dict]:
    trajectory_id = trajectory["trajectory_id"]
    messages = trajectory["messages"]
    notes = trajectory.get("notes", "")

    conversation: List[Dict[str, str]] = []
    rows: List[dict] = []

    chat_kwargs = {}
    if "qwen" in pm.model_name.lower():
        chat_kwargs["enable_thinking"] = False

    for turn_idx, msg in enumerate(messages):
        user_text = msg["content"] if isinstance(msg, dict) else str(msg)
        conversation.append({"role": "user", "content": user_text})

        assistant_text = generate_response(
            model=pm.model,
            tokenizer=pm.tokenizer,
            conversation=conversation,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            **chat_kwargs,
        )
        conversation.append({"role": "assistant", "content": assistant_text})

        rows.append(
            {
                "sample_id": f"{trajectory_id}_t{turn_idx}",
                "trajectory_id": trajectory_id,
                "turn_idx": turn_idx,
                "conversation": [dict(m) for m in conversation],
                "assistant_text": assistant_text,
                "notes": notes,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate hallucination trajectory responses")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument(
        "--trajectories_file",
        type=str,
        default="data/degeneration/hallucination_trajectories.jsonl",
        help="Input JSONL trajectory file",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    trajectories_path = Path(args.trajectories_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "responses.jsonl"

    trajectories = load_trajectories(trajectories_path)
    pm = ProbingModel(args.model, device=args.device)

    all_rows: List[dict] = []
    for trajectory in trajectories:
        all_rows.extend(generate_trajectory_rows(pm, trajectory, args))

    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(all_rows)

    print(f"Saved {len(all_rows)} assistant turns to {output_file}")


if __name__ == "__main__":
    main()
