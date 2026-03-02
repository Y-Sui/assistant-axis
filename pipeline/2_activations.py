#!/usr/bin/env python3
"""Step 2: extract per-turn assistant activations for hallucination samples."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict, List

import jsonlines
import torch

from assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel, SpanMapper


def load_rows(responses_file: Path) -> List[dict]:
    rows = []
    with jsonlines.open(responses_file, "r") as reader:
        for row in reader:
            rows.append(row)
    return rows


def collect_response_files(responses_file: str | None, responses_dir: str | None) -> List[Path]:
    files: List[Path] = []
    if responses_file:
        files.append(Path(responses_file))
    if responses_dir:
        files.extend(sorted(Path(responses_dir).glob("*.jsonl")))
    if not files:
        raise ValueError("Provide --responses_file or --responses_dir")
    return files


def parse_layers(pm: ProbingModel, layers_arg: str) -> List[int]:
    if layers_arg == "all":
        return list(range(len(pm.get_layers())))
    return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]


def extract_activations(
    pm: ProbingModel,
    rows: List[dict],
    layers: List[int],
    batch_size: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    chat_kwargs = {}
    if "qwen" in pm.model_name.lower():
        chat_kwargs["enable_thinking"] = False

    activations: Dict[str, torch.Tensor] = {}

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        conversations = [row["conversation"] for row in chunk]

        batch_acts, batch_meta = extractor.batch_conversations(
            conversations,
            layer=layers,
            max_length=max_length,
            **chat_kwargs,
        )
        _, batch_spans, _ = encoder.build_batch_turn_spans(conversations, **chat_kwargs)
        conv_acts_list = span_mapper.map_spans(batch_acts, batch_spans, batch_meta)

        for row, conv_acts in zip(chunk, conv_acts_list):
            if conv_acts.numel() == 0:
                continue
            assistant_turn_acts = conv_acts[1::2]
            if assistant_turn_acts.shape[0] == 0:
                continue
            activations[row["sample_id"]] = assistant_turn_acts[-1].cpu()

    return activations


def main():
    parser = argparse.ArgumentParser(description="Extract activations for hallucination samples")
    parser.add_argument("--model", required=True)
    parser.add_argument("--responses_file", type=str, default=None)
    parser.add_argument("--responses_dir", type=str, default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    files = collect_response_files(args.responses_file, args.responses_dir)

    rows: List[dict] = []
    for file in files:
        rows.extend(load_rows(file))

    pm = ProbingModel(args.model, device=args.device)
    layers = parse_layers(pm, args.layers)
    acts = extract_activations(pm, rows, layers, args.batch_size, args.max_length)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(acts, output_file)

    print(f"Saved {len(acts)} activations to {output_file}")


if __name__ == "__main__":
    main()
