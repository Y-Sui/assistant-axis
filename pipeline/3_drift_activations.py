"""Extract activations from trajectory replays using incremental forward passes.

For each trajectory with N assistant turns, performs N forward passes:
  - Pass k: feed conversation up to assistant turn k
  - Extract mean-pooled activations of k-th assistant response
  - Layers 26-35 (last 10 of 36 for Qwen-3-4B)

Usage:
    python pipeline/3_drift_activations.py [--input FILE] [--output-dir DIR] [--model MODEL]
"""

import argparse
import json
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from src.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper
from src.models import get_config


def extract_trajectory_activations(
    trajectory: dict,
    probing_model: ProbingModel,
    encoder: ConversationEncoder,
    extractor: ActivationExtractor,
    span_mapper: SpanMapper,
    layer_indices: list[int],
    max_length: int = 32768,
) -> dict | None:
    """Extract per-turn activations for a single trajectory via incremental replay.

    Args:
        trajectory: Parsed trajectory dict with "conversation" key.
        probing_model: Loaded ProbingModel.
        encoder: ConversationEncoder for the model.
        extractor: ActivationExtractor instance.
        span_mapper: SpanMapper for mean pooling.
        layer_indices: Which layers to extract.
        max_length: Maximum sequence length.

    Returns:
        Dict with instance_id, n_turns, activations tensor (N, n_layers, hidden_dim),
        and turn_metadata list, or None if extraction fails.
    """
    conversation = trajectory["conversation"]
    instance_id = trajectory["instance_id"]

    # Find assistant turn indices
    assistant_turn_indices = [
        i for i, msg in enumerate(conversation) if msg["role"] == "assistant"
    ]

    if not assistant_turn_indices:
        return None

    n_layers = len(layer_indices)
    hidden_dim = probing_model.hidden_size
    all_activations = []
    turn_metadata = []

    for turn_k, asst_idx in enumerate(tqdm(
        assistant_turn_indices,
        desc=f"  {instance_id}",
        leave=False,
    )):
        # Build conversation up to and including this assistant turn
        conv_prefix = conversation[: asst_idx + 1]

        # Check length
        token_ids = encoder.token_ids(conv_prefix, enable_thinking=False)
        if len(token_ids) > max_length:
            print(f"    Turn {turn_k}: {len(token_ids)} tokens > {max_length}, skipping rest")
            break

        # Get turn spans for this prefix
        full_ids, spans = encoder.build_turn_spans(conv_prefix, enable_thinking=False)

        # Find the span for the last assistant turn
        assistant_spans = [s for s in spans if s["role"] == "assistant"]
        if not assistant_spans:
            continue
        target_span = assistant_spans[-1]

        # Extract full activations for the prefix conversation
        try:
            activations = extractor.full_conversation(
                conv_prefix,
                layer=layer_indices,
                enable_thinking=False,
            )
            # activations shape: (n_layers, num_tokens, hidden_dim)
        except Exception as e:
            print(f"    Turn {turn_k}: extraction error: {e}")
            continue

        # Mean-pool over the target assistant span
        start_idx = target_span["start"]
        end_idx = target_span["end"]

        if start_idx >= activations.shape[1] or end_idx > activations.shape[1]:
            print(f"    Turn {turn_k}: span out of bounds, skipping")
            continue

        span_activations = activations[:, start_idx:end_idx, :]  # (n_layers, span_len, hidden_dim)
        mean_activation = span_activations.mean(dim=1)  # (n_layers, hidden_dim)

        all_activations.append(mean_activation.cpu())
        turn_metadata.append({
            "turn_index": turn_k,
            "global_msg_index": asst_idx,
            "n_tokens_context": len(token_ids),
            "n_tokens_response": end_idx - start_idx,
        })

        # Free GPU memory between passes
        del activations
        torch.cuda.empty_cache()

    if not all_activations:
        return None

    # Stack: (N_turns, n_layers, hidden_dim)
    stacked = torch.stack(all_activations)

    return {
        "instance_id": instance_id,
        "n_turns": len(all_activations),
        "activations": stacked,
        "turn_metadata": turn_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract drift activations from trajectories")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/drift/qwen3-4b/parsed/trajectories.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/activations"),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get model config
    config = get_config(args.model)
    total_layers = config["total_layers"]
    # Extract last 10 layers
    layer_indices = list(range(total_layers - 10, total_layers))
    print(f"Model: {args.model}")
    print(f"Extracting layers: {layer_indices}")

    # Load model
    print("Loading model...")
    pm = ProbingModel(args.model, device=args.device)
    encoder = ConversationEncoder(pm.tokenizer, args.model)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    # Load trajectories
    trajectories = []
    with open(args.input) as f:
        for line in f:
            trajectories.append(json.loads(line))

    print(f"Loaded {len(trajectories)} trajectories")

    # Process each trajectory
    for traj in trajectories:
        instance_id = traj["instance_id"]
        output_path = args.output_dir / f"{instance_id}.pt"

        if output_path.exists():
            print(f"Skipping {instance_id} (already exists)")
            continue

        print(f"Processing {instance_id} ({traj['n_assistant_turns']} assistant turns)...")

        result = extract_trajectory_activations(
            traj, pm, encoder, extractor, span_mapper,
            layer_indices, args.max_length,
        )

        if result is not None:
            torch.save(result, output_path)
            print(f"  Saved: {result['activations'].shape} -> {output_path}")
        else:
            print(f"  No activations extracted for {instance_id}")

        gc.collect()
        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
