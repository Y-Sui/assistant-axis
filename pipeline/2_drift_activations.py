"""Parse raw trajectories and extract activations via vLLM hidden state extraction.

Reads raw .traj.json files from mini-swe-agent, parses them into clean
conversations, then for each trajectory with N assistant turns, extracts
hidden states for N conversation prefixes using VllmHiddenStatesGenerator:
  - Prefix k: conversation up to assistant turn k
  - Extract hidden states at specified layers via vLLM prefill
  - Mean-pool activations over k-th assistant response span
  - Output shape per trajectory: (N_turns, n_layers, hidden_dim)

Usage:
    python pipeline/2_drift_activations.py [--input-dir DIR] [--output-dir DIR] [--model MODEL]
"""

import argparse
import json
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from speculators.data_generation import VllmHiddenStatesGenerator
from src.internals import ConversationEncoder
from src.models import get_config


def parse_trajectory(traj_path: Path) -> dict | None:
    """Parse a single .traj.json file into a clean format.

    Returns:
        Dict with instance_id, n_assistant_turns, and conversation
        (list of {role, content} dicts), or None if parsing fails.
    """
    with open(traj_path) as f:
        traj = json.load(f)

    instance_id = traj.get("instance_id", traj_path.stem.replace(".traj", ""))

    messages = traj.get("messages", traj.get("history", []))
    if not messages:
        print(f"  Skipping {instance_id}: no messages found")
        return None

    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user", "assistant"):
            conversation.append({"role": role, "content": content})
        elif role == "tool":
            # Tool responses become user messages (observation)
            conversation.append({"role": "user", "content": content})
        else:
            conversation.append({"role": role, "content": content})

    n_assistant_turns = sum(1 for m in conversation if m["role"] == "assistant")

    if n_assistant_turns == 0:
        print(f"  Skipping {instance_id}: no assistant turns")
        return None

    return {
        "instance_id": instance_id,
        "n_assistant_turns": n_assistant_turns,
        "conversation": conversation,
    }


def extract_trajectory_activations(
    trajectory: dict,
    generator: VllmHiddenStatesGenerator,
    encoder: ConversationEncoder,
    layer_indices: list[int],
    max_length: int = 32768,
) -> dict | None:
    """Extract per-turn activations for a single trajectory via vLLM hidden state extraction.

    Args:
        trajectory: Parsed trajectory dict with "conversation" key.
        generator: VllmHiddenStatesGenerator instance.
        encoder: ConversationEncoder for the model.
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

    all_activations = []
    turn_metadata = []

    for turn_k, asst_idx in enumerate(tqdm(
        assistant_turn_indices,
        desc=f"  {instance_id}",
        leave=False,
    )):
        # Build conversation up to and including this assistant turn
        conv_prefix = conversation[: asst_idx + 1]

        # Tokenize and check length
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

        # Extract hidden states via vLLM
        try:
            results = generator.generate([token_ids])
            # results[0]["hidden_states"] is a list of tensors, one per layer
            hidden_states = results[0]["hidden_states"]

            # Stack into (n_layers, num_tokens, hidden_dim)
            activations = torch.stack(hidden_states)
            # Squeeze batch dim if present: (n_layers, 1, seq, dim) -> (n_layers, seq, dim)
            if activations.dim() == 4:
                activations = activations.squeeze(1)
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

        all_activations.append(mean_activation.cpu().to(torch.bfloat16))
        turn_metadata.append({
            "turn_index": turn_k,
            "global_msg_index": asst_idx,
            "n_tokens_context": len(token_ids),
            "n_tokens_response": end_idx - start_idx,
        })

        # Free memory
        del activations, hidden_states, results

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
    parser = argparse.ArgumentParser(description="Parse trajectories and extract drift activations")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/raw"),
        help="Directory containing .traj.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/activations"),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get model config
    config = get_config(args.model)
    total_layers = config["total_layers"]
    # Extract last 10 layers
    layer_indices = list(range(total_layers - 10, total_layers))
    print(f"Model: {args.model}")
    print(f"Extracting layers: {layer_indices}")

    # Load tokenizer only (no full model needed)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = ConversationEncoder(tokenizer, args.model)

    # Initialize vLLM hidden state generator
    print("Initializing VllmHiddenStatesGenerator...")
    generator = VllmHiddenStatesGenerator(
        model_path=args.model,
        layer_ids=layer_indices,
        max_model_len=args.max_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Parse raw trajectory files
    traj_files = sorted(args.input_dir.glob("*.traj.json"))
    if not traj_files:
        print(f"No .traj.json files found in {args.input_dir}")
        return

    print(f"Found {len(traj_files)} trajectory files")

    # Process each trajectory: parse then extract
    for traj_path in traj_files:
        traj = parse_trajectory(traj_path)
        if traj is None:
            continue

        instance_id = traj["instance_id"]
        output_path = args.output_dir / f"{instance_id}.pt"

        if output_path.exists():
            print(f"Skipping {instance_id} (already exists)")
            continue

        print(f"Processing {instance_id} ({traj['n_assistant_turns']} assistant turns)...")

        result = extract_trajectory_activations(
            traj, generator, encoder,
            layer_indices, args.max_length,
        )

        if result is not None:
            torch.save(result, output_path)
            print(f"  Saved: {result['activations'].shape} -> {output_path}")
        else:
            print(f"  No activations extracted for {instance_id}")

        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
