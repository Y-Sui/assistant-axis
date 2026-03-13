"""Single-pass per-step activation extraction via vLLM.

For each trajectory, tokenizes the full conversation once and extracts
hidden states in a single forward pass. Then slices and mean-pools per
assistant turn span. This is correct for causal models since position p
can only attend to positions 0..p.

Usage:
    python pipeline/3_activations.py [--input-dir DIR] [--output-dir DIR] [--model MODEL]
"""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from speculators.data_generation import VllmHiddenStatesGenerator
from src.internals import ConversationEncoder
from src.models import get_config
from src.trajectory import parse_trajectory


def extract_trajectory_activations(
    trajectory: dict,
    generator: VllmHiddenStatesGenerator,
    encoder: ConversationEncoder,
    layer_indices: list[int],
    max_length: int = 32768,
) -> dict | None:
    """Extract per-step activations for a trajectory via single-pass vLLM extraction.

    Algorithm:
    1. Tokenize full conversation once
    2. Build turn spans to identify each assistant turn's token range
    3. Single generator.generate() call
    4. For each assistant span: slice + mean-pool -> (n_layers, hidden_dim)
    5. Stack all turns -> (n_steps, n_layers, hidden_dim)

    Args:
        trajectory: Parsed trajectory dict with "conversation" key.
        generator: VllmHiddenStatesGenerator instance.
        encoder: ConversationEncoder for the model.
        layer_indices: Which layers to extract.
        max_length: Maximum sequence length.

    Returns:
        Dict with instance_id, run_id, n_steps, activations tensor,
        and step_metadata list, or None if extraction fails.
    """
    conversation = trajectory["conversation"]
    instance_id = trajectory["instance_id"]

    # Tokenize full conversation and get turn spans
    full_ids, spans = encoder.build_turn_spans(conversation, enable_thinking=False)

    if len(full_ids) > max_length:
        print(f"  {instance_id}: {len(full_ids)} tokens > {max_length}, truncating")
        full_ids = full_ids[:max_length]

    # Find assistant turn spans
    assistant_spans = [s for s in spans if s["role"] == "assistant"]
    if not assistant_spans:
        return None

    # Single forward pass
    try:
        results = generator.generate([full_ids])
        hidden_states = results[0]["hidden_states"]

        # Stack into (n_layers, seq_len, hidden_dim)
        activations = torch.stack(hidden_states)
        if activations.dim() == 4:
            activations = activations.squeeze(1)
    except Exception as e:
        print(f"  {instance_id}: extraction error: {e}")
        return None

    seq_len = activations.shape[1]

    # Mean-pool per assistant span
    all_activations = []
    step_metadata = []

    for step_index, span in enumerate(assistant_spans):
        start = span["start"]
        end = span["end"]

        # Skip spans that fall outside the (possibly truncated) sequence
        if start >= seq_len:
            break
        end = min(end, seq_len)

        if end <= start:
            continue

        span_acts = activations[:, start:end, :]  # (n_layers, span_len, hidden_dim)
        mean_act = span_acts.mean(dim=1)  # (n_layers, hidden_dim)

        all_activations.append(mean_act.cpu().to(torch.bfloat16))

        # Count context tokens (everything up to the start of this span)
        n_tokens_context = start
        n_tokens_response = end - start

        step_metadata.append({
            "step_index": step_index,
            "global_msg_index": span.get("msg_index", step_index),
            "n_tokens_context": n_tokens_context,
            "n_tokens_response": n_tokens_response,
        })

    if not all_activations:
        return None

    stacked = torch.stack(all_activations)  # (n_steps, n_layers, hidden_dim)

    return {
        "instance_id": instance_id,
        "n_steps": len(all_activations),
        "activations": stacked,
        "step_metadata": step_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Single-pass per-step activation extraction")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/raw"),
        help="Directory containing {instance_id} subdirectories with run_XX.traj.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drift/qwen3-4b/activations"),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get model config
    config = get_config(args.model)
    total_layers = config["total_layers"]
    layer_indices = list(range(total_layers - 10, total_layers))
    print(f"Model: {args.model}")
    print(f"Extracting layers: {layer_indices}")

    # Load tokenizer
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

    # Discover trajectories in problem-based layout:
    #   {input_dir}/{instance_id}/run_XX.traj.json
    problem_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())

    all_traj_files = []
    for problem_dir in problem_dirs:
        for traj_path in sorted(problem_dir.glob("run_*.traj.json")):
            run_id = traj_path.stem.replace(".traj", "").replace("run_", "")
            all_traj_files.append((traj_path, run_id))

    if not all_traj_files:
        print(f"No .traj.json files found under {args.input_dir}")
        return

    print(f"Found {len(all_traj_files)} trajectory files")

    for traj_path, run_id in tqdm(all_traj_files, desc="Extracting activations"):
        traj = parse_trajectory(traj_path)
        if traj is None:
            continue

        instance_id = traj["instance_id"]
        output_path = args.output_dir / f"{instance_id}__run_{run_id}.pt"

        if output_path.exists():
            continue

        result = extract_trajectory_activations(
            traj, generator, encoder, layer_indices, args.max_length,
        )

        if result is not None:
            result["run_id"] = run_id
            torch.save(result, output_path)
            tqdm.write(f"  Saved: {result['activations'].shape} -> {output_path.name}")
        else:
            tqdm.write(f"  No activations for {instance_id}")

        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
