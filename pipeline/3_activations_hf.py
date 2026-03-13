"""Single-pass per-step activation extraction via transformers forward hooks.

Drop-in replacement for 3_activations.py that uses HuggingFace transformers
+ register_forward_hook instead of speculators/vLLM. Works with any model
transformers can load, including Qwen3.5 hybrid (DeltaNet + Attention) models.

Usage:
    python pipeline/3_activations_hf.py [--input-dir DIR] [--output-dir DIR] [--model MODEL]
"""

import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from src.internals import ProbingModel, ConversationEncoder
from src.models import get_config
from src.trajectory import parse_trajectory


def extract_hidden_states(
    model: ProbingModel,
    input_ids: list[int],
    layer_indices: list[int],
) -> torch.Tensor:
    """Single forward pass with hooks to capture hidden states at target layers.

    Args:
        model: ProbingModel instance.
        input_ids: Token IDs for the full conversation.
        layer_indices: Which layers to extract.

    Returns:
        Tensor of shape (n_layers, seq_len, hidden_dim) in bfloat16 on CPU.
    """
    layers = model.get_layers()
    activations = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            # Immediately move to CPU to save GPU memory
            activations[layer_idx] = act[0].cpu()
        return hook_fn

    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)

    try:
        with torch.inference_mode():
            model.model(ids_tensor)
    finally:
        for h in handles:
            h.remove()

    # Stack in layer order -> (n_layers, seq_len, hidden_dim)
    stacked = torch.stack([activations[i] for i in layer_indices])
    return stacked.to(torch.bfloat16)


def extract_trajectory_activations(
    trajectory: dict,
    model: ProbingModel,
    encoder: ConversationEncoder,
    layer_indices: list[int],
    max_length: int = 32768,
) -> dict | None:
    """Extract per-step activations for a trajectory via single-pass forward hooks.

    Algorithm:
    1. Tokenize full conversation once
    2. Build turn spans to identify each assistant turn's token range
    3. Single forward pass with hooks
    4. For each assistant span: slice + mean-pool -> (n_layers, hidden_dim)
    5. Stack all turns -> (n_steps, n_layers, hidden_dim)

    Args:
        trajectory: Parsed trajectory dict with "conversation" key.
        model: ProbingModel instance.
        encoder: ConversationEncoder for the model.
        layer_indices: Which layers to extract.
        max_length: Maximum sequence length.

    Returns:
        Dict with instance_id, n_steps, activations tensor,
        and step_metadata list, or None if extraction fails.
    """
    conversation = trajectory["conversation"]
    instance_id = trajectory["instance_id"]

    # Filter to standard roles only (exit, etc. would break chat templates)
    conversation = [m for m in conversation if m["role"] in ("system", "user", "assistant")]

    # Tokenize full conversation and get turn spans
    full_ids, spans = encoder.build_turn_spans(conversation, enable_thinking=False)

    if len(full_ids) > max_length:
        print(f"  {instance_id}: {len(full_ids)} tokens > {max_length}, truncating")
        full_ids = full_ids[:max_length]

    # Find assistant turn spans
    assistant_spans = [s for s in spans if s["role"] == "assistant"]
    if not assistant_spans:
        return None

    # Single forward pass with hooks
    try:
        activations = extract_hidden_states(model, full_ids, layer_indices)
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

        all_activations.append(mean_act.to(torch.bfloat16))

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
    parser = argparse.ArgumentParser(
        description="Single-pass per-step activation extraction (transformers forward hooks)"
    )
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
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on. Defaults to cuda:0. "
             "Use cuda:N for a specific GPU, or 'auto' to shard across all available GPUs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get model config
    config = get_config(args.model)
    total_layers = config["total_layers"]
    layer_indices = list(range(total_layers - 10, total_layers))
    print(f"Model: {args.model}")
    print(f"Extracting layers: {layer_indices}")

    # Load model via ProbingModel
    device = None if args.device == "auto" else args.device
    print(f"Loading model on {args.device}...")
    model = ProbingModel(args.model, device=device)
    encoder = ConversationEncoder(model.tokenizer, args.model)

    print(f"Model loaded on: {model.device}")
    print(f"Layers found: {len(model.get_layers())}")

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
            traj, model, encoder, layer_indices, args.max_length,
        )

        if result is not None:
            result["run_id"] = run_id
            torch.save(result, output_path)
            tqdm.write(f"  Saved: {result['activations'].shape} -> {output_path.name}")
        else:
            tqdm.write(f"  No activations for {instance_id}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
