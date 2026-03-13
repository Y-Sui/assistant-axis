# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Degeneration Axis — a research project analyzing and mitigating LLM activation drift in agentic trajectories. Investigates whether LLM internal activations drift systematically over long task trajectories (50-100+ steps) and develops steering interventions using a contrastive judge-labeled approach.

Core method: compute a **contrastive drift axis** = mean(bad_activations) - mean(good_activations), where labels come from an LLM judge (GPT-5.2 via LiteLLM/OpenRouter), then apply inference-time activation capping to mitigate drift.

## Build & Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --extra dev

# Run all tests
uv run pytest src/tests/ -v

# Run a single test file
uv run pytest src/tests/test_axis.py -v

# Run a single test
uv run pytest src/tests/test_axis.py::test_compute_axis_basic -v
```

## Pipeline (6-step experiment)

```
vLLM server → trajectories → judge labels → activations → axis → steering
    0               1              2              3           4       5
```

```bash
# Step 0: Launch vLLM server (one script per model)
./pipeline/0_vllm_server_qwen3_4b.sh

# Step 1: Collect trajectories (mini-swe-agent, 16 runs per problem)
./pipeline/1_collect_trajectories.sh [WORKERS_PER_RUN] [PARALLEL_RUNS]

# Step 2: Judge each step (async, GPT-5.2)
python pipeline/2_judge_steps.py

# Step 3: Extract activations (single forward pass via vLLM)
python pipeline/3_activations.py

# Step 4: Compute contrastive axis
python pipeline/4_compute_axis.py --metric hallucination

# Step 5: Apply drift steering (activation capping)
python pipeline/5_drift_steering.py
```

## Architecture

### Source Library (`src/`)

- **`models.py`** — Model configs (Qwen3-4B, Qwen3.5-9B/27B/35B-A3B, Gemma-2-27B, Llama-3.3-70B) with layer counts, hidden dims, target layers
- **`contrastive_axis.py`** — Core algorithm: per-problem mean(bad) - mean(good), averaged across problems, L2-normalized per layer
- **`axis.py`** — Axis operations: projection, cosine similarity, save/load
- **`judge.py`** — LLM-as-Judge prompt construction and response parsing (hallucination + dishonesty labels: 0=good, 1=bad, 2=ambiguous)
- **`steering.py`** — `ActivationSteering` context manager supporting addition, ablation, mean_ablation, and capping interventions
- **`trajectory.py`** — Parses mini-swe-agent `.traj.json` files into conversation format
- **`pca.py`** — PCA utilities (legacy approach, replaced by contrastive method)

### Internals (`src/internals/`)

- **`model.py`** — `ProbingModel`: wraps HuggingFace model+tokenizer with multi-GPU sharding and layer access
- **`conversation.py`** — `ConversationEncoder`: chat template handling, token span detection per turn
- **`activations.py`** — `ActivationExtractor`: hook-based hidden state extraction
- **`spans.py`** — `SpanMapper`: token span→activation mapping with per-turn mean pooling

### Key Design Decisions

- **Single-pass activation extraction**: For causal models, one forward pass over the full conversation gives correct activations at all positions (no need for N incremental passes)
- **Per-problem weighting**: Contrastive axis averages per-problem diffs equally, controlling for task difficulty
- **Activation extraction via vLLM**: Pipeline step 3 uses `VllmHiddenStatesGenerator` from the `speculators` library (not HuggingFace forward hooks)

## Data Layout (gitignored)

```
data/drift/{model_name}/
├── raw/{instance_id}/run_00..15.traj.json
├── judgments/{instance_id}__run_{id}.judgments.json
├── activations/{instance_id}__run_{id}.pt    # (n_steps, n_layers, hidden_dim)
└── analysis/drift_axis.pt                     # (n_layers, hidden_dim)
```

## Configuration

- **`pipeline/configs/model_registry.json`** — LiteLLM model definitions for vLLM-hosted models
- **`pipeline/configs/swebench_qwen*.yaml`** — Mini-swe-agent configs (16 runs, temperature=0.7, step_limit=250)
