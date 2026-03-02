# Hallucination Axis Pipeline

This directory now contains a degeneration-only 5-step pipeline focused on **hallucination**.

Hallucination score definition combines:
- unsupported/inaccurate claims
- confidence self-reinforcement across turns

Scoring is per assistant turn on `0..10`.

- clean: `<= 3`
- degenerated: `>= 7`
- ignored: `4..6`

Axis formula:

```text
axis = mean(clean_activations) - mean(degenerated_activations)
```

## Quick run

```bash
cd pipeline
./run_pipeline.sh
```

## Step-by-step

### 1) Generate trajectories

```bash
uv run 1_generate.py \
  --model Qwen/Qwen3-32B \
  --trajectories_file ../data/degeneration/hallucination_trajectories.jsonl \
  --output_dir ../output/hallucination/responses \
  --do_sample
```

Outputs `responses.jsonl` with one row per assistant turn:
- `sample_id`
- `trajectory_id`
- `turn_idx`
- `conversation`
- `assistant_text`

### 2) Extract activations

```bash
uv run 2_activations.py \
  --model Qwen/Qwen3-32B \
  --responses_file ../output/hallucination/responses/responses.jsonl \
  --output_file ../output/hallucination/activations.pt \
  --layers all
```

Output is `sample_id -> tensor[n_layers, hidden_dim]`.

### 3) Judge hallucination severity

```bash
uv run 3_judge.py \
  --responses_file ../output/hallucination/responses/responses.jsonl \
  --output_file ../output/hallucination/scores.json \
  --judge_model gpt-4.1-mini \
  --scale_max 10
```

Requires `OPENAI_API_KEY`.

### 4) Build clean/degenerated vectors

```bash
uv run 4_vectors.py \
  --activations_file ../output/hallucination/activations.pt \
  --scores_file ../output/hallucination/scores.json \
  --output_file ../output/hallucination/vectors.pt \
  --clean_max 3 \
  --degen_min 7 \
  --min_count 30
```

### 5) Compute axis

```bash
uv run 5_axis.py \
  --vectors_file ../output/hallucination/vectors.pt \
  --output ../output/hallucination/axis.pt \
  --model Qwen/Qwen3-32B
```

Saved artifact contains:
- `axis`
- metadata with thresholds, counts, timestamp, and source file
