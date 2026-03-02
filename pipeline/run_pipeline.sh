#!/bin/bash
set -e

MODEL="Qwen/Qwen3-32B"
OUTPUT_DIR="../output/hallucination"
TRAJ_FILE="../data/degeneration/hallucination_trajectories.jsonl"

echo "=== Hallucination Axis Pipeline ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "\n=== Step 1: Generate ==="
uv run 1_generate.py \
  --model "$MODEL" \
  --trajectories_file "$TRAJ_FILE" \
  --output_dir "$OUTPUT_DIR/responses" \
  --do_sample

echo "\n=== Step 2: Activations ==="
uv run 2_activations.py \
  --model "$MODEL" \
  --responses_file "$OUTPUT_DIR/responses/responses.jsonl" \
  --output_file "$OUTPUT_DIR/activations.pt" \
  --layers all

echo "\n=== Step 3: Judge ==="
uv run 3_judge.py \
  --responses_file "$OUTPUT_DIR/responses/responses.jsonl" \
  --output_file "$OUTPUT_DIR/scores.json" \
  --judge_model gpt-4.1-mini \
  --scale_max 10

echo "\n=== Step 4: Vectors ==="
uv run 4_vectors.py \
  --activations_file "$OUTPUT_DIR/activations.pt" \
  --scores_file "$OUTPUT_DIR/scores.json" \
  --output_file "$OUTPUT_DIR/vectors.pt" \
  --clean_max 3 \
  --degen_min 7 \
  --min_count 30

echo "\n=== Step 5: Axis ==="
uv run 5_axis.py \
  --vectors_file "$OUTPUT_DIR/vectors.pt" \
  --output "$OUTPUT_DIR/axis.pt" \
  --model "$MODEL"

echo "\nDone. Axis: $OUTPUT_DIR/axis.pt"
