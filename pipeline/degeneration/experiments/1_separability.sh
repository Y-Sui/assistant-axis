#!/usr/bin/env bash
# Experiment 1: Separability
#
# Runs the full generation pipeline (steps 1-5) then checks whether clean
# and degen activations are linearly separated by the computed axis.
#
# What we're testing:
#   axis = mean(clean) - mean(degen)
#   → project each activation onto the axis
#   → clean samples should have higher projection than degen samples
#
# Passing criteria: gap > 0 and spearman_r > 0.3 for all categories.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

MODEL=${MODEL:-"google/gemma-2-2b-it"}
OUT_DIR=${OUT_DIR:-"/home/yuansui/assistant-axis-outputs/exp1-separability"}
QCOUNT=${QCOUNT:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
TP_SIZE=${TP_SIZE:-4}
CATEGORIES_DIR="$ROOT_DIR/data/degeneration/categories"
QUESTIONS_FILE="$ROOT_DIR/data/extraction_questions.jsonl"

RESP_DIR="$OUT_DIR/responses"
ACT_DIR="$OUT_DIR/activations"
SCORES_DIR="$OUT_DIR/scores"
VEC_DIR="$OUT_DIR/vectors"
AXES_FILE="$OUT_DIR/axes.pt"

echo "=== Experiment 1: Separability ==="
echo "Model:  $MODEL"
echo "Output: $OUT_DIR"
echo ""

# Step 1: Generate paired clean/degen responses
echo "--- Step 1: Generate responses ---"
uv run pipeline/degeneration/1_generate.py \
  --model "$MODEL" \
  --questions_file "$QUESTIONS_FILE" \
  --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization "$GPU_MEM_UTIL" \
  --output_dir "$RESP_DIR" \
  --question_count "$QCOUNT" \
  --clean_max_tokens 128 \
  --degen_max_tokens 256

# Step 2: Extract activations
echo "--- Step 2: Extract activations ---"
uv run pipeline/2_activations.py \
  --model "$MODEL" \
  --responses_dir "$RESP_DIR" \
  --output_dir "$ACT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --tensor_parallel_size "$TP_SIZE"

# Step 3: Judge responses
echo "--- Step 3: Judge responses ---"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2; exit 1
fi
uv run pipeline/3_judge.py \
  --responses_dir "$RESP_DIR" \
  --roles_dir "$CATEGORIES_DIR" \
  --output_dir "$SCORES_DIR" \
  --judge_model "$JUDGE_MODEL"

# Step 4: Compute per-category vectors
echo "--- Step 4: Compute vectors ---"
uv run pipeline/degeneration/4_vectors.py \
  --activations_dir "$ACT_DIR" \
  --scores_dir "$SCORES_DIR" \
  --output_dir "$VEC_DIR" \
  --min_count 5

# Step 5: Compute axes
echo "--- Step 5: Compute axes ---"
uv run pipeline/degeneration/5_axes.py \
  --vectors_dir "$VEC_DIR" \
  --output "$AXES_FILE"

# Analysis: check linear separability
echo ""
echo "=== Separability Results ==="
uv run pipeline/degeneration/experiments/analysis/separability.py \
  --activations_dir "$ACT_DIR" \
  --scores_dir "$SCORES_DIR" \
  --axes_file "$AXES_FILE"
