#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present (for OPENAI_API_KEY)
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

MODEL=${MODEL:-"google/gemma-2-2b-it"}
# Store large outputs (pt/jsonl) outside the repo by default
OUT_DIR=${OUT_DIR:-"/home/yuansui/assistant-axis-outputs/degeneration-quick"}
QCOUNT=${QCOUNT:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
TP_SIZE=${TP_SIZE:-4}

RESP_DIR="$OUT_DIR/responses"
ACT_DIR="$OUT_DIR/activations"
SCORES_DIR="$OUT_DIR/scores"
VEC_DIR="$OUT_DIR/vectors"
AXES_FILE="$OUT_DIR/axes.pt"

echo "Model: $MODEL"
echo "Output: $OUT_DIR"

# 1) Generate paired responses (good vs degen)
uv run pipeline/degeneration/1_generate.py \
  --model "$MODEL" \
  --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization "$GPU_MEM_UTIL" \
  --output_dir "$RESP_DIR" \
  --question_count "$QCOUNT" \
  --clean_max_tokens 128 \
  --degen_max_tokens 256

# 2) Extract activations
uv run pipeline/2_activations.py \
  --model "$MODEL" \
  --responses_dir "$RESP_DIR" \
  --output_dir "$ACT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --tensor_parallel_size "$TP_SIZE"

# 3) Judge scoring
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it to run judging." >&2
  exit 1
fi

uv run pipeline/3_judge.py \
  --responses_dir "$RESP_DIR" \
  --roles_dir data/degeneration/categories \
  --output_dir "$SCORES_DIR" \
  --judge_model "$JUDGE_MODEL"

# 4) Compute vectors
uv run pipeline/degeneration/4_vectors.py \
  --activations_dir "$ACT_DIR" \
  --scores_dir "$SCORES_DIR" \
  --output_dir "$VEC_DIR" \
  --min_count 5

# 5) Compute axes
uv run pipeline/degeneration/5_axes.py \
  --vectors_dir "$VEC_DIR" \
  --output "$AXES_FILE"

echo "Done. Axes at: $AXES_FILE"
