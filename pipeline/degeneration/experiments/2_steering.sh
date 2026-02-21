#!/usr/bin/env bash
# Experiment 2: Steering
#
# Tests whether adding the axis to activations at inference time
# actually suppresses degeneration.
#
# Protocol:
#   - Load the model (HuggingFace, for hook support)
#   - For each category, generate responses using the DEGEN system prompt:
#       baseline: no intervention          → should score low (degenerate)
#       steered:  axis added at best layer → should score higher
#   - Judge both with the category eval_prompt
#   - Print mean score comparison
#
# Passing criteria: steered_mean > baseline_mean for all categories.
# Run 1_separability.sh first to produce the axes file.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

MODEL=${MODEL:-"google/gemma-2-2b-it"}
AXES_FILE=${AXES_FILE:-"/home/yuansui/assistant-axis-outputs/exp1-separability/axes.pt"}
OUT_DIR=${OUT_DIR:-"/home/yuansui/assistant-axis-outputs/exp2-steering"}
QCOUNT=${QCOUNT:-30}
COEFF=${COEFF:-20.0}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini"}
CATEGORIES_DIR="$ROOT_DIR/data/degeneration/categories"
QUESTIONS_FILE="$ROOT_DIR/data/extraction_questions.jsonl"

PAIRS_DIR="$OUT_DIR/pairs"

echo "=== Experiment 2: Steering ==="
echo "Model:     $MODEL"
echo "Axes file: $AXES_FILE"
echo "Coeff:     $COEFF"
echo "Output:    $OUT_DIR"
echo ""

# Generate baseline and steered response pairs
echo "--- Generating baseline and steered responses ---"
uv run pipeline/degeneration/experiments/analysis/steering_generate.py \
  --model "$MODEL" \
  --axes_file "$AXES_FILE" \
  --categories_dir "$CATEGORIES_DIR" \
  --questions_file "$QUESTIONS_FILE" \
  --output_dir "$PAIRS_DIR" \
  --question_count "$QCOUNT" \
  --coeff "$COEFF"

# Judge both and compare
echo ""
echo "--- Judging baseline vs steered ---"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2; exit 1
fi

echo ""
echo "=== Steering Results (baseline → steered, higher is better) ==="
uv run pipeline/degeneration/experiments/analysis/steering_judge.py \
  --pairs_dir "$PAIRS_DIR" \
  --categories_dir "$CATEGORIES_DIR" \
  --judge_model "$JUDGE_MODEL"
