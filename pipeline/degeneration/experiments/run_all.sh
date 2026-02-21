#!/usr/bin/env bash
# Run all 3 degeneration axis experiments in order.
#
# Experiment 1 (separability) must run first — it produces the activations,
# scores, and axes that experiments 2 and 3 depend on.
#
# Usage:
#   MODEL=google/gemma-2-2b-it \
#   OUT_DIR=/path/to/outputs \
#   OPENAI_API_KEY=sk-... \
#   bash pipeline/degeneration/experiments/run_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL=${MODEL:-"google/gemma-2-2b-it"}
export QCOUNT=${QCOUNT:-50}
export JUDGE_MODEL=${JUDGE_MODEL:-"gpt-4.1-mini"}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}

BASE_OUT=${OUT_DIR:-"/home/yuansui/assistant-axis-outputs/degeneration-experiments"}

# Experiment 1: separability (also builds all shared artifacts)
export OUT_DIR="$BASE_OUT/exp1"
bash "$SCRIPT_DIR/1_separability.sh"

# Experiment 2: steering (uses axes from exp1, fewer questions suffice)
export AXES_FILE="$BASE_OUT/exp1/axes.pt"
export OUT_DIR="$BASE_OUT/exp2"
QCOUNT=${STEERING_QCOUNT:-30} bash "$SCRIPT_DIR/2_steering.sh"

# Experiment 3: specificity (uses activations+scores+axes from exp1)
export ACT_DIR="$BASE_OUT/exp1/activations"
export SCORES_DIR="$BASE_OUT/exp1/scores"
export AXES_FILE="$BASE_OUT/exp1/axes.pt"
bash "$SCRIPT_DIR/3_specificity.sh"

echo ""
echo "All experiments complete. Results in: $BASE_OUT"
