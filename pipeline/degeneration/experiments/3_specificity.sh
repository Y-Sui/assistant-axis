#!/usr/bin/env bash
# Experiment 3: Specificity
#
# Tests whether each axis is specific to its own category or whether
# there's just one generic "degeneration" direction.
#
# For every pair (data_category, axis_category), computes the
# clean-degen projection gap. Prints a matrix:
#
#   rows  = which category's activations are projected
#   cols  = which axis is used for projection
#   entry = mean(clean) - mean(degen) projection gap
#
# Passing criteria: diagonal entries are the largest in each row.
# Run 1_separability.sh first to produce activations, scores, and axes.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

ACT_DIR=${ACT_DIR:-"/home/yuansui/assistant-axis-outputs/exp1-separability/activations"}
SCORES_DIR=${SCORES_DIR:-"/home/yuansui/assistant-axis-outputs/exp1-separability/scores"}
AXES_FILE=${AXES_FILE:-"/home/yuansui/assistant-axis-outputs/exp1-separability/axes.pt"}

echo "=== Experiment 3: Specificity ==="
echo "Activations: $ACT_DIR"
echo "Scores:      $SCORES_DIR"
echo "Axes:        $AXES_FILE"
echo ""
echo "=== Cross-category projection gap matrix (* = own-category axis) ==="
uv run pipeline/degeneration/experiments/analysis/specificity.py \
  --activations_dir "$ACT_DIR" \
  --scores_dir "$SCORES_DIR" \
  --axes_file "$AXES_FILE"
