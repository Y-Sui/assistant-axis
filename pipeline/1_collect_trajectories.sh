#!/usr/bin/env bash
# Run mini-swe-agent on SWE-bench instances and collect trajectories.
#
# Prerequisites:
#   - vLLM server running (see 0_vllm_server.sh)
#   - mini-swe-agent installed: uv tool install mini-swe-agent
#
# Usage:
#   ./pipeline/1_collect_trajectories.sh [WORKERS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${SCRIPT_DIR}/configs/swebench_qwen4b.yaml"
REGISTRY="${SCRIPT_DIR}/configs/model_registry.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/drift/qwen3-4b/raw"
WORKERS="${1:-4}"

mkdir -p "${OUTPUT_DIR}"

export LITELLM_MODEL_REGISTRY_PATH="${REGISTRY}"
export MSWEA_COST_TRACKING="ignore_errors"

echo "Collecting trajectories (${WORKERS} workers)..."
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

mini-extra swebench \
    -c "${CONFIG}" \
    -o "${OUTPUT_DIR}" \
    -w "${WORKERS}" \
    --subset verified \
    --split test

echo "Done. Trajectory files saved to ${OUTPUT_DIR}"
ls "${OUTPUT_DIR}"/*.traj.json 2>/dev/null | wc -l
echo "trajectory files collected."
