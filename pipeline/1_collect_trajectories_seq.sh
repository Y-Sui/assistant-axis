#!/usr/bin/env bash
# Run mini-swe-agent on SWE-bench instances and collect trajectories.
# Runs 16 diverse trajectories per problem (temperature=0.7).
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
BASE_OUTPUT_DIR="${PROJECT_ROOT}/data/drift/qwen3-4b/raw"
WORKERS="${1:-4}"
NUM_RUNS=16

export LITELLM_MODEL_REGISTRY_PATH="${REGISTRY}"
export MSWEA_COST_TRACKING="ignore_errors"

echo "Collecting trajectories: ${NUM_RUNS} runs x ${WORKERS} workers"
echo "Config: ${CONFIG}"

for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
    RUN_ID=$(printf "run_%02d" "${RUN_IDX}")
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_ID}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=== ${RUN_ID} (${RUN_IDX}/${NUM_RUNS}) ==="
    echo "Output: ${OUTPUT_DIR}"

    mini-extra swebench \
        -c "${CONFIG}" \
        -o "${OUTPUT_DIR}" \
        -w "${WORKERS}" \
        --subset verified \
        --split test

    N_FILES=$(ls "${OUTPUT_DIR}"/*.traj.json 2>/dev/null | wc -l)
    echo "${RUN_ID}: ${N_FILES} trajectory files collected."
done

echo ""
echo "Done. All runs saved under ${BASE_OUTPUT_DIR}"
