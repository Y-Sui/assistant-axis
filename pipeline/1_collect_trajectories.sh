#!/usr/bin/env bash
# Run mini-swe-agent on SWE-bench instances and collect trajectories.
# Runs 16 diverse trajectories per problem (temperature=0.7).
# Uses GNU Parallel to run multiple collection jobs simultaneously.
#
# Prerequisites:
#   - vLLM server running (see 0_vllm_server.sh)
#   - mini-swe-agent installed: uv tool install mini-swe-agent
#   - GNU Parallel installed: apt install parallel / brew install parallel
#
# Usage:
#   ./pipeline/1_collect_trajectories.sh [WORKERS_PER_RUN] [PARALLEL_RUNS]
#
#   WORKERS_PER_RUN: mini-swe-agent concurrency within each run (default: 4)
#   PARALLEL_RUNS:   number of runs to execute simultaneously (default: 4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${SCRIPT_DIR}/configs/swebench_qwen4b.yaml"
REGISTRY="${SCRIPT_DIR}/configs/model_registry.json"
BASE_OUTPUT_DIR="${PROJECT_ROOT}/data/drift/qwen3-4b/raw"
WORKERS_PER_RUN="${1:-4}"
PARALLEL_RUNS="${2:-4}"
NUM_RUNS=16

if ! command -v parallel &>/dev/null; then
    echo "ERROR: GNU Parallel is required. Install with: apt install parallel"
    exit 1
fi

export LITELLM_MODEL_REGISTRY_PATH="${REGISTRY}"
export MSWEA_COST_TRACKING="ignore_errors"
export CONFIG WORKERS_PER_RUN BASE_OUTPUT_DIR

echo "Collecting trajectories: ${NUM_RUNS} runs"
echo "  ${PARALLEL_RUNS} parallel runs x ${WORKERS_PER_RUN} workers each"
echo "Config: ${CONFIG}"
echo "Output: ${BASE_OUTPUT_DIR}"

# Create all output directories upfront
for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
    mkdir -p "${BASE_OUTPUT_DIR}/$(printf 'run_%02d' "${RUN_IDX}")"
done

run_single() {
    local RUN_IDX=$1
    local RUN_ID
    RUN_ID=$(printf "run_%02d" "${RUN_IDX}")
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_ID}"

    echo "[${RUN_ID}] Starting..."
    mini-extra swebench \
        -c "${CONFIG}" \
        -o "${OUTPUT_DIR}" \
        -w "${WORKERS_PER_RUN}" \
        --subset verified \
        --split test

    local N_FILES
    N_FILES=$(find "${OUTPUT_DIR}" -name "*.traj.json" 2>/dev/null | wc -l)
    echo "[${RUN_ID}] Done: ${N_FILES} trajectory files."
}
export -f run_single

seq 0 $((NUM_RUNS - 1)) | parallel -j "${PARALLEL_RUNS}" --ungroup run_single {}

# Reorganize in place: run-based -> problem-based layout
#   run_00/{instance_id}.traj.json -> {instance_id}/run_00.traj.json
echo ""
echo "Reorganizing into problem-based layout..."
for RUN_DIR in "${BASE_OUTPUT_DIR}"/run_*; do
    [ -d "${RUN_DIR}" ] || continue
    RUN_ID=$(basename "${RUN_DIR}")
    for TRAJ_FILE in "${RUN_DIR}"/*.traj.json; do
        [ -f "${TRAJ_FILE}" ] || continue
        INSTANCE_ID=$(basename "${TRAJ_FILE}" .traj.json)
        DEST_DIR="${BASE_OUTPUT_DIR}/${INSTANCE_ID}"
        mkdir -p "${DEST_DIR}"
        mv "${TRAJ_FILE}" "${DEST_DIR}/${RUN_ID}.traj.json"
    done
    rmdir "${RUN_DIR}" 2>/dev/null || true
done

echo ""
TOTAL=$(find "${BASE_OUTPUT_DIR}" -name "*.traj.json" 2>/dev/null | wc -l)
N_PROBLEMS=$(find "${BASE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Done. ${TOTAL} trajectory files across ${N_PROBLEMS} problems."
echo "Layout: ${BASE_OUTPUT_DIR}/{instance_id}/run_XX.traj.json"
