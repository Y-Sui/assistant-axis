#!/usr/bin/env bash
# Launch vLLM server for Qwen-3-4B with OpenAI-compatible API
set -euo pipefail

MODEL="Qwen/Qwen3-4B"
PORT=8000
MAX_MODEL_LEN=32768

echo "Starting vLLM server for ${MODEL} on port ${PORT}..."
echo "Max model length: ${MAX_MODEL_LEN}"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --trust-remote-code
