#!/usr/bin/env bash
# Launch vLLM server for Qwen-3.5-9B with OpenAI-compatible API
set -euo pipefail

MODEL="Qwen/Qwen3.5-9B"
PORT=8000
MAX_MODEL_LEN=262144

echo "Starting vLLM server for ${MODEL} on port ${PORT}..."
echo "Max model length: ${MAX_MODEL_LEN}"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
