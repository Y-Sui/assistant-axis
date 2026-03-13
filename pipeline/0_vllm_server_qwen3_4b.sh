#!/usr/bin/env bash
# Launch vLLM server for Qwen-3-4B with OpenAI-compatible API
set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL="Qwen/Qwen3-4B"
PORT=8000
MAX_MODEL_LEN=32768

echo "Starting vLLM server for ${MODEL} on port ${PORT}..."
echo "Max model length: ${MAX_MODEL_LEN}"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
