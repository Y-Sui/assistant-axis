#!/usr/bin/env bash
# Launch vLLM server for gpt-oss-20B with OpenAI-compatible API
# No --reasoning-parser or --tool-call-parser needed: vLLM auto-enables
# harmony parsing for reasoning and tool calls when model_type == "gpt_oss".
set -euo pipefail

MODEL="openai/gpt-oss-20B"
PORT=8000
MAX_MODEL_LEN=131072

echo "Starting vLLM server for ${MODEL} on port ${PORT}..."
echo "Max model length: ${MAX_MODEL_LEN}"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90
