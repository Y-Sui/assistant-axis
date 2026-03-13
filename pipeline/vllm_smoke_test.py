#!/usr/bin/env python3
"""Quick smoke test for vLLM server (OpenAI-compatible API)."""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"
MODEL = None  # auto-detected from server

def check_health():
    """Check if the server is up."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"[Health] status={r.status_code}")
        return r.status_code == 200
    except requests.ConnectionError:
        print("[Health] Server not reachable")
        return False

def detect_model():
    """Detect the model served by vLLM."""
    global MODEL
    r = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    r.raise_for_status()
    models = r.json()
    MODEL = models["data"][0]["id"]
    print(f"[Models] Detected: {MODEL}")
    return models

def test_chat_completion(use_reasoning=False):
    """Test /v1/chat/completions endpoint."""
    label = "reasoning" if use_reasoning else "standard"
    print(f"\n{'='*60}")
    print(f"[Chat/{label}] Sending request...")

    messages = [
        {"role": "user", "content": "What is 27 * 43? Think step by step."}
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.6,
    }
    if not use_reasoning:
        # Disable thinking for the non-reasoning call
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.time()
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()

    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})

    # Print reasoning content if present
    reasoning = msg.get("reasoning_content")
    if reasoning:
        print(f"[Reasoning]\n{reasoning}")
        print("-" * 40)

    print(f"[Response]\n{msg['content']}")
    print(f"\n[Stats] {elapsed:.2f}s | "
          f"prompt_tokens={usage.get('prompt_tokens', '?')} "
          f"completion_tokens={usage.get('completion_tokens', '?')} "
          f"total_tokens={usage.get('total_tokens', '?')}")
    return data

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

def test_tool_calling(enable_thinking=False):
    """Test tool calling with and without thinking mode."""
    label = "thinking+tools" if enable_thinking else "tools-only"
    print(f"\n{'='*60}")
    print(f"[ToolCall/{label}] Sending request...")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "List the files in /testbed"}],
        "tools": [BASH_TOOL],
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    t0 = time.time()
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()

    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})
    tool_calls = msg.get("tool_calls") or []
    reasoning = msg.get("reasoning")

    if reasoning:
        print(f"[Reasoning] {reasoning[:200]}...")
    if msg.get("content"):
        print(f"[Content] {msg['content'][:200]}")

    print(f"[ToolCalls] count={len(tool_calls)}")
    for i, tc in enumerate(tool_calls):
        func = tc.get("function", {})
        print(f"  [{i}] name={func.get('name')}, args={func.get('arguments')}")

    print(f"[Stats] {elapsed:.2f}s | "
          f"prompt_tokens={usage.get('prompt_tokens', '?')} "
          f"completion_tokens={usage.get('completion_tokens', '?')} "
          f"finish_reason={choice.get('finish_reason', '?')}")

    # Validate
    assert len(tool_calls) > 0, f"FAIL: no tool_calls in response for {label}"
    assert tool_calls[0]["function"]["name"] == "bash", "FAIL: tool name is not 'bash'"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert "command" in args, "FAIL: missing 'command' in tool call arguments"
    print(f"[ToolCall/{label}] PASSED")
    return data

def test_streaming():
    """Test streaming chat completion."""
    print(f"\n{'='*60}")
    print("[Stream] Sending streaming request...")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=120)
    r.raise_for_status()

    collected = []
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            chunk_str = line[len("data: "):]
            if chunk_str.strip() == "[DONE]":
                break
            chunk = json.loads(chunk_str)
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if token:
                collected.append(token)
                print(token, end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n[Stream] Done in {elapsed:.2f}s, {len(collected)} chunks")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Smoke Test")
    print("=" * 60)

    if not check_health():
        print("Server is not healthy. Exiting.")
        sys.exit(1)

    detect_model()
    test_chat_completion(use_reasoning=True)   # with thinking
    test_chat_completion(use_reasoning=False)   # without thinking
    test_tool_calling(enable_thinking=False)    # tool calling without thinking
    test_tool_calling(enable_thinking=True)     # tool calling with thinking
    test_streaming()

    print(f"\n{'='*60}")
    print("All tests passed!")