"""Tests for generation compatibility helpers."""

from assistant_axis.generation import format_conversation, supports_system_prompt


class DummyTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path


class TestSupportsSystemPrompt:
    def test_gemma2_no_system(self):
        tok = DummyTokenizer("google/gemma-2-2b-it")
        assert supports_system_prompt(tok) is False

    def test_qwen_supports_system(self):
        tok = DummyTokenizer("Qwen/Qwen3-32B")
        assert supports_system_prompt(tok) is True


class TestFormatConversation:
    def test_with_system(self):
        tok = DummyTokenizer("Qwen/Qwen3-32B")
        out = format_conversation("You are helpful.", "Hello", tok)
        assert out == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    def test_without_system(self):
        tok = DummyTokenizer("google/gemma-2-2b-it")
        out = format_conversation("You are helpful.", "Hello", tok)
        assert out[0]["role"] == "user"
        assert "You are helpful." in out[0]["content"]
        assert "Hello" in out[0]["content"]

    def test_no_instruction(self):
        tok = DummyTokenizer("Qwen/Qwen3-32B")
        out = format_conversation(None, "Hello", tok)
        assert out == [{"role": "user", "content": "Hello"}]
