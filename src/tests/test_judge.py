"""Tests for judge prompt building and response parsing."""

import pytest

from src.judge import build_judge_prompt, parse_judge_response


class TestBuildJudgePrompt:
    """Tests for build_judge_prompt function."""

    def test_returns_message_list(self):
        """Should return a list of system + user messages."""
        messages = build_judge_prompt(
            problem_statement="Fix the bug in foo.py",
            conversation_prefix=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please fix the bug."},
            ],
            step_index=0,
            assistant_content="I'll look at foo.py.",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_includes_problem_statement(self):
        """Problem statement should appear in the user message."""
        messages = build_judge_prompt(
            problem_statement="TypeError in module X",
            conversation_prefix=[],
            step_index=0,
            assistant_content="Let me check.",
        )

        assert "TypeError in module X" in messages[1]["content"]

    def test_includes_step_index(self):
        """Step index should appear in the user message."""
        messages = build_judge_prompt(
            problem_statement="Fix bug",
            conversation_prefix=[],
            step_index=7,
            assistant_content="Done.",
        )

        assert "Step 7" in messages[1]["content"]

    def test_includes_assistant_content(self):
        """The assistant content to evaluate should be in the user message."""
        messages = build_judge_prompt(
            problem_statement="Fix bug",
            conversation_prefix=[],
            step_index=0,
            assistant_content="I found the issue in line 42.",
        )

        assert "I found the issue in line 42." in messages[1]["content"]


class TestParseJudgeResponse:
    """Tests for parse_judge_response function."""

    def test_valid_json(self):
        """Should parse well-formed JSON response."""
        response = '{"hallucination": {"label": 0, "explanation": "No issues"}, "dishonesty": {"label": 1, "explanation": "Claimed success"}}'
        result = parse_judge_response(response)

        assert result is not None
        assert result["hallucination"]["label"] == 0
        assert result["dishonesty"]["label"] == 1

    def test_json_in_code_block(self):
        """Should handle JSON wrapped in markdown code fences."""
        response = '```json\n{"hallucination": {"label": 0, "explanation": "ok"}, "dishonesty": {"label": 0, "explanation": "ok"}}\n```'
        result = parse_judge_response(response)

        assert result is not None
        assert result["hallucination"]["label"] == 0

    def test_json_with_surrounding_text(self):
        """Should extract JSON from surrounding text."""
        response = 'Here is my evaluation:\n{"hallucination": {"label": 2, "explanation": "unclear"}, "dishonesty": {"label": 0, "explanation": "fine"}}\nThat is my assessment.'
        result = parse_judge_response(response)

        assert result is not None
        assert result["hallucination"]["label"] == 2

    def test_missing_metric_returns_none(self):
        """Should return None if a required metric is missing."""
        response = '{"hallucination": {"label": 0, "explanation": "ok"}}'
        result = parse_judge_response(response)

        assert result is None

    def test_invalid_label_returns_none(self):
        """Should return None if label is not 0, 1, or 2."""
        response = '{"hallucination": {"label": 5, "explanation": "ok"}, "dishonesty": {"label": 0, "explanation": "ok"}}'
        result = parse_judge_response(response)

        assert result is None

    def test_garbage_input_returns_none(self):
        """Should return None for unparseable input."""
        result = parse_judge_response("This is not JSON at all")

        assert result is None

    def test_empty_input_returns_none(self):
        """Should return None for empty input."""
        result = parse_judge_response("")

        assert result is None

    def test_missing_explanation_gets_default(self):
        """Should add empty explanation if missing."""
        response = '{"hallucination": {"label": 0}, "dishonesty": {"label": 1}}'
        result = parse_judge_response(response)

        assert result is not None
        assert result["hallucination"]["explanation"] == ""
        assert result["dishonesty"]["explanation"] == ""
