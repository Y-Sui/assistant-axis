"""Tests for judge score parsing."""

from assistant_axis.judge import parse_judge_score


def test_parse_judge_score_0_to_10_valid():
    assert parse_judge_score("7", min_score=0, max_score=10) == 7
    assert parse_judge_score("score: 10", min_score=0, max_score=10) == 10
    assert parse_judge_score("0\n", min_score=0, max_score=10) == 0


def test_parse_judge_score_invalid():
    assert parse_judge_score("11", min_score=0, max_score=10) is None
    assert parse_judge_score("-1", min_score=0, max_score=10) is None
    assert parse_judge_score("no score", min_score=0, max_score=10) is None
