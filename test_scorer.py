"""
test_scorer.py
Covers: JSON parsing from Claude responses (including markdown-fenced
responses), and malformed-JSON error handling.

IMPORTANT: These tests mock `scorer.client.messages.create` so they never
make a real network call and never need ANTHROPIC_API_KEY to be set.
"""

from unittest.mock import patch

import pytest

import scorer
from tests.conftest import make_fake_claude_response


def test_extract_structured_data_parses_valid_json():
    fake_payload = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "total_experience_years": 5,
        "skills": ["Python", "FastAPI"],
        "education": ["B.Tech Computer Science"],
        "most_recent_title": "Backend Engineer",
        "summary": "Experienced backend engineer.",
    }
    fake_response = make_fake_claude_response(fake_payload)

    with patch.object(scorer.client.messages, "create", return_value=fake_response):
        result = scorer.extract_structured_data("some raw resume text")

    assert result["name"] == "Jane Doe"
    assert "Python" in result["skills"]


def test_extract_structured_data_handles_markdown_fenced_response():
    fake_payload = {"name": "Jane Doe", "skills": ["Python"]}

    class FakeBlock:
        type = "text"
        text = "```json\n" + __import__("json").dumps(fake_payload) + "\n```"

    class FakeResponse:
        content = [FakeBlock()]

    with patch.object(scorer.client.messages, "create", return_value=FakeResponse()):
        result = scorer.extract_structured_data("some raw resume text")

    assert result["name"] == "Jane Doe"


def test_extract_structured_data_raises_on_malformed_json():
    class FakeBlock:
        type = "text"
        text = "Sorry, here is the candidate info: Jane Doe, Python developer"  # not JSON

    class FakeResponse:
        content = [FakeBlock()]

    with patch.object(scorer.client.messages, "create", return_value=FakeResponse()):
        with pytest.raises(ValueError, match="Could not parse Claude's response as JSON"):
            scorer.extract_structured_data("some raw resume text")


def test_score_candidate_parses_valid_json():
    fake_payload = {
        "llm_score": 82,
        "matched_requirements": ["Python", "PostgreSQL"],
        "missing_requirements": ["AWS"],
        "reasoning": "Strong backend skills, missing cloud experience.",
    }
    fake_response = make_fake_claude_response(fake_payload)

    with patch.object(scorer.client.messages, "create", return_value=fake_response):
        result = scorer.score_candidate({"name": "Jane"}, "some jd text")

    assert result["llm_score"] == 82
    assert "AWS" in result["missing_requirements"]
