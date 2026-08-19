"""
test_main_integration.py
Runs the full pipeline end-to-end against small fixture data, with the Claude
calls mocked out (deterministic fake scores instead of a real API call).

Covers:
  - happy path: correct files written, correct ranking order
  - unsupported file types and empty files are skipped, not crashed on
  - an empty resumes folder exits cleanly instead of crashing
  - missing ANTHROPIC_API_KEY is caught before any work starts
"""

import json
import os

import pytest

import main


def fake_extract_structured_data(resume_text):
    """Deterministic stand-in for scorer.extract_structured_data."""
    is_strong = "FastAPI" in resume_text
    return {
        "name": "Strong Candidate" if is_strong else "Weak Candidate",
        "email": "test@example.com",
        "total_experience_years": 5 if is_strong else 1,
        "skills": ["Python", "FastAPI"] if is_strong else ["Photoshop"],
        "education": ["B.Tech"],
        "most_recent_title": "Engineer",
        "summary": "n/a",
        "_strong": is_strong,  # test-only signal, not part of the real schema
    }


def fake_score_candidate(structured_data, jd_text):
    """Deterministic stand-in for scorer.score_candidate."""
    is_strong = structured_data.get("_strong")
    return {
        "llm_score": 90 if is_strong else 15,
        "matched_requirements": ["Python"] if is_strong else [],
        "missing_requirements": [] if is_strong else ["Python", "REST APIs"],
        "reasoning": "Strong match." if is_strong else "Weak match.",
    }


@pytest.fixture(autouse=True)
def mock_claude_calls(monkeypatch):
    """Applied to every test in this file: no real API calls are made."""
    monkeypatch.setattr(main, "extract_structured_data", fake_extract_structured_data)
    monkeypatch.setattr(main, "score_candidate", fake_score_candidate)


def test_pipeline_end_to_end_writes_outputs_and_ranks_correctly(
    temp_resumes_dir, temp_jd_file, tmp_path
):
    output_dir = str(tmp_path / "output")

    results = main.run_pipeline(temp_resumes_dir, temp_jd_file, output_dir)

    # Only strong_fit.txt and weak_fit.txt should be scored:
    # empty.txt has no text, notes.rtf is an unsupported extension.
    assert len(results) == 2

    # Strong fit should rank first
    assert results[0]["name"] == "Strong Candidate"
    assert results[0]["rank"] == 1
    assert results[1]["name"] == "Weak Candidate"
    assert results[1]["rank"] == 2

    # Ranking is by descending final_score
    assert results[0]["final_score"] >= results[1]["final_score"]

    # Output files actually exist and are well-formed
    json_path = os.path.join(output_dir, "ranked_candidates.json")
    csv_path = os.path.join(output_dir, "ranked_candidates.csv")
    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)

    with open(json_path) as f:
        loaded = json.load(f)
    assert len(loaded) == 2


def test_unsupported_and_empty_files_are_skipped_not_crashed_on(temp_resumes_dir):
    resumes = main.load_resumes(temp_resumes_dir)
    filenames = {r["filename"] for r in resumes}

    assert "strong_fit.txt" in filenames
    assert "weak_fit.txt" in filenames
    assert "notes.rtf" not in filenames   # unsupported extension, ignored
    assert "empty.txt" not in filenames   # empty text, skipped with a warning


def test_empty_resumes_folder_exits_cleanly(tmp_path, temp_jd_file):
    empty_dir = tmp_path / "no_resumes_here"
    empty_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main.run_pipeline(str(empty_dir), temp_jd_file, str(tmp_path / "output"))

    assert exc_info.value.code == 1


def test_missing_api_key_exits_before_any_work(monkeypatch, tmp_path, temp_resumes_dir, temp_jd_file):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here, so load_dotenv() finds nothing
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--resumes", temp_resumes_dir, "--jd", temp_jd_file, "--out", str(tmp_path / "output")],
    )

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
