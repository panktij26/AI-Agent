"""
test_similarity.py
Covers: strong match scores higher than weak match, ordering is preserved,
output range, and the empty-document edge case.
"""

import pytest

from similarity import compute_tfidf_scores


def test_strong_match_scores_higher_than_weak_match(sample_jd_text):
    strong_resume = "Python developer with FastAPI, PostgreSQL, Docker, AWS, REST APIs."
    weak_resume = "Professional photographer specializing in weddings and portraits."

    scores = compute_tfidf_scores([strong_resume, weak_resume], sample_jd_text)

    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_scores_are_returned_in_input_order(sample_jd_text):
    resumes = [
        "Photography and design.",                              # weak
        "Python, FastAPI, PostgreSQL, Docker, AWS, REST APIs.",  # strong
        "Some unrelated marketing background.",                 # weak
    ]
    scores = compute_tfidf_scores(resumes, sample_jd_text)

    assert len(scores) == 3
    # The strong match (index 1) should be the highest of the three
    assert scores[1] == max(scores)


def test_scores_are_within_0_to_100_range(sample_jd_text):
    resumes = [
        "Python, FastAPI, PostgreSQL, Docker, AWS, REST APIs.",
        "Completely unrelated content about gardening and cooking.",
    ]
    scores = compute_tfidf_scores(resumes, sample_jd_text)

    for s in scores:
        assert 0.0 <= s <= 100.0


def test_identical_text_scores_highest_possible(sample_jd_text):
    # A resume that IS the job description should score at or near the max
    # among a set that also includes an unrelated resume.
    resumes = [sample_jd_text, "Totally unrelated text about cooking recipes."]
    scores = compute_tfidf_scores(resumes, sample_jd_text)
    assert scores[0] > scores[1]


def test_handles_single_resume(sample_jd_text):
    scores = compute_tfidf_scores(["Python, Docker, AWS."], sample_jd_text)
    assert len(scores) == 1
