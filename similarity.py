"""
similarity.py
Computes an objective, deterministic NLP similarity score between each resume
and the job description using TF-IDF + cosine similarity.

This is intentionally kept separate from the LLM-based scoring in scorer.py.
The idea: TF-IDF gives a keyword/term-overlap-based number that can't
hallucinate, while the LLM gives a qualitative, reasoning-based number.
The final score in main.py blends both (see README for the rationale).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_scores(resume_texts: list[str], jd_text: str) -> list[float]:
    """
    Given a list of resume texts and the job description text, return a list
    of similarity scores (0-100) in the same order as resume_texts.
    """
    documents = [jd_text] + resume_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # capture skill phrases like "rest api"
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    # Scale from 0-1 to 0-100 for readability alongside the LLM score
    return [round(float(s) * 100, 2) for s in similarities]
