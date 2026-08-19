"""
main.py
Resume Screening Agent — CLI entrypoint.

Pipeline (Input -> Think -> Act -> Output):
  1. Load the job description.
  2. Load every resume in the resumes folder (.pdf / .docx / .txt).
  3. THINK: extract structured fields from each resume via Claude.
  4. THINK: compute a TF-IDF keyword-similarity score (deterministic, no LLM).
  5. THINK: compute an LLM-based qualitative fit score + reasoning via Claude.
  6. ACT: combine both scores into one final_score per candidate.
  7. OUTPUT: rank candidates and write ranked_candidates.csv / .json.

Usage:
    python main.py --resumes data/resumes --jd data/job_description.txt --out output
"""

import argparse
import csv
import json
import os
import sys

from dotenv import load_dotenv

from extractors import extract_text_from_file
from similarity import compute_tfidf_scores
from scorer import extract_structured_data, score_candidate

# Weighting between the deterministic TF-IDF score and the LLM qualitative score.
# See README "Tradeoff Notes" for why this split was chosen.
TFIDF_WEIGHT = 0.4
LLM_WEIGHT = 0.6


def load_resumes(resumes_dir: str) -> list[dict]:
    """Read every supported resume file in the directory into memory."""
    supported_ext = (".pdf", ".docx", ".txt")
    resumes = []

    for filename in sorted(os.listdir(resumes_dir)):
        if filename.lower().endswith(supported_ext):
            path = os.path.join(resumes_dir, filename)
            try:
                text = extract_text_from_file(path)
                if text.strip():
                    resumes.append({"filename": filename, "text": text})
                else:
                    print(f"  [WARN] {filename} extracted empty text, skipping.")
            except Exception as e:
                print(f"  [WARN] Failed to read {filename}: {e}")

    return resumes


def run_pipeline(resumes_dir: str, jd_path: str, output_dir: str) -> list[dict]:
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print(f"Loading resumes from '{resumes_dir}'...")
    resumes = load_resumes(resumes_dir)
    print(f"  Loaded {len(resumes)} resume(s).\n")

    if len(resumes) == 0:
        print("No resumes found. Exiting.")
        sys.exit(1)

    print("Computing TF-IDF keyword similarity scores...")
    resume_texts = [r["text"] for r in resumes]
    tfidf_scores = compute_tfidf_scores(resume_texts, jd_text)
    print("  Done.\n")

    results = []
    for i, resume in enumerate(resumes):
        print(f"Processing {resume['filename']} ({i + 1}/{len(resumes)})...")

        print("  -> extracting structured fields via Claude...")
        structured = extract_structured_data(resume["text"])

        print("  -> scoring fit via Claude...")
        score_result = score_candidate(structured, jd_text)

        tfidf_score = tfidf_scores[i]
        llm_score = score_result.get("llm_score", 0)
        final_score = round(TFIDF_WEIGHT * tfidf_score + LLM_WEIGHT * llm_score, 2)

        results.append({
            "filename": resume["filename"],
            "name": structured.get("name"),
            "email": structured.get("email"),
            "total_experience_years": structured.get("total_experience_years"),
            "skills": structured.get("skills", []),
            "education": structured.get("education", []),
            "tfidf_score": tfidf_score,
            "llm_score": llm_score,
            "final_score": final_score,
            "matched_requirements": score_result.get("matched_requirements", []),
            "missing_requirements": score_result.get("missing_requirements", []),
            "reasoning": score_result.get("reasoning", ""),
        })
        print(f"  -> final_score = {final_score}\n")

    # Rank: highest final_score first
    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    write_outputs(results, output_dir)
    return results


def write_outputs(results: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "ranked_candidates.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(output_dir, "ranked_candidates.csv")
    fieldnames = [
        "rank", "filename", "name", "email", "total_experience_years",
        "tfidf_score", "llm_score", "final_score",
        "matched_requirements", "missing_requirements", "reasoning",
        "skills", "education",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            # Flatten lists into readable strings for CSV
            for key in ("matched_requirements", "missing_requirements", "skills", "education"):
                row[key] = "; ".join(row.get(key) or [])
            writer.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def print_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("RANKED SHORTLIST")
    print("=" * 70)
    for r in results:
        print(f"#{r['rank']:>2}  {r['final_score']:>6.2f}  {r['name'] or r['filename']}")
        print(f"      TF-IDF: {r['tfidf_score']:.2f}  |  LLM: {r['llm_score']}")
        print(f"      {r['reasoning']}")
        print("-" * 70)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--resumes", default="data/resumes", help="Folder of resumes (.pdf/.docx/.txt)")
    parser.add_argument("--jd", default="data/job_description.txt", help="Path to job description text file")
    parser.add_argument("--out", default="output", help="Output folder for ranked results")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    results = run_pipeline(args.resumes, args.jd, args.out)
    print_summary(results)


if __name__ == "__main__":
    main()
