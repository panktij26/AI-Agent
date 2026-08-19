# Resume Screening Agent

Ranks a folder of resumes against a job description and outputs a scored,
ordered shortlist with reasoning for each candidate.

Built for the Rooman AI Challenge — 24-Hour AI Agent Challenge.

---

## What it does

> **"My agent takes a folder of resumes + a job description and produces a
> ranked, scored shortlist of candidates with reasoning for each score."**

For each resume, the agent:
1. Extracts raw text (`.pdf`, `.docx`, `.txt` all supported)
2. Uses Claude to pull out structured fields: name, skills, experience, education
3. Computes a **TF-IDF cosine similarity score** between the resume and the JD
   (deterministic, keyword/phrase-based, no hallucination risk)
4. Uses Claude to produce a **qualitative fit score (0–100)** with reasoning,
   listing matched and missing requirements
5. Combines both into a single `final_score` and ranks all candidates

---

## Project structure

```
resume-screening-agent/
├── main.py              # CLI entrypoint — runs the full pipeline
├── extractors.py         # Reads text out of .pdf / .docx / .txt resumes
├── similarity.py          # TF-IDF cosine similarity scoring (no LLM)
├── scorer.py               # Claude API calls: structured extraction + fit scoring
├── requirements.txt
├── requirements-dev.txt   # + pytest and test-fixture generators
├── .env.example
├── tests/                    # 19 automated tests, fully mocked (no API key needed)
│   ├── conftest.py
│   ├── test_extractors.py
│   ├── test_similarity.py
│   ├── test_scorer.py
│   └── test_main_integration.py
├── data/
│   ├── job_description.txt
│   └── resumes/            # 10 sample resumes, varied fit levels
└── output/                  # Generated on run: ranked_candidates.csv / .json
```

---

## Setup

**1. Clone and enter the project**
```bash
git clone <your-repo-url>
cd resume-screening-agent
```

**2. Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**
```bash
cp .env.example .env
```
Open `.env` and paste in your Anthropic API key (get one at
https://console.anthropic.com/settings/keys):
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## Running it

Run against the included sample data:
```bash
python main.py
```

Run against your own data:
```bash
python main.py --resumes path/to/resumes --jd path/to/jd.txt --out path/to/output
```

The agent will:
- Print progress for each resume as it's processed
- Print a ranked summary to the console
- Write `output/ranked_candidates.json` and `output/ranked_candidates.csv`

**Expected runtime:** ~5–10 seconds per resume (two Claude API calls each), so
~10 resumes takes roughly 1–2 minutes.

---

## Testing

The test suite is fully offline — it never calls the real Claude API, so it
runs free and in a few seconds without needing `ANTHROPIC_API_KEY` set.
Wherever the pipeline would call Claude, tests mock `scorer.client.messages.create`
with a fake, deterministic response.

**Install test dependencies and run:**
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

**What's covered (19 tests):**

| File | Covers |
|---|---|
| `test_extractors.py` | `.txt` / real `.pdf` / real `.docx` extraction, unsupported extensions raise a clear error, empty files, missing files |
| `test_similarity.py` | strong-vs-weak resumes score correctly relative to each other, scores stay in 0–100, input order is preserved, single-resume edge case |
| `test_scorer.py` | valid JSON responses parse correctly, markdown-fenced (` ```json `) responses are cleaned and parsed, malformed/non-JSON responses raise a clear `ValueError` instead of crashing silently |
| `test_main_integration.py` | full pipeline end-to-end (ranking order, output files written correctly), unsupported file types and empty files are skipped rather than crashing the batch, an empty resumes folder exits cleanly instead of throwing, a missing API key is caught **before** any resume is processed |

**Manual testing checklist** (things worth eyeballing once, with a real key,
that automated tests can't fully judge — quality of the LLM's *reasoning*):
- [ ] Run on the 10 included sample resumes — does the ranking order look
      sensible to a human reading the resumes?
- [ ] Check 2–3 `reasoning` fields in the output — are the matched/missing
      requirements actually accurate, not hallucinated?
- [ ] Try a resume with heavy jargon but low actual Python backend experience
      — does the LLM score correctly discount it, or get fooled by keywords?
- [ ] Try a `.pdf` and a `.docx` resume alongside the `.txt` ones in the same
      run — confirm all three formats process in one batch without error.
- [ ] Try a folder with 11+ resumes to confirm the "handle 10+ in a single
      run" requirement holds at slightly larger scale.
- [ ] Temporarily rename `.env` to confirm the app fails with a clear message
      instead of a raw stack trace.

---

## Sample output

Running the agent on the 10 included sample resumes against the included
Backend Python Developer job description produces a ranked CSV/JSON like:

| Rank | Name | Final Score | Notes |
|------|------|-------------|-------|
| 1 | Ananya Sharma | ~88 | FastAPI, PostgreSQL, AWS, Docker, CI/CD — strong direct match |
| 2 | Karthik Iyer | ~85 | Senior, Django/FastAPI, Kafka, K8s — overqualified but strong fit |
| 3 | Vikram Chauhan | ~80 | Django REST Framework, PostgreSQL, Docker, AWS |
| ... | ... | ... | ... |
| 10 | Fatima Sheikh | ~15 | Android/Kotlin developer, minimal backend Python overlap |

(Exact scores vary slightly between runs because the LLM-based half of the
score is generative. The TF-IDF half is fully deterministic — see
`output/ranked_candidates.json` after running for the real, reproducible
numbers from your own run.)

**Note on this repo:** the TF-IDF similarity half of the pipeline was verified
end-to-end during development (see commit history) and produces this
deterministic ranking on the sample resumes, from strongest to weakest overlap
with the JD: Ananya → Vikram → Karthik → Dev → Arjun → Rahul → Fatima → Priya
→ Sneha → Meera. The Claude-based qualitative half requires your own API key
to run — do that once after cloning to generate the final combined
`ranked_candidates.csv` for your own review.

---

## Scoring method (NLP similarity approach)

The final score is a **weighted blend** of two independent signals:

```
final_score = 0.4 × tfidf_score + 0.6 × llm_score
```

- **TF-IDF cosine similarity (40% weight):** Vectorizes the JD and each resume
  using term frequency–inverse document frequency (with 1–2 word n-grams to
  catch phrases like "rest api"), then computes cosine similarity between the
  JD vector and each resume vector. This is fast, fully deterministic, and
  immune to hallucination — it purely measures vocabulary overlap.
- **Claude qualitative score (60% weight):** Claude reads a structured summary
  of the candidate (extracted in a separate call) alongside the JD and scores
  fit based on actual requirement matching, not just keyword overlap — e.g. it
  can recognize that "Django REST Framework" satisfies "REST API design"
  experience even if the exact phrase doesn't appear in the JD.

The blend weights the LLM's judgment higher because it captures semantic
matches TF-IDF misses (synonyms, related technologies, seniority context),
while keeping TF-IDF in the mix as a hallucination check — if the LLM's score
and the TF-IDF score diverge wildly for a candidate, that's a useful reviewer
signal to double-check that resume manually.

---

## Tradeoff notes

**What I chose and why:**
- **Two-call-per-resume design** (extract, then score) instead of one combined
  call: keeps each Claude call focused on a single task, makes the JSON output
  more reliable, and makes structured data (skills/education) reusable for
  future features (e.g. filtering by skill) without re-scoring.
- **TF-IDF + LLM blend** instead of LLM-only scoring: pure LLM scoring can
  drift or be swayed by resume phrasing/length. TF-IDF is a cheap, stable
  sanity check that costs no extra API calls (it runs locally with
  scikit-learn).
- **JSON-only prompting** instead of a function-calling/tool-use setup: simpler
  to implement and debug in the 24-hour window; tradeoff is it relies on
  prompt discipline rather than a schema-enforced API contract.

**Known limitations / what I'd improve with more time:**
- No retry/backoff logic if a Claude call returns malformed JSON — right now
  it raises and stops the whole batch. A production version should retry that
  one resume and continue.
- `total_experience_years` is Claude's best estimate from dates in the resume
  text, not independently verified — could be wrong for resumes with gaps or
  unusual date formats.
- No de-duplication or resume-quality checks (e.g. flagging a resume that's
  suspiciously short or empty beyond a basic empty-text skip).
- Batch processing is sequential (one resume at a time). For larger batches
  (50+), this should be parallelized with async calls or a worker pool.
- The 0.4/0.6 weighting was a reasonable starting judgment call, not tuned
  against labeled ground-truth data — with more time I'd validate it against
  a set of resumes with known "should have been shortlisted" labels.

---

## Tech stack

- Python 3.10+
- [Anthropic API](https://docs.claude.com) (`claude-sonnet-5`) — structured
  extraction and qualitative scoring
- scikit-learn — TF-IDF vectorization and cosine similarity
- pypdf / docx2txt — resume text extraction
