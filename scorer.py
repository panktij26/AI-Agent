"""
scorer.py
Uses the Claude API to:
  1. Extract structured fields from raw resume text (skills, experience, education).
  2. Produce a qualitative fit score (0-100) with reasoning against the job description.

Both calls ask Claude to return ONLY valid JSON so the output can be parsed reliably.
"""

import json
import os

import anthropic

MODEL = "claude-sonnet-5"  # see README for how to change this

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


EXTRACTION_SYSTEM_PROMPT = """You are a precise resume-parsing assistant.
Given raw resume text, extract structured information.

Respond with ONLY a valid JSON object, no markdown fences, no preamble, no
explanation. If a field is not present in the resume, use null or an empty list.

JSON schema:
{
  "name": string,
  "email": string or null,
  "total_experience_years": number (your best estimate from dates listed),
  "skills": [string],
  "education": [string],
  "most_recent_title": string or null,
  "summary": string (1-2 sentence neutral summary of the candidate)
}
"""

SCORING_SYSTEM_PROMPT = """You are an experienced technical recruiter.
You will be given a Job Description and a structured summary of one candidate.

Score how well this candidate fits the job on a scale of 0-100, where:
  - 90-100: Excellent fit, meets or exceeds nearly all requirements
  - 70-89: Strong fit, meets most requirements with minor gaps
  - 50-69: Moderate fit, meets some requirements, notable gaps
  - 30-49: Weak fit, few requirements met
  - 0-29: Poor fit, largely unrelated background

Base your score only on evidence in the candidate summary. Do not invent skills
or experience that aren't stated. Be specific in your reasoning about which
required skills are present and which are missing.

Respond with ONLY a valid JSON object, no markdown fences, no preamble.

JSON schema:
{
  "llm_score": number (0-100),
  "matched_requirements": [string],
  "missing_requirements": [string],
  "reasoning": string (2-4 sentences explaining the score)
}
"""


def _call_claude_json(system_prompt: str, user_content: str) -> dict:
    """Call Claude and parse the response as JSON, with basic cleanup."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Defensive cleanup in case the model wraps output in markdown fences anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse Claude's response as JSON.\nRaw response:\n{raw_text}"
        ) from e


def extract_structured_data(resume_text: str) -> dict:
    """Step 1: turn raw resume text into structured fields via Claude."""
    return _call_claude_json(EXTRACTION_SYSTEM_PROMPT, resume_text)


def score_candidate(structured_data: dict, jd_text: str) -> dict:
    """Step 2: ask Claude for a qualitative fit score + reasoning."""
    user_content = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE SUMMARY (structured):\n{json.dumps(structured_data, indent=2)}"
    )
    return _call_claude_json(SCORING_SYSTEM_PROMPT, user_content)
