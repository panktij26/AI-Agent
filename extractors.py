"""
extractors.py
Handles reading resume text out of .pdf, .docx, and .txt files.
"""

import os


def extract_text_from_file(path: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        return _extract_txt(path)
    elif ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".docx":
        return _extract_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for {path}. "
            "Supported types: .txt, .pdf, .docx"
        )


def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _extract_docx(path: str) -> str:
    import docx2txt

    text = docx2txt.process(path)
    return text or ""
