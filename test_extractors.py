"""
test_extractors.py
Covers: .txt / .pdf / .docx extraction, unsupported file types, empty files.
"""

import pytest

from extractors import extract_text_from_file


def test_extract_txt_returns_exact_content(temp_txt_resume):
    path, expected_content = temp_txt_resume
    result = extract_text_from_file(path)
    assert result == expected_content


def test_extract_pdf_finds_known_text(temp_pdf_resume):
    path, known_phrase = temp_pdf_resume
    result = extract_text_from_file(path)
    assert known_phrase in result


def test_extract_docx_finds_known_text(temp_docx_resume):
    path, known_phrase = temp_docx_resume
    result = extract_text_from_file(path)
    assert known_phrase in result


def test_unsupported_extension_raises_value_error(tmp_path):
    bad_file = tmp_path / "resume.rtf"
    bad_file.write_text("some content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text_from_file(str(bad_file))


def test_empty_txt_file_returns_empty_string(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    result = extract_text_from_file(str(empty_file))
    assert result == ""


def test_missing_file_raises_error(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.txt")
    with pytest.raises(FileNotFoundError):
        extract_text_from_file(missing_path)
