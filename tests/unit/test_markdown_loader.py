"""Tests for the Markdown Loader used by the I5 full-chain fixture."""

from __future__ import annotations

import pytest

from core.types import Document
from libs.loader.base_loader import BaseLoader
from libs.loader.markdown_loader import MarkdownLoader


def test_markdown_loader_is_a_base_loader() -> None:
    assert isinstance(MarkdownLoader(), BaseLoader)
    assert MarkdownLoader.supported_extensions == (".md", ".markdown")


def test_markdown_loader_reads_text_and_metadata(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\n## RAG\nBM25 and Dense.", encoding="utf-8")

    document = MarkdownLoader().load(str(path))

    assert isinstance(document, Document)
    assert document.id.startswith("notes_")
    assert document.text.startswith("# Notes")
    assert document.metadata["doc_type"] == "markdown"
    assert document.metadata["loader"] == "markdown"
    assert document.metadata["title"] == "Notes"
    assert document.metadata["heading_outline"] == [
        {"level": 1, "text": "Notes"},
        {"level": 2, "text": "RAG"},
    ]
    assert document.metadata["images"] == []


def test_markdown_loader_supports_utf8_bom_and_explicit_id(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("\ufeffcontent", encoding="utf-8")

    document = MarkdownLoader(doc_id="fixed").load(str(path))

    assert document.id == "fixed"
    assert document.text == "content"


@pytest.mark.parametrize("name", ["missing.md", "empty.md"])
def test_markdown_loader_rejects_missing_or_empty_file(tmp_path, name: str) -> None:
    path = tmp_path / name
    if name == "empty.md":
        path.write_text("  \n", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        MarkdownLoader().load(str(path))
