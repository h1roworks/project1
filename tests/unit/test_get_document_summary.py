"""Unit tests for the document-summary MCP tool."""

from __future__ import annotations

import pytest

from libs.vector_store.base_vector_store import VectorMatch
from mcp_server.tools.get_document_summary import (
    DocumentNotFoundError,
    DocumentSummaryTool,
)


def test_summary_uses_stored_metadata_and_reports_chunk_count() -> None:
    calls: list[str] = []

    def reader(doc_id: str) -> list[VectorMatch]:
        calls.append(doc_id)
        return [
            VectorMatch(
                id="doc-123_0000",
                score=0.0,
                text="RAG combines retrieval and generation.",
                metadata={
                    "title": "RAG Overview",
                    "summary": "An introduction to RAG.",
                    "tags": '["RAG", "retrieval"]',
                    "source_path": "docs/rag.md",
                },
            ),
            VectorMatch(id="doc-123_0001", score=0.0, text="more", metadata={}),
        ]

    response = DocumentSummaryTool(document_reader=reader)("  doc-123 ")

    assert calls == ["doc-123"]
    assert "## RAG Overview" in response["content"][0]["text"]
    assert response["structuredContent"] == {
        "document": {
            "doc_id": "doc-123",
            "title": "RAG Overview",
            "summary": "An introduction to RAG.",
            "tags": ["RAG", "retrieval"],
            "source": "docs/rag.md",
            "chunk_count": 2,
        }
    }


def test_summary_falls_back_to_source_and_chunk_text() -> None:
    response = DocumentSummaryTool(
        document_reader=lambda _doc_id: [
            VectorMatch(
                id="doc-1_0000",
                score=0.0,
                text="  First useful sentence from the document.  ",
                metadata={"source_path": "docs/guide.md"},
            )
        ]
    )("doc-1")

    document = response["structuredContent"]["document"]
    assert document["title"] == "guide"
    assert document["summary"] == "First useful sentence from the document."
    assert document["tags"] == []


def test_missing_document_raises_not_found_error() -> None:
    tool = DocumentSummaryTool(document_reader=lambda _doc_id: [])

    with pytest.raises(DocumentNotFoundError, match="missing-doc"):
        tool("missing-doc")


@pytest.mark.parametrize("doc_id", ["", "   ", None])
def test_invalid_document_id_is_rejected(doc_id) -> None:
    tool = DocumentSummaryTool(document_reader=lambda _doc_id: [])

    with pytest.raises(ValueError, match="doc_id"):
        tool(doc_id)
