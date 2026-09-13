"""Unit tests for the local collection-browsing MCP tool."""

from __future__ import annotations

from mcp_server.tools.list_collections import ListCollectionsTool


def test_list_collections_returns_sorted_folders_and_file_statistics(tmp_path) -> None:
    documents = tmp_path / "documents"
    research = documents / "research"
    course_notes = documents / "course-notes"
    nested = research / "papers"
    nested.mkdir(parents=True)
    course_notes.mkdir()
    (research / "overview.md").write_text("RAG", encoding="utf-8")
    (nested / "paper.pdf").write_bytes(b"PDF")
    (course_notes / "week-1.md").write_text("BM25", encoding="utf-8")
    (documents / "not-a-collection.md").write_text("ignore", encoding="utf-8")

    response = ListCollectionsTool(documents)()

    assert "`course-notes`：1 个文档" in response["content"][0]["text"]
    assert response["structuredContent"] == {
        "collections": [
            {"name": "course-notes", "document_count": 1, "total_size_bytes": 4},
            {"name": "research", "document_count": 2, "total_size_bytes": 6},
        ]
    }


def test_list_collections_returns_friendly_response_when_directory_is_missing(tmp_path) -> None:
    response = ListCollectionsTool(tmp_path / "missing")()

    assert "暂未发现知识库集合" in response["content"][0]["text"]
    assert response["structuredContent"] == {"collections": []}
