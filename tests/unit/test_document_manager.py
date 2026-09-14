"""Tests for G2's cross-store document lifecycle manager."""

from __future__ import annotations

from types import SimpleNamespace

from core.types import ChunkRecord
from ingestion.document_manager import DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import VectorMatch


SOURCE_A = "/documents/a.pdf"
SOURCE_B = "/documents/b.pdf"


class FakeChroma:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(collection="research")
        self.matches = [
            VectorMatch("a-1", 0.0, "first chunk", {"source_path": SOURCE_A, "doc_id": "doc-a", "collection": "research", "title": "Alpha"}),
            VectorMatch("a-2", 0.0, "second chunk", {"source_path": SOURCE_A, "doc_id": "doc-a", "collection": "research", "title": "Alpha"}),
            VectorMatch("b-1", 0.0, "other chunk", {"source_path": SOURCE_B, "doc_id": "doc-b", "collection": "research", "title": "Beta"}),
        ]
        self.deleted_filters: list[dict] = []

    def find_by_metadata(self, key: str, value: str) -> list[VectorMatch]:
        return [match for match in self.matches if match.metadata.get(key) == value]

    def delete_by_metadata(self, filters: dict) -> int:
        self.deleted_filters.append(filters)
        before = len(self.matches)
        self.matches = [
            match for match in self.matches
            if match.metadata.get("source_path") != filters["source_path"]
        ]
        return before - len(self.matches)

    def get_collection_stats(self) -> dict[str, int]:
        return {"database_size_bytes": 2048}


class FakeImages:
    def __init__(self) -> None:
        self.records = [
            {"image_id": "a-image", "collection": "research", "doc_hash": "hash-a"},
            {"image_id": "b-image", "collection": "research", "doc_hash": "hash-b"},
        ]

    def list_images(self, collection=None, doc_hash=None):
        return [
            record for record in self.records
            if (collection is None or record["collection"] == collection)
            and (doc_hash is None or record["doc_hash"] == doc_hash)
        ]

    def delete_images(self, collection, doc_hash=None):
        selected = self.list_images(collection, doc_hash)
        self.records = [record for record in self.records if record not in selected]
        return len(selected)


class FakeIntegrity:
    def __init__(self) -> None:
        self.records = [
            {"file_hash": "hash-a", "file_path": SOURCE_A, "status": "success", "processed_at": "2026-09-14", "chunk_count": 2},
            {"file_hash": "hash-b", "file_path": SOURCE_B, "status": "success", "processed_at": "2026-09-13", "chunk_count": 1},
            {"file_hash": "failed", "file_path": "/documents/failed.pdf", "status": "failed", "chunk_count": 0},
        ]

    def list_processed(self):
        return list(self.records)

    def remove_record(self, file_hash):
        before = len(self.records)
        self.records = [record for record in self.records if record["file_hash"] != file_hash]
        return before - len(self.records)


def _manager() -> tuple[DocumentManager, FakeChroma, FakeImages, FakeIntegrity, BM25Indexer]:
    chroma = FakeChroma()
    images = FakeImages()
    integrity = FakeIntegrity()
    bm25 = BM25Indexer()
    bm25.add([
        ChunkRecord("a-1", "", {"source_path": SOURCE_A}, sparse_vector={"alpha": 1}),
        ChunkRecord("a-2", "", {"source_path": SOURCE_A}, sparse_vector={"alpha": 1}),
        ChunkRecord("b-1", "", {"source_path": SOURCE_B}, sparse_vector={"beta": 1}),
    ])
    return DocumentManager(chroma, bm25, images, integrity), chroma, images, integrity, bm25


def test_list_documents_groups_history_with_current_store_counts() -> None:
    manager, *_ = _manager()

    documents = manager.list_documents()

    assert [document.doc_id for document in documents] == ["doc-a", "doc-b"]
    assert documents[0].title == "Alpha"
    assert documents[0].chunk_count == 2
    assert documents[0].image_count == 1


def test_list_documents_honors_explicit_collection_filter() -> None:
    manager, *_ = _manager()

    assert manager.list_documents(collection="other") == []


def test_get_document_detail_includes_chunks_and_images() -> None:
    manager, *_ = _manager()

    detail = manager.get_document_detail("doc-a")

    assert detail is not None
    assert [chunk.chunk_id for chunk in detail.chunks] == ["a-1", "a-2"]
    assert detail.chunks[0].text == "first chunk"
    assert [image["image_id"] for image in detail.images] == ["a-image"]
    assert manager.get_document_detail("missing") is None


def test_delete_document_coordinates_all_four_stores() -> None:
    manager, chroma, images, integrity, bm25 = _manager()

    result = manager.delete_document(SOURCE_A, "research")

    assert result.chunks_deleted == 2
    assert result.bm25_entries_deleted == 2
    assert result.images_deleted == 1
    assert result.integrity_records_deleted == 1
    assert chroma.deleted_filters == [{"source_path": SOURCE_A}]
    assert bm25.total_docs == 1
    assert images.list_images(collection="research") == [{"image_id": "b-image", "collection": "research", "doc_hash": "hash-b"}]
    assert [document.doc_id for document in manager.list_documents()] == ["doc-b"]


def test_collection_stats_uses_document_manager_counts() -> None:
    manager, *_ = _manager()

    stats = manager.get_collection_stats()

    assert stats.collection == "research"
    assert stats.document_count == 2
    assert stats.chunk_count == 3
    assert stats.image_count == 2
    assert stats.database_size_bytes == 2048
