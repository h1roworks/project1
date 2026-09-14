"""Tests for the G3 Dashboard data-service adapter."""

from __future__ import annotations

from types import SimpleNamespace

from ingestion.document_manager import DocumentChunk, DocumentDetail, DocumentInfo
from observability.dashboard.services.data_service import DataService


class FakeDocumentManager:
    def __init__(self) -> None:
        self.document = DocumentInfo(
            doc_id="doc-1",
            source_path="/docs/one.pdf",
            collection="research",
            chunk_count=1,
            image_count=1,
            title="One",
            file_hash="hash-1",
        )

    def list_documents(self, collection=None):
        return [self.document] if collection in (None, "research") else []

    def get_document_detail(self, doc_id):
        if doc_id != "doc-1":
            return None
        return DocumentDetail(
            **self.document.__dict__,
            chunks=[DocumentChunk("chunk-1", "text", {"doc_id": "doc-1"})],
            images=[{"image_id": "image-1", "collection": "research"}],
        )


class FakeChroma:
    def get_by_metadata(self, key, value):
        return [{"key": key, "value": value}]


class FakeImages:
    def list_images(self):
        return [{"image_id": "image-1", "collection": "images"}]

    def read_image(self, image_id):
        return b"image-data" if image_id == "image-1" else None


def _service() -> DataService:
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection="research"))
    return DataService(
        settings=settings,
        document_manager=FakeDocumentManager(),
        chroma_store=FakeChroma(),
        image_storage=FakeImages(),
        file_integrity=object(),
        bm25_indexer=object(),
    )


def test_data_service_lists_known_collections_and_documents() -> None:
    service = _service()

    assert service.list_collections() == ["images", "research"]
    assert [document.doc_id for document in service.list_documents("research")] == ["doc-1"]


def test_data_service_reads_chunks_detail_and_image_bytes() -> None:
    service = _service()

    assert service.get_chunks("doc-1") == [{"key": "doc_id", "value": "doc-1"}]
    assert service.get_document_detail("doc-1").chunks[0].chunk_id == "chunk-1"
    assert service.read_image("image-1") == b"image-data"
    assert service.read_image("missing") is None
