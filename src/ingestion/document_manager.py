"""Document lifecycle operations coordinated across the local RAG stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from libs.vector_store.base_vector_store import VectorMatch


@dataclass(frozen=True)
class DocumentInfo:
    """A compact row used by the Dashboard document list."""

    doc_id: str
    source_path: str
    collection: str
    chunk_count: int
    image_count: int
    processed_at: str | None = None
    title: str = ""
    file_hash: str = ""


@dataclass(frozen=True)
class DocumentChunk:
    """A stored chunk and the metadata needed to inspect it."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentDetail(DocumentInfo):
    """A document list row enriched with its chunks and image records."""

    chunks: list[DocumentChunk] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class DeleteResult:
    """Counts returned after deleting one document across all stores."""

    source_path: str
    collection: str
    chunks_deleted: int = 0
    bm25_entries_deleted: int = 0
    images_deleted: int = 0
    integrity_records_deleted: int = 0


@dataclass(frozen=True)
class CollectionStats:
    """Dashboard-facing asset counts for one logical collection."""

    collection: str
    document_count: int
    chunk_count: int
    image_count: int
    database_size_bytes: int = 0


class DocumentManager:
    """Coordinate document reads and deletes across four local stores.

    The manager deliberately does not instantiate dependencies itself.  The
    application composes it with the Chroma collection and BM25 index belonging
    to the collection being managed; tests can supply small in-memory fakes.
    """

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
    ) -> None:
        self._chroma_store = chroma_store
        self._bm25_indexer = bm25_indexer
        self._image_storage = image_storage
        self._file_integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """List successfully ingested documents, newest first.

        FileIntegrity is the authoritative document history; Chroma provides
        the current chunk metadata and ImageStorage provides image counts.  A
        history row without chunks is retained so an interrupted/manual store
        operation remains visible to the operator.
        """
        target_collection = collection or self._default_collection()
        documents: list[DocumentInfo] = []
        for record in self._file_integrity.list_processed():
            if record.get("status") != "success":
                continue
            source_path = str(record.get("file_path") or "")
            if not source_path:
                continue
            matches = self._matches_for_source(source_path, target_collection)
            # A collection was explicitly requested: do not leak a document
            # from another logical collection merely because its history row
            # is shared by the local integrity database.
            if collection is not None and not matches:
                continue
            metadata = matches[0].metadata if matches else {}
            doc_id = str(metadata.get("doc_id") or self._fallback_doc_id(record, source_path))
            image_count = len(
                self._image_storage.list_images(
                    collection=target_collection,
                    doc_hash=record.get("file_hash"),
                )
            )
            documents.append(
                DocumentInfo(
                    doc_id=doc_id,
                    source_path=source_path,
                    collection=target_collection,
                    chunk_count=len(matches) if matches else int(record.get("chunk_count") or 0),
                    image_count=image_count,
                    processed_at=record.get("processed_at"),
                    title=str(metadata.get("title") or Path(source_path).stem),
                    file_hash=str(record.get("file_hash") or ""),
                )
            )
        return documents

    def get_document_detail(self, doc_id: str) -> DocumentDetail | None:
        """Return all chunks and image records for ``doc_id``, or ``None``."""
        document = next(
            (item for item in self.list_documents() if item.doc_id == doc_id),
            None,
        )
        if document is None:
            return None
        matches = self._matches_for_source(document.source_path, document.collection)
        chunks = [
            DocumentChunk(chunk_id=match.id, text=match.text, metadata=dict(match.metadata))
            for match in matches
            if match.metadata.get("doc_id", document.doc_id) == doc_id
        ]
        images = self._image_storage.list_images(
            collection=document.collection,
            doc_hash=document.file_hash or None,
        )
        return DocumentDetail(**document.__dict__, chunks=chunks, images=images)

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        """Delete a document from Chroma, BM25, images, and integrity history.

        Local stores do not share a transaction mechanism, so backend errors
        are intentionally allowed to propagate.  This avoids claiming a delete
        succeeded when only some stores were changed.
        """
        documents = [
            item
            for item in self.list_documents(collection)
            if item.source_path == source_path
        ]
        file_hashes = {item.file_hash for item in documents if item.file_hash}
        if not file_hashes:
            file_hashes = {
                str(record.get("file_hash"))
                for record in self._file_integrity.list_processed()
                if record.get("file_path") == source_path and record.get("file_hash")
            }

        chunks_deleted = self._chroma_store.delete_by_metadata({"source_path": source_path})
        bm25_deleted = self._bm25_indexer.remove_document(source_path)
        if getattr(self._bm25_indexer, "index_path", None) is not None:
            self._bm25_indexer.save()

        images_deleted = sum(
            self._image_storage.delete_images(collection, doc_hash=file_hash)
            for file_hash in file_hashes
        )
        integrity_deleted = sum(
            self._file_integrity.remove_record(file_hash) for file_hash in file_hashes
        )
        return DeleteResult(
            source_path=source_path,
            collection=collection,
            chunks_deleted=chunks_deleted,
            bm25_entries_deleted=bm25_deleted,
            images_deleted=images_deleted,
            integrity_records_deleted=integrity_deleted,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Return document, chunk, image and local vector-store size counts."""
        target_collection = collection or self._default_collection()
        documents = self.list_documents(target_collection)
        database_size = 0
        get_stats = getattr(self._chroma_store, "get_collection_stats", None)
        if callable(get_stats):
            database_size = int(get_stats().get("database_size_bytes", 0))
        return CollectionStats(
            collection=target_collection,
            document_count=len(documents),
            chunk_count=sum(item.chunk_count for item in documents),
            image_count=sum(item.image_count for item in documents),
            database_size_bytes=database_size,
        )

    def _matches_for_source(self, source_path: str, collection: str) -> list[VectorMatch]:
        matches = self._chroma_store.find_by_metadata("source_path", source_path)
        return [
            match
            for match in matches
            if match.metadata.get("collection", self._default_collection()) == collection
        ]

    def _default_collection(self) -> str:
        settings = getattr(self._chroma_store, "settings", None)
        return str(getattr(settings, "collection", "") or "default")

    @staticmethod
    def _fallback_doc_id(record: dict[str, Any], source_path: str) -> str:
        return str(record.get("file_hash") or Path(source_path).stem)
