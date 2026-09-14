"""Read-only data access used by the Dashboard data-browser page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.settings import Settings
from ingestion.document_manager import DocumentDetail, DocumentInfo, DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


class DataService:
    """Compose local stores into a small, read-only Dashboard API.

    Page code should use this service instead of knowing how Chroma, SQLite
    image mappings, BM25, and ingestion history are wired together.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        document_manager: DocumentManager | None = None,
        chroma_store: Any | None = None,
        image_storage: Any | None = None,
        file_integrity: Any | None = None,
        bm25_indexer: Any | None = None,
    ) -> None:
        settings = settings or ConfigService().get_settings()
        self._settings = settings
        self._chroma_store = chroma_store or ChromaStore(settings.vector_store)
        project_root = Path(__file__).resolve().parents[4]
        collection = settings.vector_store.collection or "default"
        self._image_storage = image_storage or ImageStorage(
            images_dir=project_root / "data" / "images",
            db_path=project_root / "data" / "db" / "image_index.db",
        )
        self._file_integrity = file_integrity or SQLiteIntegrityChecker(
            project_root / "data" / "db" / "ingestion_history.db"
        )
        self._bm25_indexer = bm25_indexer or BM25Indexer(
            project_root / "data" / "db" / "bm25" / f"{collection}.pkl"
        )
        self._document_manager = document_manager or DocumentManager(
            self._chroma_store,
            self._bm25_indexer,
            self._image_storage,
            self._file_integrity,
        )

    def list_collections(self) -> list[str]:
        """Return known logical collections for the page's filter control."""
        names = {self._settings.vector_store.collection or "default"}
        names.update(item.collection for item in self._document_manager.list_documents())
        names.update(
            str(record["collection"])
            for record in self._image_storage.list_images()
            if record.get("collection")
        )
        return sorted(names)

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """Return Dashboard document rows, optionally scoped to a collection."""
        return self._document_manager.list_documents(collection)

    def get_document_detail(self, doc_id: str) -> DocumentDetail | None:
        """Return a selected document with its chunks and image records."""
        return self._document_manager.get_document_detail(doc_id)

    def get_chunks(self, doc_id: str) -> list[Any]:
        """Read chunks directly by document metadata for focused browse views."""
        getter = getattr(self._chroma_store, "get_by_metadata", None)
        if getter is None:
            getter = self._chroma_store.find_by_metadata
        return getter("doc_id", doc_id)

    def read_image(self, image_id: str) -> bytes | None:
        """Return raw image bytes suitable for ``st.image``, if still present."""
        return self._image_storage.read_image(image_id)
