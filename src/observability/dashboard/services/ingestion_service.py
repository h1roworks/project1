"""Dashboard-facing orchestration for uploading and ingesting a document."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from core.settings import Settings
from ingestion.pipeline import IngestionPipeline, IngestionResult
from observability.dashboard.services.config_service import ConfigService
from observability.dashboard.services.data_service import DataService

ProgressCallback = Callable[[str, int, int], None]


class IngestionService:
    """Keep Streamlit UI code separate from file and pipeline operations."""

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: IngestionPipeline | Any | None = None,
        data_service: DataService | Any | None = None,
        upload_dir: str | Path | None = None,
    ) -> None:
        self._settings = settings or ConfigService().get_settings()
        self._pipeline = pipeline or IngestionPipeline(settings=self._settings)
        self._data_service = data_service or DataService(settings=self._settings)
        project_root = Path(__file__).resolve().parents[4]
        self._upload_dir = Path(upload_dir or project_root / "data" / "uploads")

    def save_upload(self, filename: str, content: bytes) -> Path:
        """Persist an uploaded file under a safe, content-addressed name.

        The original browser filename is reduced to its basename, preventing a
        malicious filename from escaping ``data/uploads``.  The SHA256 prefix
        gives different file contents different source paths while identical
        re-uploads reuse a stable path for incremental ingestion.
        """
        safe_name = Path(filename).name
        if not safe_name:
            raise ValueError("上传文件缺少有效文件名")
        digest = hashlib.sha256(content).hexdigest()[:16]
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        target = self._upload_dir / f"{digest}_{safe_name}"
        target.write_bytes(content)
        return target

    def ingest_path(
        self,
        path: str | Path,
        collection: str,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Run the established ingestion pipeline for one local file."""
        return self._pipeline.run(str(path), collection=collection, on_progress=on_progress)

    def list_collections(self) -> list[str]:
        return self._data_service.list_collections()

    def default_collection(self) -> str:
        """Return the configured collection prefilled in the ingestion form."""
        return self._settings.vector_store.collection or "default"

    def list_documents(self, collection: str | None = None):
        return self._data_service.list_documents(collection)

    def delete_document(self, source_path: str, collection: str):
        return self._data_service.delete_document(source_path, collection)
