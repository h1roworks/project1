"""Tests for G4's Dashboard ingestion orchestration service."""

from __future__ import annotations

from types import SimpleNamespace

from ingestion.pipeline import IngestionResult
from observability.dashboard.pages.ingestion_manager import _progress_percent
from observability.dashboard.services.ingestion_service import IngestionService


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def run(self, path, collection, on_progress=None):
        self.calls.append((path, collection))
        on_progress("load", 1, 1)
        return IngestionResult(path, collection, "hash", doc_id="doc-1", total_chunks=2)


class FakeDataService:
    def __init__(self) -> None:
        self.deleted = None

    def list_collections(self):
        return ["research"]

    def list_documents(self, collection=None):
        return ["doc"]

    def delete_document(self, source_path, collection):
        self.deleted = (source_path, collection)
        return "deleted"


def _service(tmp_path):
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection="research"))
    pipeline = FakePipeline()
    data = FakeDataService()
    service = IngestionService(
        settings=settings,
        pipeline=pipeline,
        data_service=data,
        upload_dir=tmp_path / "uploads",
    )
    return service, pipeline, data


def test_save_upload_sanitizes_filename_and_reuses_content_address(tmp_path) -> None:
    service, _, _ = _service(tmp_path)

    saved = service.save_upload("../../report.pdf", b"same content")

    assert saved.parent == tmp_path / "uploads"
    assert saved.name.endswith("_report.pdf")
    assert saved.read_bytes() == b"same content"
    assert service.save_upload("report.pdf", b"same content") == saved


def test_ingest_path_forwards_collection_and_progress_callback(tmp_path) -> None:
    service, pipeline, _ = _service(tmp_path)
    progress = []

    result = service.ingest_path(
        tmp_path / "report.pdf",
        "research",
        lambda stage, current, total: progress.append((stage, current, total)),
    )

    assert result.doc_id == "doc-1"
    assert pipeline.calls == [(str(tmp_path / "report.pdf"), "research")]
    assert progress == [("load", 1, 1)]


def test_ingestion_service_delegates_document_management(tmp_path) -> None:
    service, _, data = _service(tmp_path)

    assert service.list_collections() == ["research"]
    assert service.default_collection() == "research"
    assert service.list_documents("research") == ["doc"]
    assert service.delete_document("/docs/a.pdf", "research") == "deleted"
    assert data.deleted == ("/docs/a.pdf", "research")


def test_progress_percent_maps_all_pipeline_stages_to_one_bar() -> None:
    assert _progress_percent("load", 0, 1) == 0
    assert _progress_percent("load", 1, 1) == 20
    assert _progress_percent("transform", 1, 3) == 46
    assert _progress_percent("upsert", 1, 1) == 100
