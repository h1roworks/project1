"""I2: render every Dashboard route with deterministic in-memory data."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from ingestion.document_manager import DocumentChunk, DocumentDetail, DocumentInfo
from observability.dashboard.pages import (
    data_browser,
    evaluation_panel,
    ingestion_manager,
    ingestion_traces,
    overview,
    query_traces,
)
from observability.dashboard.services.config_service import ConfigService
from observability.dashboard.services.trace_service import (
    IngestionTrace,
    QueryTrace,
    TraceStage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_APP = PROJECT_ROOT / "src" / "observability" / "dashboard" / "app.py"


def _settings() -> Settings:
    """Use valid settings without reading the user's provider configuration."""
    return Settings(
        llm=LLMSettings(provider="test", model="test-llm"),
        embedding=EmbeddingSettings(provider="test", model="test-embedding"),
        vector_store=VectorStoreSettings(
            provider="chroma", collection="smoke", persist_dir="data/db/chroma"
        ),
        splitter=SplitterSettings(chunk_size=500, chunk_overlap=50),
        rerank=RerankSettings(enabled=False),
        evaluation=EvaluationSettings(provider="custom", metrics=["hit_rate"]),
    )


def _document() -> DocumentInfo:
    return DocumentInfo(
        doc_id="doc-smoke",
        source_path="fixtures/smoke.pdf",
        collection="smoke",
        chunk_count=1,
        image_count=0,
        processed_at="2026-09-14T00:00:00",
        title="Smoke fixture",
        file_hash="hash-smoke",
    )


class _ConfigService:
    def __init__(self, *_args, **_kwargs) -> None:
        self._settings = _settings()

    def get_settings(self) -> Settings:
        return self._settings

    def get_component_configs(self):
        return ConfigService(settings=self._settings).get_component_configs()


class _ChromaStore:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_collection_stats(self):
        return {
            "collection": "smoke",
            "documents": 1,
            "chunks": 1,
            "images": 0,
            "database_size_bytes": 1024,
        }


class _DataService:
    def list_collections(self):
        return ["smoke"]

    def list_documents(self, collection=None):
        return [_document()] if collection in (None, "smoke") else []

    def get_document_detail(self, doc_id):
        if doc_id != "doc-smoke":
            return None
        document = _document()
        return DocumentDetail(
            **document.__dict__,
            chunks=[
                DocumentChunk(
                    "chunk-smoke",
                    "A deterministic Dashboard smoke-test chunk.",
                    {"doc_id": "doc-smoke", "page_num": 1},
                )
            ],
        )


class _IngestionService(_DataService):
    def default_collection(self):
        return "smoke"


class _TraceService:
    def list_ingestion_traces(self):
        return [
            IngestionTrace(
                trace_id="ingestion-smoke",
                started_at="2026-09-14T00:00:00",
                finished_at="2026-09-14T00:00:01",
                total_elapsed_ms=1000,
                source_path="fixtures/smoke.pdf",
                collection="smoke",
                status="success",
                stages=[TraceStage("load", 100, provider="test")],
            )
        ]

    def list_query_traces(self, _query_filter=""):
        return [
            QueryTrace(
                trace_id="query-smoke",
                started_at="2026-09-14T00:00:00",
                finished_at="2026-09-14T00:00:01",
                total_elapsed_ms=1000,
                query="What is RAG?",
                stages=[
                    TraceStage(
                        "dense_retrieval",
                        100,
                        details={"results": [{"chunk_id": "chunk-smoke", "score": 0.9}]},
                    ),
                    TraceStage(
                        "sparse_retrieval",
                        80,
                        details={"results": [{"chunk_id": "chunk-smoke", "score": 0.8}]},
                    ),
                    TraceStage(
                        "rerank",
                        40,
                        details={
                            "before_ids": ["chunk-smoke"],
                            "after_ids": ["chunk-smoke"],
                        },
                    ),
                ],
            )
        ]


def _patch_dashboard_dependencies(monkeypatch) -> None:
    """Keep the smoke test independent from Chroma, Ollama, and log files."""
    monkeypatch.setattr(overview, "ConfigService", _ConfigService)
    monkeypatch.setattr(overview, "ChromaStore", _ChromaStore)
    monkeypatch.setattr(evaluation_panel, "ConfigService", _ConfigService)
    monkeypatch.setattr(data_browser, "DataService", _DataService)
    monkeypatch.setattr(ingestion_manager, "IngestionService", _IngestionService)
    monkeypatch.setattr(ingestion_traces, "TraceService", _TraceService)
    monkeypatch.setattr(query_traces, "TraceService", _TraceService)


@pytest.mark.e2e
def test_dashboard_all_six_pages_render_without_exceptions(monkeypatch) -> None:
    """Every registered route renders its data view without Python exceptions."""
    _patch_dashboard_dependencies(monkeypatch)

    app = AppTest.from_file(DASHBOARD_APP, default_timeout=10).run()
    assert not app.exception
    registered_pages = {
        info["url_pathname"]: page_hash
        for page_hash, info in app._registered_pages.items()
    }
    assert set(registered_pages) == {
        "",
        "data-browser",
        "ingestion-manager",
        "ingestion-traces",
        "query-traces",
        "evaluation",
    }

    # Callable st.Page entries have no script path, so Streamlit 1.63's
    # AppTest exposes no public switch_page target. Set the registered hash to
    # exercise each callable route through the real dashboard navigation.
    for route in (
        "data-browser",
        "ingestion-manager",
        "ingestion-traces",
        "query-traces",
        "evaluation",
    ):
        app._page_hash = registered_pages[route]
        app.run()
        assert not app.exception, f"Dashboard page {route!r} raised an exception"
        assert len(app.title) == 1
