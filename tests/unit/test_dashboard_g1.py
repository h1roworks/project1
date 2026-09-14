"""Focused tests for the Phase G1 Dashboard support code."""

from __future__ import annotations

from types import SimpleNamespace

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


def _settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="ollama", model="qwen"),
        embedding=EmbeddingSettings(provider="ollama", model="nomic", dimensions=768),
        vector_store=VectorStoreSettings(collection="demo", persist_dir="unused"),
        splitter=SplitterSettings(strategy="recursive", chunk_size=500, chunk_overlap=50),
        rerank=RerankSettings(enabled=False),
        evaluation=EvaluationSettings(provider="custom", metrics=["hit_rate"]),
    )


def test_config_service_formats_component_cards() -> None:
    cards = ConfigService(settings=_settings()).get_component_configs()

    assert [card.name for card in cards] == ["LLM", "Embedding", "Splitter", "Reranker", "Evaluator"]
    assert cards[1].details == "dimensions=768, batch_size=32"
    assert cards[3].provider == "disabled"
    assert cards[4].details == "hit_rate"


def test_chroma_collection_stats_derives_document_and_image_counts(tmp_path) -> None:
    store = ChromaStore.__new__(ChromaStore)
    store.settings = SimpleNamespace(collection="demo", persist_dir=str(tmp_path))
    store._collection = SimpleNamespace(
        get=lambda include: {
            "metadatas": [
                {"source_path": "a.pdf", "images": '[{"id": "image-1"}]'},
                {"source_path": "a.pdf", "images": '[{"id": "image-1"}, {"id": "image-2"}]'},
                {"source_path": "b.pdf"},
            ]
        },
        count=lambda: 3,
    )
    (tmp_path / "chroma.sqlite3").write_bytes(b"1234")

    assert store.get_collection_stats() == {
        "collection": "demo",
        "documents": 2,
        "chunks": 3,
        "images": 2,
        "database_size_bytes": 4,
    }
