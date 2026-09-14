"""F5: IngestionPipeline 的进度回调契约测试。"""

from __future__ import annotations

from pathlib import Path

from core.types import Document
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch, VectorRecord


class ProgressLoader(BaseLoader):
    loader_name = "progress_fake"
    supported_extensions = (".txt",)

    def load(self, path: str) -> Document:
        return Document(
            id="progress-doc",
            text="Progress callbacks make an ingestion UI responsive.",
            metadata={"source_path": path, "loader": self.loader_name, "images": []},
        )


class ProgressEmbedding:
    def embed(self, texts, trace=None):
        return [[float(len(text)), 1.0] for text in texts]


class ProgressVectorStore(BaseVectorStore):
    provider = "progress_fake"

    def upsert(self, records: list[VectorRecord], trace=None) -> int:
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None) -> list[VectorMatch]:
        return []

    def get_by_ids(self, ids, trace=None) -> list[VectorMatch]:
        return []


def build_pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        loaders=[ProgressLoader()],
        embedding=ProgressEmbedding(),
        vector_store=ProgressVectorStore(),
        integrity=SQLiteIntegrityChecker(db_path=tmp_path / "db" / "ingestion.db"),
        image_storage=ImageStorage(
            images_dir=tmp_path / "images", db_path=tmp_path / "db" / "images.db"
        ),
        bm25_indexer=BM25Indexer(index_path=tmp_path / "bm25" / "index.pkl"),
        trace_writer=lambda _trace: None,
    )


def write_source(tmp_path: Path) -> str:
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    return str(source)


def test_pipeline_reports_all_stages_with_valid_progress_values(tmp_path: Path) -> None:
    events: list[tuple[str, int, int]] = []

    result = build_pipeline(tmp_path).run(
        write_source(tmp_path),
        on_progress=lambda stage, current, total: events.append((stage, current, total)),
    )

    assert result.error is None
    assert {stage for stage, _, _ in events} == {
        "load", "split", "transform", "embed", "upsert",
    }
    assert events[0] == ("load", 0, 1)
    assert ("load", 1, 1) in events
    assert all(isinstance(current, int) and isinstance(total, int) for _, current, total in events)
    assert all(0 <= current <= total for _, current, total in events)


def test_progress_callback_failure_does_not_fail_ingestion(tmp_path: Path) -> None:
    def broken_callback(_stage: str, _current: int, _total: int) -> None:
        raise RuntimeError("dashboard disconnected")

    result = build_pipeline(tmp_path).run(write_source(tmp_path), on_progress=broken_callback)

    assert result.error is None
    assert result.total_chunks == 1
