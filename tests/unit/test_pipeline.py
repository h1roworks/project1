"""C14: IngestionPipeline 单元测试（Fake 组件，注入模式）。

用 Fake Loader / Fake Embedding / 内存 VectorStore 验证编排逻辑本身：
- 串行编排：load → split → transform → embed → upsert 全链路产出统计结果
- 增量跳过：同文件二次摄取 returned skipped=True（零成本）
- on_progress 进度回调：各 stage 均被调用，参数为 (str, int, int)
- 失败兜底：Loader 抛异常 → integrity 记录 failed + result.error 非空（不重抛）
- 文件不存在 → FileNotFoundError
- collection 默认取自 settings.vector_store.collection
- 空文档 → 0 chunk 也能正常完成（success）
- 图片落盘：Document.images 引用存在 → ImageStorage 保存
"""

import pytest

from core.settings import EmbeddingSettings, LLMSettings, Settings, VectorStoreSettings
from core.types import Document, ImageRef
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch, VectorRecord


# ---------- Fake 组件 ----------


class FakeEmbedding:
    """确定性向量：文本越长向量值越大，便于后续排序断言。"""

    def __init__(self) -> None:
        self.traces = []

    def embed(self, texts, trace=None):
        self.traces.append(trace)
        return [[float(len(t)), 0.5, 1.0] for t in texts]


class FakeLoader(BaseLoader):
    loader_name = "fake"
    supported_extensions = (".md", ".txt")

    def __init__(
        self,
        doc_id: str = "fake_doc",
        text: str = "RAG retrieval system with BM25 keyword matching\n\n"
        "Dense embedding captures semantic similarity.",
        images: list[ImageRef] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._doc_id = doc_id
        self._text = text
        self._images = images or []
        self._raise_error = raise_error

    def load(self, path: str) -> Document:
        if self._raise_error is not None:
            raise self._raise_error
        return Document(
            id=self._doc_id,
            text=self._text,
            metadata={
                "source_path": str(path),
                "doc_type": "md",
                "loader": self.loader_name,
                "images": self._images,
            },
        )


class FakeVectorStore(BaseVectorStore):
    """内存向量库：记录写入，按 filters 过滤后返回（MVP 不做相似度排序）。"""

    provider = "fake"

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        self.traces = []

    def upsert(self, records, trace=None) -> int:
        self.traces.append(trace)
        for record in records:
            self._records[record.id] = record
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None) -> list[VectorMatch]:
        results = []
        for record in self._records.values():
            if filters and any(
                record.metadata.get(k) != v for k, v in filters.items()
            ):
                continue
            results.append(
                VectorMatch(
                    id=record.id,
                    score=1.0,
                    text=record.text,
                    metadata=record.metadata,
                )
            )
        return results[:top_k]

    def get_by_ids(self, ids, trace=None) -> list[VectorMatch]:
        return [
            VectorMatch(
                id=record.id,
                score=0.0,
                text=record.text,
                metadata=record.metadata,
            )
            for chunk_id in ids
            if (record := self._records.get(chunk_id)) is not None
        ]


# ---------- 测试工具 ----------


def _write(path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _build_pipeline(tmp_path, loader=None, vector_store=None, trace_writer=lambda _trace: None) -> IngestionPipeline:
    """注入模式装配 pipeline：所有外部依赖落在 tmp_path，不污染真实 data/。"""
    return IngestionPipeline(
        loaders=[loader or FakeLoader()],
        embedding=FakeEmbedding(),
        vector_store=vector_store or FakeVectorStore(),
        integrity=SQLiteIntegrityChecker(db_path=tmp_path / "db" / "ingestion.db"),
        image_storage=ImageStorage(
            images_dir=tmp_path / "images", db_path=tmp_path / "db" / "img.db"
        ),
        bm25_indexer=BM25Indexer(index_path=tmp_path / "bm25" / "index.pkl"),
        trace_writer=trace_writer,
    )


# ---------- 测试用例 ----------


def test_full_pipeline_returns_result(tmp_path) -> None:
    pipe = _build_pipeline(tmp_path)
    src = _write(tmp_path / "doc.md", "hello world content\n\nmore content here")
    result = pipe.run(src, collection="test")

    assert result.skipped is False
    assert result.error is None
    assert result.doc_id == "fake_doc"
    assert result.collection == "test"
    assert result.total_chunks > 0
    assert set(result.stages) == {"load", "split", "transform", "embed", "upsert"}
    # 稠密向量全部写入向量库（FakeEmbedding 必有向量）
    assert result.stages["upsert"]["vector_store_count"] == result.total_chunks
    # BM25 索引同步更新
    assert result.stages["upsert"]["bm25_count"] == result.total_chunks


def test_incremental_skip_second_run(tmp_path) -> None:
    pipe = _build_pipeline(tmp_path)
    src = _write(tmp_path / "doc.md", "some stable content")
    first = pipe.run(src)
    assert first.skipped is False

    second = pipe.run(src)
    assert second.skipped is True
    assert second.total_chunks == 0
    assert second.stages == {}


def test_on_progress_reports_all_stages(tmp_path) -> None:
    pipe = _build_pipeline(tmp_path)
    src = _write(tmp_path / "doc.md", "progress callback content")
    calls: list[tuple[str, int, int]] = []
    pipe.run(src, on_progress=lambda stage, cur, total: calls.append((stage, cur, total)))

    stages_seen = {c[0] for c in calls}
    assert stages_seen == {"load", "split", "transform", "embed", "upsert"}
    for stage, current, total in calls:
        assert isinstance(stage, str)
        assert isinstance(current, int) and isinstance(total, int)
        assert 0 <= current <= total


def test_failure_marks_failed_and_returns_error(tmp_path) -> None:
    pipe = _build_pipeline(tmp_path, loader=FakeLoader(raise_error=ValueError("parse failed")))
    src = _write(tmp_path / "doc.md", "content")
    result = pipe.run(src)

    assert result.error is not None
    assert "parse failed" in result.error
    assert result.skipped is False
    assert result.total_chunks == 0

    file_hash = pipe._integrity.compute_sha256(src)
    assert pipe._integrity.get_status(file_hash) == "failed"


def test_missing_file_raises(tmp_path) -> None:
    pipe = _build_pipeline(tmp_path)
    with pytest.raises(FileNotFoundError):
        pipe.run(str(tmp_path / "nope.md"))


def test_settings_default_collection(tmp_path) -> None:
    settings = Settings(
        llm=LLMSettings(provider="dashscope", model="qwen-plus"),
        embedding=EmbeddingSettings(provider="dashscope", model="text-embedding-v3"),
        vector_store=VectorStoreSettings(
            provider="chroma", collection="cfg_col", persist_dir=str(tmp_path / "chroma")
        ),
    )
    pipe = IngestionPipeline(
        settings=settings,
        loaders=[FakeLoader()],
        embedding=FakeEmbedding(),
        vector_store=FakeVectorStore(),
        integrity=SQLiteIntegrityChecker(db_path=tmp_path / "db" / "ingestion.db"),
        image_storage=ImageStorage(
            images_dir=tmp_path / "images", db_path=tmp_path / "db" / "img.db"
        ),
        bm25_indexer=BM25Indexer(index_path=tmp_path / "bm25" / "index.pkl"),
        trace_writer=lambda _trace: None,
    )
    src = _write(tmp_path / "doc.md", "content for collection test")
    result = pipe.run(src)
    assert result.collection == "cfg_col"


def test_empty_document_zero_chunks(tmp_path) -> None:
    pipe = _build_pipeline(tmp_path, loader=FakeLoader(text="   \n\n  "))
    src = _write(tmp_path / "doc.md", "whatever")
    result = pipe.run(src)

    assert result.error is None
    assert result.skipped is False
    assert result.total_chunks == 0
    assert result.stages["upsert"]["vector_store_count"] == 0


def test_images_saved_when_present(tmp_path) -> None:
    img = tmp_path / "img" / "pic.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake image bytes")
    images = [ImageRef(id="img1", path=str(img), page=1)]

    pipe = _build_pipeline(tmp_path, loader=FakeLoader(text="content with image", images=images))
    src = _write(tmp_path / "doc.md", "content")
    result = pipe.run(src)

    assert result.total_images == 1
    assert pipe._image_storage.exists("img1")


def test_pipeline_persists_complete_ingestion_trace(tmp_path) -> None:
    """F4: Pipeline 创建一份 trace，贯穿五个摄取阶段并交给 F2 写入。"""
    persisted = []
    store = FakeVectorStore()
    pipe = _build_pipeline(tmp_path, vector_store=store, trace_writer=persisted.append)
    src = _write(tmp_path / "doc.md", "trace the ingestion pipeline")

    result = pipe.run(src, collection="test")

    assert result.error is None
    assert len(persisted) == 1
    trace = persisted[0]
    assert trace["trace_type"] == "ingestion"
    assert trace["finished_at"] is not None
    assert [stage["name"] for stage in trace["stages"]] == [
        "load", "split", "transform", "embed", "upsert",
    ]
    assert all(stage["elapsed_ms"] >= 0 for stage in trace["stages"])
    assert all(stage["method"] for stage in trace["stages"])
    assert pipe._embedding.traces[0].trace_id == trace["trace_id"]
    assert store.traces[0].trace_id == trace["trace_id"]
