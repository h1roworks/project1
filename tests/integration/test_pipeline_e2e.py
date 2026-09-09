"""C14: IngestionPipeline 端到端集成测试（真实存储，临时目录）。

用真实 ChromaStore / BM25Indexer（自动落盘）/ ImageStorage / SQLite 验证：
- 端到端编排：run() 后向量库可检索命中、BM25 索引建立并持久化
- BM25 索引按 collection 自动创建（``bm25_dir/{collection}.pkl``），不注入实例
- 幂等：同文件二次 run → skipped=True，向量不重复
- 清理沿用 test_vector_upserter_chroma.py 的 clear_system_cache + rmtree 模式
  （Windows 上 chroma 持有文件句柄）。
"""

import shutil

import pytest

from core.settings import VectorStoreSettings
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


class FakeEmbedding:
    """确定性向量：文本越长向量值越大，便于排序断言。"""

    def embed(self, texts, trace=None):
        return [[float(len(t)), 0.5, 1.0] for t in texts]


class FakeLoader(BaseLoader):
    loader_name = "fake"
    supported_extensions = (".md", ".txt")

    def load(self, path):
        from core.types import Document

        return Document(
            id="e2e_doc",
            text="RAG retrieval system with BM25 keyword matching\n\n"
            "Dense embedding captures semantic similarity for hybrid search.",
            metadata={"source_path": str(path), "doc_type": "md", "images": []},
        )


@pytest.fixture
def pipeline(tmp_path):
    store = ChromaStore(
        VectorStoreSettings(
            provider="chroma",
            collection="pipeline_e2e",
            persist_dir=str(tmp_path / "chroma"),
        )
    )
    pipe = IngestionPipeline(
        loaders=[FakeLoader()],
        embedding=FakeEmbedding(),
        vector_store=store,
        integrity=SQLiteIntegrityChecker(db_path=tmp_path / "db" / "ingestion.db"),
        image_storage=ImageStorage(
            images_dir=tmp_path / "images", db_path=tmp_path / "db" / "img.db"
        ),
        bm25_dir=tmp_path / "bm25",  # 不注入实例 → 测按 collection 自动创建路径
    )
    yield pipe
    try:
        store._client.clear_system_cache()
    except Exception:  # noqa: BLE001 - 清理失败不影响测试结果
        pass
    shutil.rmtree(tmp_path, ignore_errors=True)


def _write(path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_e2e_ingestion_roundtrip(pipeline, tmp_path) -> None:
    src = _write(tmp_path / "doc.md", "rag and bm25 content")
    result = pipeline.run(src, collection="hybrid")

    assert result.error is None
    assert result.skipped is False
    assert result.total_chunks > 0

    # 稠密路径：向量库可检索命中（FakeEmbedding 向量由文本长度决定）
    store = pipeline._vector_store
    matches = store.query([float(len("rag and bm25 content")), 0.5, 1.0], top_k=10)
    assert len(matches) == result.total_chunks
    assert all(m.id.startswith("e2e_doc") for m in matches)

    # 稀疏路径：BM25 索引自动创建并持久化到 bm25_dir/{collection}.pkl
    bm25_path = tmp_path / "bm25" / "hybrid.pkl"
    assert bm25_path.exists()
    loaded = BM25Indexer.load_from(bm25_path)
    assert loaded.total_docs == result.total_chunks
    assert loaded.postings.get("rag") is not None  # 关键词已入倒排索引


def test_e2e_rerun_skips_duplicate(pipeline, tmp_path) -> None:
    src = _write(tmp_path / "doc.md", "duplicate ingestion check")
    first = pipeline.run(src, collection="hybrid")
    assert first.skipped is False

    second = pipeline.run(src, collection="hybrid")
    assert second.skipped is True
    assert second.total_chunks == 0

    # 向量库未因重复摄取而膨胀（幂等：第二次直接跳过）
    store = pipeline._vector_store
    matches = store.query([0.0, 0.0, 0.0], top_k=50)
    assert len(matches) == first.total_chunks
