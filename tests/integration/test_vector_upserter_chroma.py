"""C12: VectorUpserter × ChromaStore 集成测试。

用真实 chromadb（临时持久化目录）验证：
- 富 Metadata（含 dict/嵌套列表）经规约后可通过 chroma 的 metadata 类型校验
- Dense + Sparse 双路写入在真实后端生效：向量可检索、BM25 可更新
- 幂等：同 id 重复 upsert 不产生重复向量
- source_path 作为 metadata 过滤字段可被 chroma where 查询命中
测试结束清理临时目录（Windows 上 chroma 持有文件句柄）。
"""

import json
import shutil

import pytest

from core.settings import VectorStoreSettings
from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.fixture
def chroma_store(tmp_path):
    settings = VectorStoreSettings(
        provider="chroma",
        collection="upsert_col",
        persist_dir=str(tmp_path / "chroma"),
    )
    s = ChromaStore(settings)
    yield s
    try:
        s._client.clear_system_cache()
    except Exception:  # noqa: BLE001 - 清理失败不影响测试结果
        pass
    shutil.rmtree(tmp_path, ignore_errors=True)


def make_record(
    chunk_id: str,
    source: str = "/data/sample.pdf",
    text: str = "RAG retrieval system with BM25 keyword matching",
    dense: list[float] | None = None,
    sparse: dict[str, int] | None = None,
    extra_meta: dict | None = None,
) -> ChunkRecord:
    meta = {"source_path": source, "chunk_index": 0, "image_refs": []}
    if extra_meta:
        meta.update(extra_meta)
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata=meta,
        dense_vector=dense,
        sparse_vector=sparse,
    )


def _make_vector(i: int) -> list[float]:
    """确定性向量：文本越长向量值越大，便于断言排序。"""
    return [float(i + 1), 0.0, 1.0]


def _build_upserter(chroma_store, tmp_path) -> tuple[VectorUpserter, BM25Indexer]:
    indexer = BM25Indexer(index_path=tmp_path / "bm25" / "index.pkl")
    return VectorUpserter(chroma_store, indexer), indexer


def test_rich_metadata_accepted_by_chroma(chroma_store, tmp_path) -> None:
    up, indexer = _build_upserter(chroma_store, tmp_path)
    records = [
        make_record(
            "c1",
            text="RAG overview",
            dense=_make_vector(1),
            sparse={"rag": 1, "overview": 1},
            extra_meta={
                "images": [{"id": "img1", "path": "/x.png"}],  # dict 列表
                "position": {"x": 10, "y": 20},  # dict
                "tags": ["rag", "bm25"],  # 标量列表
            },
        )
    ]
    result = up.upsert(records)
    assert result.vector_store_count == 1

    # 向量可检索，metadata 中 source_path 可被 where 过滤命中
    matches = chroma_store.query(_make_vector(1), top_k=5, filters={"source_path": "/data/sample.pdf"})
    assert matches[0].id == "c1"
    meta = matches[0].metadata
    # 复杂结构以 JSON 字符串回存，可还原
    assert json.loads(meta["images"]) == [{"id": "img1", "path": "/x.png"}]
    assert json.loads(meta["position"]) == {"x": 10, "y": 20}
    assert meta["tags"] == ["rag", "bm25"]


def test_dual_path_write_roundtrip(chroma_store, tmp_path) -> None:
    up, indexer = _build_upserter(chroma_store, tmp_path)
    records = [
        make_record(
            f"c{i}",
            source="/data/multi.pdf",
            text=f"document chunk number {i} about retrieval",
            dense=_make_vector(i),
            sparse={"chunk": 1, "retrieval": 1, f"term{i}": 1},
        )
        for i in range(3)
    ]
    up.upsert(records)

    # 稠密路径：向量库可检索到全部 3 条
    assert len(chroma_store.query(_make_vector(2), top_k=10)) == 3
    # 稀疏路径：BM25 索引建立并可持久化重载
    assert indexer.total_docs == 3
    assert indexer.postings["retrieval"] == {"c0": 1, "c1": 1, "c2": 1}
    loaded = BM25Indexer.load_from(tmp_path / "bm25" / "index.pkl")
    assert loaded.postings == indexer.postings


def test_upsert_idempotent_on_chroma(chroma_store, tmp_path) -> None:
    up, _ = _build_upserter(chroma_store, tmp_path)
    records = [
        make_record("c1", text="RAG overview", dense=_make_vector(1), sparse={"rag": 1}),
        make_record("c2", text="BM25 details", dense=_make_vector(2), sparse={"bm25": 1}),
    ]
    up.upsert(records)
    up.upsert(records)  # 重复摄取同一批

    # 向量库仍只有 2 条（同 id 覆盖），检索结果数不变
    assert len(chroma_store.query(_make_vector(2), top_k=10)) == 2


def test_chroma_factory_wires_upserter(tmp_path) -> None:
    """经 VectorStoreFactory 创建的 ChromaStore 可直接用于 VectorUpserter。"""
    settings = VectorStoreSettings(
        provider="chroma",
        collection="factory_col",
        persist_dir=str(tmp_path / "chroma"),
    )
    store = VectorStoreFactory.create(settings)
    assert isinstance(store, ChromaStore)
    up = VectorUpserter(store)
    result = up.upsert(
        [make_record("c1", text="hello world", dense=_make_vector(1), sparse={"hello": 1})]
    )
    assert result.vector_store_count == 1
    assert result.bm25_count == 0  # 未注入 BM25Indexer → dense-only
    try:
        store._client.clear_system_cache()
    except Exception:  # noqa: BLE001
        pass
