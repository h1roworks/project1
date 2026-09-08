"""B7.6: ChromaStore 集成测试——完整的 upsert→query roundtrip。

使用临时目录做真实持久化，验证返回结果的确定性与正确性；测试结束清理。
"""

import shutil

import pytest

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import VectorMatch, VectorRecord
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.fixture
def store(tmp_path):
    settings = VectorStoreSettings(
        provider="chroma",
        collection="test_col",
        persist_dir=str(tmp_path / "chroma"),
    )
    s = ChromaStore(settings)
    yield s
    # Windows 上 chroma 会持有文件句柄，先清空系统缓存再删除临时目录
    try:
        s._client.clear_system_cache()
    except Exception:  # noqa: BLE001 - 清理失败不影响测试结果
        pass
    shutil.rmtree(tmp_path, ignore_errors=True)


def make_record(i: int) -> VectorRecord:
    return VectorRecord(
        id=f"chunk-{i}",
        text=f"text-{i}",
        vector=[float(i), 0.0, 1.0],
        metadata={"doc_id": "doc-a", "page": i},
    )


def _upsert_five(store: ChromaStore) -> None:
    assert store.upsert([make_record(i) for i in range(5)]) == 5


# ---------- roundtrip ----------

def test_upsert_query_roundtrip(store: ChromaStore) -> None:
    _upsert_five(store)
    results = store.query([5.0, 5.0, 0.0], top_k=5)
    assert len(results) == 5
    # 与查询向量最相似的是 chunk-4
    assert results[0].id == "chunk-4"
    assert results[0].text == "text-4"
    assert results[0].metadata["doc_id"] == "doc-a"


def test_query_is_deterministic(store: ChromaStore) -> None:
    _upsert_five(store)
    r1 = [m.id for m in store.query([1.0, 1.0, 0.0], top_k=5)]
    r2 = [m.id for m in store.query([1.0, 1.0, 0.0], top_k=5)]
    assert r1 == r2


# ---------- top_k ----------

def test_query_respects_top_k(store: ChromaStore) -> None:
    _upsert_five(store)
    assert len(store.query([1.0, 1.0, 0.0], top_k=2)) == 2


# ---------- metadata filters ----------

def test_query_applies_metadata_filters(store: ChromaStore) -> None:
    _upsert_five(store)
    results = store.query([1.0, 1.0, 0.0], top_k=10, filters={"page": 1})
    assert [m.id for m in results] == ["chunk-1"]


# ---------- upsert 幂等 ----------

def test_upsert_is_idempotent(store: ChromaStore) -> None:
    _upsert_five(store)
    store.upsert([make_record(0)])  # 同 id 覆盖
    results = store.query([1.0, 1.0, 0.0], top_k=10)
    assert len(results) == 5


# ---------- 结果契约 ----------

def test_query_returns_vector_match_shape(store: ChromaStore) -> None:
    _upsert_five(store)
    for m in store.query([1.0, 1.0, 0.0], top_k=5):
        assert isinstance(m, VectorMatch)
        assert isinstance(m.id, str) and m.id
        assert isinstance(m.score, float)
        assert isinstance(m.text, str)
        assert isinstance(m.metadata, dict)


# ---------- 工厂路由 ----------

def test_factory_creates_chroma(tmp_path) -> None:
    settings = VectorStoreSettings(
        provider="chroma",
        collection="test_col",
        persist_dir=str(tmp_path / "chroma"),
    )
    s = VectorStoreFactory.create(settings)
    assert isinstance(s, ChromaStore)
    try:
        s._client.clear_system_cache()
    except Exception:  # noqa: BLE001
        pass
