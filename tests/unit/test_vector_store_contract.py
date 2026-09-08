"""B4: VectorStore 契约测试。

用一个内存实现验证 upsert/query 契约的输入输出 shape，
以及 VectorStoreFactory 的路由逻辑。
"""

import math

import pytest

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


class InMemoryStore(BaseVectorStore):
    """最小内存实现：存储记录 + 余弦相似度检索。"""

    provider = "memory"

    def __init__(self, settings: VectorStoreSettings | None = None) -> None:
        self.settings = settings
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records, trace=None) -> int:
        for r in records:
            self._records[r.id] = r
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None) -> list[VectorMatch]:
        scored = []
        for rec in self._records.values():
            if filters and not all(rec.metadata.get(k) == v for k, v in filters.items()):
                continue
            denom = _norm(vector) * _norm(rec.vector) or 1.0
            score = _dot(vector, rec.vector) / denom
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            VectorMatch(id=rec.id, score=s, text=rec.text, metadata=rec.metadata)
            for s, rec in scored[:top_k]
        ]


def make_record(i: int) -> VectorRecord:
    return VectorRecord(
        id=f"chunk-{i}",
        text=f"text-{i}",
        vector=[float(i), 0.0, 1.0],
        metadata={"doc_id": "doc-a", "page": i},
    )


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore()
    s.upsert([make_record(i) for i in range(5)])
    return s


# ---------- 契约：upsert ----------

def test_upsert_returns_count(store: InMemoryStore) -> None:
    assert store.upsert([make_record(99)]) == 1


def test_upsert_is_idempotent(store: InMemoryStore) -> None:
    before = len(store.query([1.0, 1.0, 0.0], top_k=10))
    store.upsert([make_record(0)])  # 同 id 覆盖
    after = len(store.query([1.0, 1.0, 0.0], top_k=10))
    assert before == after


# ---------- 契约：query ----------

def test_query_returns_vector_match_shape(store: InMemoryStore) -> None:
    results = store.query([1.0, 1.0, 0.0], top_k=3)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, VectorMatch)
        assert isinstance(r.id, str) and r.id
        assert isinstance(r.score, float)
        assert isinstance(r.text, str)
        assert isinstance(r.metadata, dict)


def test_query_respects_top_k(store: InMemoryStore) -> None:
    assert len(store.query([1.0, 1.0, 0.0], top_k=2)) == 2
    assert len(store.query([1.0, 1.0, 0.0], top_k=100)) == 5


def test_query_ranks_by_similarity(store: InMemoryStore) -> None:
    results = store.query([5.0, 5.0, 0.0], top_k=5)
    assert results[0].id == "chunk-4"  # 与查询向量最相似


def test_query_applies_filters(store: InMemoryStore) -> None:
    results = store.query([1.0, 1.0, 0.0], top_k=10, filters={"page": 1})
    assert [r.id for r in results] == ["chunk-1"]


# ---------- 契约：工厂路由 ----------

def test_factory_returns_registered_impl() -> None:
    VectorStoreFactory.register("memory", InMemoryStore)
    try:
        store = VectorStoreFactory.create(VectorStoreSettings(provider="memory"))
        assert isinstance(store, BaseVectorStore)
        assert isinstance(store, InMemoryStore)
    finally:
        VectorStoreFactory._registry.pop("memory", None)


def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="未知的 VectorStore provider"):
        VectorStoreFactory.create(VectorStoreSettings(provider="nonexistent"))


def test_factory_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="provider 未配置"):
        VectorStoreFactory.create(VectorStoreSettings(provider=""))
