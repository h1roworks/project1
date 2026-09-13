"""D2：DenseRetriever 的单元测试。

所有外部依赖均使用简单替身对象，因此不会访问真实 Embedding API 或 ChromaDB。
"""

from types import SimpleNamespace

import pytest

from core.query_engine.dense_retriever import DenseRetriever
from core.types import RetrievalResult
from libs.vector_store.base_vector_store import VectorMatch


class RecordingEmbedding:
    """记录调用参数的假 Embedding 客户端。"""

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = [[0.1, 0.2, 0.3]] if vectors is None else vectors
        self.calls: list[tuple[list[str], object]] = []

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.calls.append((texts, trace))
        return self.vectors


class RecordingStore:
    """记录调用参数的假向量库。"""

    def __init__(self, matches=None) -> None:
        self.matches = matches if matches is not None else [
            VectorMatch(
                id="chunk-1",
                score=0.88,
                text="RAG combines retrieval and generation.",
                metadata={"source_path": "rag.md", "page": 1},
            )
        ]
        self.calls: list[tuple[list[float], int, object, object]] = []

    def query(self, vector, top_k=10, filters=None, trace=None):
        self.calls.append((vector, top_k, filters, trace))
        return self.matches


def make_retriever(embedding=None, store=None) -> DenseRetriever:
    # 注入两个替身后，settings 不会被读取；保留它使构造签名与生产代码一致。
    return DenseRetriever(
        SimpleNamespace(),
        embedding_client=embedding or RecordingEmbedding(),
        vector_store=store or RecordingStore(),
    )


def test_retrieve_embeds_query_and_returns_normalized_result() -> None:
    embedding = RecordingEmbedding()
    store = RecordingStore()
    retriever = make_retriever(embedding, store)

    results = retriever.retrieve("What is RAG?", top_k=3)

    assert results == [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.88,
            text="RAG combines retrieval and generation.",
            metadata={"source_path": "rag.md", "page": 1},
        )
    ]
    assert embedding.calls == [(["What is RAG?"], None)]
    assert store.calls == [([0.1, 0.2, 0.3], 3, None, None)]


def test_retrieve_forwards_filters_and_trace() -> None:
    embedding = RecordingEmbedding()
    store = RecordingStore()
    trace = object()

    make_retriever(embedding, store).retrieve(
        "RAG", top_k=2, filters={"collection": "docs"}, trace=trace
    )

    assert embedding.calls == [(["RAG"], trace)]
    assert store.calls == [([0.1, 0.2, 0.3], 2, {"collection": "docs"}, trace)]


def test_retrieve_normalizes_multiple_matches() -> None:
    store = RecordingStore([
        VectorMatch(id="chunk-1", score=0.9, text="one", metadata={}),
        VectorMatch(id="chunk-2", score=0.7, text="two", metadata={"page": 2}),
    ])

    results = make_retriever(store=store).retrieve("query")

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-2"]
    assert [result.score for result in results] == [0.9, 0.7]
    assert results[1].metadata == {"page": 2}


def test_retrieve_accepts_mapping_result_from_compatible_store() -> None:
    store = RecordingStore([
        {
            "chunk_id": "chunk-from-mapping",
            "score": "0.5",
            "text": "text from a compatible store",
            "metadata": {"doc_type": "md"},
        }
    ])

    result = make_retriever(store=store).retrieve("query")[0]

    assert result == RetrievalResult(
        chunk_id="chunk-from-mapping",
        score=0.5,
        text="text from a compatible store",
        metadata={"doc_type": "md"},
    )


def test_blank_query_returns_empty_without_external_calls() -> None:
    embedding = RecordingEmbedding()
    store = RecordingStore()

    results = make_retriever(embedding, store).retrieve("   ")

    assert results == []
    assert embedding.calls == []
    assert store.calls == []


def test_non_string_query_is_rejected() -> None:
    with pytest.raises(TypeError, match="query"):
        make_retriever().retrieve(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        make_retriever().retrieve("query", top_k=top_k)


@pytest.mark.parametrize("vectors", [[], [[]]])
def test_missing_query_vector_has_clear_error(vectors: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="查询向量"):
        make_retriever(embedding=RecordingEmbedding(vectors)).retrieve("query")


@pytest.mark.parametrize(
    ("match", "message"),
    [
        ({"score": 0.1, "text": "text", "metadata": {}}, "chunk_id"),
        ({"id": "chunk", "score": 0.1, "metadata": {}}, "text"),
        ({"id": "chunk", "score": 0.1, "text": "text", "metadata": []}, "metadata"),
    ],
)
def test_invalid_store_result_has_clear_error(match, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_retriever(store=RecordingStore([match])).retrieve("query")


def test_constructor_uses_factories_when_dependencies_are_not_injected(monkeypatch) -> None:
    embedding = RecordingEmbedding()
    store = RecordingStore()
    settings = SimpleNamespace(embedding="embedding-settings", vector_store="store-settings")

    monkeypatch.setattr(
        "core.query_engine.dense_retriever.EmbeddingFactory.create",
        lambda received: embedding if received == "embedding-settings" else None,
    )
    monkeypatch.setattr(
        "core.query_engine.dense_retriever.VectorStoreFactory.create",
        lambda received: store if received == "store-settings" else None,
    )

    retriever = DenseRetriever(settings)

    assert retriever.embedding_client is embedding
    assert retriever.vector_store is store
