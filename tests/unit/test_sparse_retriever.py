"""D3：SparseRetriever 的单元测试，不依赖真实 ChromaDB 或网络服务。"""

from types import SimpleNamespace

import pytest

from core.query_engine.sparse_retriever import SparseRetriever
from core.types import ChunkRecord, RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import VectorMatch


class RecordingBM25:
    def __init__(self, candidates=None) -> None:
        self.candidates = candidates if candidates is not None else [
            {"chunk_id": "chunk-1", "score": 2.5}
        ]
        self.calls: list[tuple[object, int, object]] = []

    def query(self, keywords, top_k=10, trace=None):
        self.calls.append((keywords, top_k, trace))
        return self.candidates


class RecordingStore:
    def __init__(self, matches=None) -> None:
        self.matches = matches if matches is not None else [
            VectorMatch(
                id="chunk-1",
                score=0.0,
                text="BM25 finds exact keyword matches.",
                metadata={"source_path": "bm25.md"},
            )
        ]
        self.calls: list[tuple[list[str], object]] = []

    def get_by_ids(self, ids, trace=None):
        self.calls.append((ids, trace))
        return self.matches


def make_retriever(bm25=None, store=None) -> SparseRetriever:
    return SparseRetriever(
        SimpleNamespace(),
        bm25_indexer=bm25 or RecordingBM25(),
        vector_store=store or RecordingStore(),
    )


def test_retrieve_queries_bm25_then_returns_complete_result() -> None:
    bm25 = RecordingBM25()
    store = RecordingStore()

    results = make_retriever(bm25, store).retrieve(["bm25", "keyword"], top_k=3)

    assert results == [
        RetrievalResult(
            chunk_id="chunk-1",
            score=2.5,
            text="BM25 finds exact keyword matches.",
            metadata={"source_path": "bm25.md"},
        )
    ]
    assert bm25.calls == [(["bm25", "keyword"], 3, None)]
    assert store.calls == [(["chunk-1"], None)]


def test_retrieve_forwards_weighted_terms_and_trace() -> None:
    bm25 = RecordingBM25()
    store = RecordingStore()
    terms = {"rag": 1.0, "retrieval": 0.8}
    trace = object()

    make_retriever(bm25, store).retrieve(terms, top_k=2, trace=trace)

    assert bm25.calls == [(terms, 2, trace)]
    assert store.calls == [(["chunk-1"], trace)]


def test_retrieve_preserves_bm25_ranking_not_store_order() -> None:
    bm25 = RecordingBM25([
        {"chunk_id": "first", "score": 3.0},
        {"chunk_id": "second", "score": 2.0},
    ])
    store = RecordingStore([
        VectorMatch(id="second", score=0.0, text="second text", metadata={}),
        VectorMatch(id="first", score=0.0, text="first text", metadata={}),
    ])

    results = make_retriever(bm25, store).retrieve(["keyword"])

    assert [result.chunk_id for result in results] == ["first", "second"]
    assert [result.score for result in results] == [3.0, 2.0]


def test_retrieve_skips_bm25_id_missing_from_vector_store() -> None:
    bm25 = RecordingBM25([
        {"chunk_id": "kept", "score": 2.0},
        {"chunk_id": "removed", "score": 1.0},
    ])
    store = RecordingStore([
        VectorMatch(id="kept", score=0.0, text="still stored", metadata={})
    ])

    results = make_retriever(bm25, store).retrieve(["keyword"])

    assert [result.chunk_id for result in results] == ["kept"]


def test_retrieve_accepts_mapping_style_vector_store_result() -> None:
    store = RecordingStore([
        {"id": "chunk-1", "text": "mapping text", "metadata": {"page": 1}}
    ])

    result = make_retriever(store=store).retrieve(["keyword"])[0]

    assert result.text == "mapping text"
    assert result.metadata == {"page": 1}


@pytest.mark.parametrize("keywords", [[], {}])
def test_empty_keywords_skip_all_external_calls(keywords) -> None:
    bm25 = RecordingBM25()
    store = RecordingStore()

    assert make_retriever(bm25, store).retrieve(keywords) == []
    assert bm25.calls == []
    assert store.calls == []


def test_invalid_keyword_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="keywords"):
        make_retriever().retrieve("bm25")  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        make_retriever().retrieve(["bm25"], top_k=top_k)


@pytest.mark.parametrize(
    ("match", "message"),
    [
        ({"text": "text", "metadata": {}}, "chunk_id"),
        ({"id": "chunk", "metadata": {}}, "text"),
        ({"id": "chunk", "text": "text", "metadata": []}, "metadata"),
    ],
)
def test_invalid_vector_store_result_has_clear_error(match, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_retriever(store=RecordingStore([match])).retrieve(["bm25"])


def test_retrieve_with_real_bm25_indexer_honors_keyword_weights() -> None:
    indexer = BM25Indexer()
    indexer.add([
        ChunkRecord(id="rag", text="RAG text", sparse_vector={"rag": 1}),
        ChunkRecord(id="alias", text="alias text", sparse_vector={"retrieval": 1}),
    ])
    store = RecordingStore([
        VectorMatch(id="rag", score=0.0, text="RAG text", metadata={}),
        VectorMatch(id="alias", score=0.0, text="alias text", metadata={}),
    ])

    results = make_retriever(indexer, store).retrieve({"rag": 1.0, "retrieval": 0.2})

    assert [result.chunk_id for result in results] == ["rag", "alias"]


def test_constructor_creates_vector_store_when_not_injected(monkeypatch) -> None:
    store = RecordingStore()
    settings = SimpleNamespace(vector_store="store-settings")
    bm25 = RecordingBM25()
    monkeypatch.setattr(
        "core.query_engine.sparse_retriever.VectorStoreFactory.create",
        lambda received: store if received == "store-settings" else None,
    )

    retriever = SparseRetriever(settings, bm25_indexer=bm25)

    assert retriever.vector_store is store
    assert retriever.bm25_indexer is bm25
