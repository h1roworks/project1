"""C12: VectorUpserter 单元测试。

验证：
- 双路写入：有稠密向量的记录写入向量库（VectorRecord 形态），全部记录送入 BM25
- 顺序对齐 / 计数：upsert 返回 UpsertResult（total / vector_store_count / bm25_count）
- 幂等：同 id 重复 upsert 不产生重复索引（向量库覆盖、BM25 替换）
- metadata 规约：富 Metadata（dict/嵌套列表）转 JSON 字符串，source_path 原样保留
- All-in-One：sparse_vector 以 JSON 存入向量库记录 metadata，可解析还原
- 边界行为：空记录、全缺稠密向量（跳过向量库）、bm25_indexer=None（dense-only）
- BM25 持久化：配置 index_path 后 upsert 自动保存，可重新加载
- 链路（C9 → C10 → C12）：真实 PDF 经 Chunker → BatchProcessor → VectorUpserter
"""

import json

import pytest

from core.types import Chunk, ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import UpsertResult, VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch, VectorRecord

# ---------- fixtures / helpers ----------


class FakeVectorStore(BaseVectorStore):
    """最小内存向量库：记录写入历史，供断言 upsert 收到的内容。"""

    provider = "c12fake"

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.upsert_calls: list[list[VectorRecord]] = []

    def upsert(self, records, trace=None) -> int:
        self.upsert_calls.append(list(records))
        for r in records:
            self.records[r.id] = r
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None) -> list[VectorMatch]:
        return []

    def get_by_ids(self, ids, trace=None) -> list[VectorMatch]:
        return [
            VectorMatch(id=record.id, score=0.0, text=record.text, metadata=record.metadata)
            for chunk_id in ids
            if (record := self.records.get(chunk_id)) is not None
        ]


def make_record(
    chunk_id: str,
    source: str = "/data/sample.pdf",
    text: str = "content",
    dense: list[float] | None = [1.0, 2.0],
    sparse: dict[str, int] | None = None,
    index: int = 0,
    extra_meta: dict | None = None,
) -> ChunkRecord:
    """构造携带双路向量的 ChunkRecord，供 upserter 消费。"""
    meta = {"source_path": source, "chunk_index": index, "image_refs": []}
    if extra_meta:
        meta.update(extra_meta)
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata=meta,
        dense_vector=dense,
        sparse_vector=sparse,
    )


@pytest.fixture
def upserter() -> tuple[VectorUpserter, FakeVectorStore, BM25Indexer]:
    store = FakeVectorStore()
    indexer = BM25Indexer()
    return VectorUpserter(store, indexer), store, indexer


# ---------- 双路写入 ----------


def test_upsert_writes_both_paths(upserter) -> None:
    up, store, indexer = upserter
    result = up.upsert(
        [
            make_record("c1", sparse={"rag": 1, "retrieval": 1}),
            make_record("c2", sparse={"bm25": 2}),
        ]
    )
    assert isinstance(result, UpsertResult)
    assert result.total == 2
    assert result.vector_store_count == 2
    assert result.bm25_count == 2
    assert result.skipped_no_dense == 0

    # 稠密路径：向量库收到 VectorRecord，字段完整
    stored = store.records["c1"]
    assert isinstance(stored, VectorRecord)
    assert stored.id == "c1"
    assert stored.text == "content"
    assert stored.vector == [1.0, 2.0]
    assert stored.metadata["source_path"] == "/data/sample.pdf"

    # 稀疏路径：BM25 索引建立倒排
    assert indexer.postings["rag"] == {"c1": 1}
    assert indexer.postings["bm25"] == {"c2": 2}
    assert indexer.total_docs == 2


def test_upsert_returns_counts_on_odd_input(upserter) -> None:
    up, _, _ = upserter
    result = up.upsert([make_record("c1", dense=None, sparse={"x": 1})])
    assert result.total == 1
    assert result.vector_store_count == 0
    assert result.bm25_count == 1
    assert result.skipped_no_dense == 1


# ---------- 幂等 ----------


def test_upsert_is_idempotent(upserter) -> None:
    up, store, indexer = upserter
    records = [
        make_record("c1", sparse={"rag": 1, "old": 1}),
        make_record("c2", sparse={"rag": 1}),
    ]
    up.upsert(records)
    up.upsert([make_record("c1", sparse={"rag": 1, "new": 1})])  # 同 id 覆盖

    assert len(store.records) == 2  # 向量库同 id 覆盖，不新增
    assert indexer.total_docs == 2  # BM25 同一 chunk 只计数一次
    assert "old" not in indexer.postings  # 旧词项被替换
    assert indexer.postings["new"] == {"c1": 1}


# ---------- metadata 规约 ----------


def test_metadata_sanitized_for_vector_store(upserter) -> None:
    up, store, _ = upserter
    up.upsert(
        [
            make_record(
                "c1",
                extra_meta={
                    "images": [{"id": "img1", "path": "/x.png"}],  # dict 列表 → JSON
                    "position": {"x": 10},  # dict → JSON
                    "tags": ["rag", "bm25"],  # 标量列表 → 原样保留
                    "page": 3,  # 标量 → 原样保留
                },
            )
        ]
    )
    meta = store.records["c1"].metadata
    assert meta["source_path"] == "/data/sample.pdf"  # 可过滤字段保留
    assert meta["page"] == 3
    assert meta["tags"] == ["rag", "bm25"]
    # 复杂结构转为可解析的 JSON 字符串
    assert json.loads(meta["images"]) == [{"id": "img1", "path": "/x.png"}]
    assert json.loads(meta["position"]) == {"x": 10}


def test_empty_list_metadata_serialized_to_json(upserter) -> None:
    """chroma 拒绝空列表 metadata 值，空列表须转为 ``"[]"`` 字符串。"""
    up, store, _ = upserter
    up.upsert([make_record("c1", extra_meta={"image_refs": []})])
    assert store.records["c1"].metadata["image_refs"] == "[]"


def test_sparse_vector_preserved_in_metadata(upserter) -> None:
    up, store, _ = upserter
    up.upsert([make_record("c1", sparse={"rag": 1, "bm25": 2})])
    sparse = json.loads(store.records["c1"].metadata["sparse_vector"])
    assert sparse == {"rag": 1, "bm25": 2}


def test_input_records_not_mutated(upserter) -> None:
    up, _, _ = upserter
    record = make_record("c1", sparse={"rag": 1})
    before_meta = dict(record.metadata)
    up.upsert([record])
    assert record.metadata == before_meta  # 规约发生在副本上，原记录不受影响


# ---------- 边界行为 ----------


def test_empty_records_returns_empty_result(upserter) -> None:
    up, store, indexer = upserter
    result = up.upsert([])
    assert result.total == 0
    assert result.vector_store_count == 0
    assert result.bm25_count == 0
    assert store.upsert_calls == []  # 不触发任何写入
    assert indexer.total_docs == 0


def test_all_missing_dense_skips_vector_store(upserter) -> None:
    up, store, indexer = upserter
    result = up.upsert([make_record("c1", dense=None, sparse={"x": 1})])
    assert store.upsert_calls == []  # 全部缺稠密向量 → 不调用向量库
    assert result.skipped_no_dense == 1
    assert indexer.postings["x"] == {"c1": 1}  # 稀疏路径仍写入


def test_dense_only_mode_without_bm25(upserter) -> None:
    up, store, _ = upserter
    dense_only = VectorUpserter(store)  # bm25_indexer=None
    result = dense_only.upsert([make_record("c1", sparse={"rag": 1})])
    assert result.bm25_count == 0
    assert result.vector_store_count == 1
    assert store.records["c1"].id == "c1"


# ---------- BM25 持久化 ----------


def test_upsert_autosaves_bm25_index(tmp_path) -> None:
    store = FakeVectorStore()
    index_path = tmp_path / "bm25" / "index.pkl"
    indexer = BM25Indexer(index_path=index_path)
    up = VectorUpserter(store, indexer)
    up.upsert(
        [
            make_record("c1", sparse={"rag": 1}),
            make_record("c2", sparse={"bm25": 2}),
        ]
    )
    assert index_path.exists()  # 配置 index_path 后 upsert 自动落盘

    loaded = BM25Indexer.load_from(index_path)
    assert loaded.postings == indexer.postings
    assert loaded.total_docs == 2


def test_upsert_no_save_without_index_path(upserter) -> None:
    up, store, indexer = upserter  # indexer 无 index_path
    up.upsert([make_record("c1", sparse={"x": 1})])
    assert indexer.postings["x"] == {"c1": 1}  # 索引正常，但不要求落盘


# ---------- 链路（C9 → C10 → C12） ----------


def test_chains_with_batch_processor(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → BatchProcessor → VectorUpserter。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from ingestion.chunking.document_chunker import DocumentChunker
    from ingestion.embedding.batch_processor import BatchProcessor
    from ingestion.embedding.dense_encoder import DenseEncoder
    from ingestion.embedding.sparse_encoder import SparseEncoder
    from libs.embedding.base_embedding import BaseEmbedding
    from libs.loader.pdf_loader import PDFLoader

    class FakeEmbedding(BaseEmbedding):
        provider = "c12fake"

        def embed(self, texts, trace=None) -> list[list[float]]:
            return [[float(len(t)) % 100] * 4 for t in texts]

    pdf = tmp_path / "upsert_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "RAG Overview")
    c.drawString(72, 770, "Vector upserter writes dense and sparse indexes together.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks

    proc = BatchProcessor(
        DenseEncoder(FakeEmbedding()), SparseEncoder(), batch_size=2
    )
    records = proc.process(chunks)

    store = FakeVectorStore()
    indexer = BM25Indexer()
    up = VectorUpserter(store, indexer)
    result = up.upsert(records)

    assert result.total == len(chunks)
    assert result.vector_store_count == len(chunks)
    assert result.bm25_count == len(chunks)
    # 向量库与 BM25 索引内容一致、可检索
    assert len(store.records) == len(chunks)
    assert any("overview" in p for p in indexer.postings)
    # 重复摄取幂等：索引规模不变
    up.upsert(records)
    assert len(store.records) == len(chunks)
    assert indexer.total_docs == len(chunks)
