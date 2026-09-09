"""C11: BM25Indexer 单元测试。

验证：
- 输出契约：add(records) 建立倒排索引（term → {chunk_id: tf}）与文档表
- IDF 计算：词项文档频率越高 IDF 越低；未出现词项 IDF 为 0；平滑公式恒非负
- Upsert 幂等：同一 chunk_id 重复加入替换旧条目，不产生重复索引
- 文档长度：doc_len = sum(tf)，供 BM25 长度归一化；avgdl 正确
- remove_document：按 source 移除该文档全部条目，返回移除条数并更新 df
- 持久化：save → 新建 indexer 自动加载，postings/df 一致
- 边界行为：空 records、sparse_vector 为 None / 空、未知 source、无路径保存报错
- 与 SparseEncoder / BatchProcessor 的链路（C9 → C10 → C11）
"""

import math

import pytest

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer

# ---------- fixtures / helpers ----------


def make_record(
    chunk_id: str,
    source: str = "/data/sample.pdf",
    sparse: dict[str, int] | None = None,
    index: int = 0,
) -> ChunkRecord:
    """构造携带稀疏向量的 ChunkRecord，供索引消费。"""
    meta = {"source_path": source, "chunk_index": index, "image_refs": []}
    return ChunkRecord(
        id=chunk_id,
        text=f"content {index}",
        metadata=meta,
        sparse_vector=sparse,
    )


# ---------- 输出契约：倒排索引构建 ----------


def test_add_builds_inverted_index() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("d0_0001_aaaa", sparse={"rag": 1, "retrieval": 1}),
            make_record("d0_0002_bbbb", sparse={"rag": 1, "bm25": 2}),
        ]
    )
    assert indexer.postings["rag"] == {"d0_0001_aaaa": 1, "d0_0002_bbbb": 1}
    assert indexer.postings["retrieval"] == {"d0_0001_aaaa": 1}
    assert indexer.postings["bm25"] == {"d0_0002_bbbb": 2}


def test_add_tracks_document_source() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [make_record("c1", source="/a.pdf", sparse={"x": 1}, index=1)]
    )
    assert indexer.documents["c1"]["source"] == "/a.pdf"


def test_add_returns_processed_count() -> None:
    indexer = BM25Indexer()
    assert indexer.add([make_record("c1", sparse={"x": 1})]) == 1
    assert indexer.add([]) == 0


# ---------- 文档长度 / avgdl ----------


def test_add_counts_document_length() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"rag": 1, "bm25": 2, "sparse": 1})])
    assert indexer.documents["c1"]["doc_len"] == 4  # 词频之和


def test_avgdl_averages_document_lengths() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("c1", sparse={"a": 1, "b": 1}),  # 长度 2
            make_record("c2", sparse={"a": 1, "b": 1, "c": 1, "d": 1}),  # 长度 4
        ]
    )
    assert indexer.avgdl == 3.0
    assert BM25Indexer().avgdl == 0.0  # 无文档时返回 0


# ---------- IDF 计算 ----------


def test_idf_absent_term_is_zero() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"rag": 1})])
    assert indexer.idf("不存在词") == 0.0


def test_idf_rare_term_weights_higher() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("c1", sparse={"rare": 1, "common": 1}),
            make_record("c2", sparse={"common": 1}),
            make_record("c3", sparse={"common": 1}),
            make_record("c4", sparse={"common": 1}),
        ]
    )
    # N=4：rare 出现在 1 个文档，common 出现在 4 个文档
    assert indexer.idf("rare") > indexer.idf("common")


def test_idf_smoothed_formula_non_negative() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("c1", sparse={"term": 1}),
            make_record("c2", sparse={"term": 1}),
        ]
    )
    # df == N 时经典公式可能取负；平滑版恒为正
    assert indexer.idf("term") > 0


def test_idf_formula_exact_value() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("c1", sparse={"x": 1, "y": 1}),
            make_record("c2", sparse={"y": 1}),
        ]
    )
    # N=2, df(x)=1 → ln(1 + (2-1+0.5)/(1+0.5)) = ln(1 + 1.5/1.5) = ln(2)
    assert math.isclose(indexer.idf("x"), math.log(2), rel_tol=1e-9)


def test_df_counts_documents_per_term() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("c1", sparse={"a": 1, "b": 1}),
            make_record("c2", sparse={"a": 1}),
        ]
    )
    assert indexer.df("a") == 2
    assert indexer.df("b") == 1
    assert indexer.df("c") == 0


# ---------- Upsert 幂等 ----------


def test_add_same_chunk_id_replaces_entry() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"rag": 1, "old": 1})])
    indexer.add([make_record("c1", sparse={"rag": 1, "new": 1})])
    # 旧词项 old 被移除，倒排索引不残留
    assert indexer.postings["rag"] == {"c1": 1}
    assert "old" not in indexer.postings
    assert indexer.postings["new"] == {"c1": 1}
    assert indexer.total_docs == 1  # 同一 chunk 只计数一次


def test_add_same_chunk_twice_keeps_doc_len_updated() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"a": 1, "b": 1})])
    indexer.add([make_record("c1", sparse={"a": 5})])
    assert indexer.documents["c1"]["doc_len"] == 5  # 以最后一次为准


# ---------- remove_document ----------


def test_remove_document_removes_all_entries() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("a1", source="/a.pdf", sparse={"rag": 1}),
            make_record("a2", source="/a.pdf", sparse={"bm25": 1}),
            make_record("b1", source="/b.pdf", sparse={"rag": 1}),
        ]
    )
    removed = indexer.remove_document("/a.pdf")
    assert removed == 2
    assert "a1" not in indexer.documents
    assert "a2" not in indexer.documents
    assert indexer.documents["b1"]["source"] == "/b.pdf"  # 其他文档不受影响
    assert indexer.postings["rag"] == {"b1": 1}  # a1 的词项已从倒排索引移除
    assert "bm25" not in indexer.postings  # 空词项被清理


def test_remove_document_unknown_source_returns_zero() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"x": 1})])
    assert indexer.remove_document("/不存在的.pdf") == 0
    assert indexer.total_docs == 1


def test_remove_document_updates_df() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("a1", source="/a.pdf", sparse={"rag": 1}),
            make_record("b1", source="/b.pdf", sparse={"rag": 1}),
        ]
    )
    assert indexer.df("rag") == 2
    indexer.remove_document("/a.pdf")
    assert indexer.df("rag") == 1


# ---------- 边界行为 ----------


def test_empty_sparse_vector_indexed_without_postings() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={})])
    assert indexer.total_docs == 1  # 仍计入文档计数
    assert indexer.documents["c1"]["doc_len"] == 0
    assert indexer.postings == {}  # 不产生任何词项


def test_none_sparse_vector_treated_as_empty() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse=None)])
    assert indexer.total_docs == 1
    assert indexer.documents["c1"]["doc_len"] == 0
    assert indexer.postings == {}


def test_zero_or_negative_tf_ignored() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"zero": 0, "neg": -1, "ok": 2})])
    assert indexer.postings == {"ok": {"c1": 2}}
    assert indexer.documents["c1"]["doc_len"] == 2


# ---------- 持久化 ----------


def test_save_then_new_indexer_autoloads(tmp_path) -> None:
    path = tmp_path / "bm25" / "index.pkl"
    indexer = BM25Indexer(index_path=path)
    indexer.add(
        [
            make_record("c1", source="/a.pdf", sparse={"rag": 1, "bm25": 2}),
            make_record("c2", source="/a.pdf", sparse={"rag": 1}),
        ]
    )
    indexer.save()

    loaded = BM25Indexer(index_path=path)  # 构造时自动加载
    assert loaded.postings == indexer.postings
    assert loaded.documents == indexer.documents
    assert loaded.df("rag") == 2
    assert loaded.idf("bm25") == indexer.idf("bm25")


def test_load_from_classmethod(tmp_path) -> None:
    path = tmp_path / "index.pkl"
    indexer = BM25Indexer(index_path=path)
    indexer.add([make_record("c1", sparse={"rag": 1})])
    indexer.save()

    loaded = BM25Indexer.load_from(path)
    assert loaded.postings["rag"] == {"c1": 1}
    assert loaded.total_docs == 1


def test_save_without_path_raises() -> None:
    indexer = BM25Indexer()
    indexer.add([make_record("c1", sparse={"x": 1})])
    with pytest.raises(ValueError, match="index_path"):
        indexer.save()


def test_stats() -> None:
    indexer = BM25Indexer()
    indexer.add(
        [
            make_record("c1", sparse={"a": 1, "b": 1}),
            make_record("c2", sparse={"b": 1, "c": 1}),
        ]
    )
    assert indexer.stats() == {
        "total_docs": 2,
        "total_terms": 3,
        "total_postings": 4,
    }


# ---------- 与 SparseEncoder / BatchProcessor 的链路（C9 → C10 → C11） ----------


def test_chains_with_sparse_encoder(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → SparseEncoder → BM25Indexer。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from ingestion.chunking.document_chunker import DocumentChunker
    from ingestion.embedding.sparse_encoder import SparseEncoder
    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "bm25_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "RAG Overview")
    c.drawString(72, 770, "Inverted index maps terms to the chunks containing them.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)

    index_path = tmp_path / "bm25_index.pkl"
    indexer = BM25Indexer(index_path=index_path)
    records = [
        ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            sparse_vector=sparse,
        )
        for chunk, sparse in zip(chunks, SparseEncoder().encode(chunks))
    ]
    indexer.add(records)

    assert indexer.total_docs == len(chunks)
    # 原文实义词进入倒排索引
    assert "overview" in indexer.postings
    assert "inverted" in indexer.postings
    # 每个 chunk 都有可用的词项频率
    assert any(indexer.df(t) >= 1 for t in indexer.postings)
    # 索引可持久化后重新加载
    indexer.save()
    loaded = BM25Indexer.load_from(index_path)
    assert loaded.postings == indexer.postings


def test_chains_with_batch_processor() -> None:
    """BatchProcessor（C10）产出的 ChunkRecord 可直接喂给 BM25Indexer。"""
    from ingestion.embedding.batch_processor import BatchProcessor
    from ingestion.embedding.dense_encoder import DenseEncoder
    from ingestion.embedding.sparse_encoder import SparseEncoder
    from libs.embedding.base_embedding import BaseEmbedding

    class FakeEmbedding(BaseEmbedding):
        provider = "c11fake"

        def embed(self, texts, trace=None) -> list[list[float]]:
            return [[float(len(t))] * 2 for t in texts]

    from core.types import Chunk

    chunks = [
        Chunk(id=f"c{i}", text=text, metadata={"source_path": "/data/x.pdf"})
        for i, text in enumerate(["RAG retrieval system", "BM25 keyword matching"])
    ]
    proc = BatchProcessor(
        DenseEncoder(FakeEmbedding()), SparseEncoder(), batch_size=2
    )
    records = proc.process(chunks)

    indexer = BM25Indexer()
    indexer.add(records)
    assert indexer.total_docs == 2
    assert indexer.postings.get("retrieval")  # 第一段的实义词
    assert indexer.postings.get("matching")  # 第二段的实义词
    # 删除该文档后索引清空
    assert indexer.remove_document("/data/x.pdf") == 2
    assert indexer.total_docs == 0
    assert indexer.postings == {}
