"""D1: QueryProcessor 单元测试。

验证：
- 输出契约：返回 ProcessedQuery，字段完整、method="rule"
- 关键词提取：英文去停用词、中文单字、去重保序
- 与索引端一致：与 SparseEncoder 共用同一套分词规则
- 同义词扩展：扩展项并入稀疏词项（权重 0.8），原始关键词权重 1.0，
  稠密查询保持单次不受扩展影响
- Metadata 过滤解析：显式 filters 优先、内联 key:value 解析、未知键不误伤
- 边界行为：空/空白查询、纯停用词查询、纯过滤约束查询、URL 不误解析
"""

from core.query_engine.query_processor import QueryProcessor
from core.types import ProcessedQuery

# ---------- 输出契约 ----------


def test_process_returns_processed_query() -> None:
    result = QueryProcessor().process("What is RAG?")
    assert isinstance(result, ProcessedQuery)
    assert result.original_query == "What is RAG?"
    assert result.method == "rule"


def test_processed_query_field_stability() -> None:
    pq = QueryProcessor().process("RAG")
    assert set(pq.to_dict()) == {
        "original_query", "keywords", "sparse_terms", "dense_query", "filters", "method",
    }


def test_processed_query_roundtrip() -> None:
    pq = ProcessedQuery(
        original_query="What is RAG?",
        keywords=["rag"],
        sparse_terms={"rag": 1.0},
        dense_query="What is RAG?",
        filters={"collection": "docs"},
        method="rule",
    )
    assert ProcessedQuery.from_dict(pq.to_dict()) == pq


def test_processed_query_defaults() -> None:
    pq = ProcessedQuery(original_query="")
    assert pq.keywords == []
    assert pq.sparse_terms == {}
    assert pq.dense_query == ""
    assert pq.filters == {}
    assert pq.method == "rule"


def test_processed_query_json_serializable() -> None:
    import json

    pq = ProcessedQuery(original_query="RAG", keywords=["rag"], sparse_terms={"rag": 1.0})
    assert json.loads(json.dumps(pq.to_dict()))["keywords"] == ["rag"]


# ---------- 关键词提取 ----------


def test_keyword_extraction_english_removes_stopwords() -> None:
    result = QueryProcessor().process("What is the best RAG system?")
    assert result.keywords == ["best", "rag", "system"]
    assert result.dense_query == "What is the best RAG system?"


def test_keyword_extraction_chinese_single_chars() -> None:
    result = QueryProcessor().process("如何实现检索增强")
    assert result.keywords == ["如", "何", "实", "现", "检", "索", "增", "强"]


def test_keywords_deduplicated_preserving_order() -> None:
    result = QueryProcessor().process("RAG RAG is RAG")
    assert result.keywords == ["rag"]
    assert result.sparse_terms == {"rag": 1.0}


def test_keywords_share_tokenizer_with_sparse_encoder() -> None:
    from ingestion.embedding.sparse_encoder import SparseEncoder

    text = "BM25 sparse retrieval keywords"
    expected = list(dict.fromkeys(SparseEncoder().tokenize(text)))
    assert QueryProcessor().process(text).keywords == expected


# ---------- 同义词扩展 ----------


def test_synonym_expansion_merges_into_sparse_terms() -> None:
    proc = QueryProcessor(synonym_dict={"rag": ["retrieval", "augmented", "generation"]})
    result = proc.process("what is rag?")
    assert result.keywords == ["rag"]
    assert result.sparse_terms == {
        "rag": 1.0,
        "retrieval": 0.8,
        "augmented": 0.8,
        "generation": 0.8,
    }
    # 稠密检索保持单次：扩展不改变 dense_query
    assert result.dense_query == "what is rag?"


def test_synonym_colliding_with_keyword_keeps_higher_weight() -> None:
    proc = QueryProcessor(synonym_dict={"rag": ["rag", "retrieval"]})
    result = proc.process("rag system")
    assert result.sparse_terms == {"rag": 1.0, "system": 1.0, "retrieval": 0.8}


def test_no_synonym_expansion_by_default() -> None:
    assert QueryProcessor().process("rag").sparse_terms == {"rag": 1.0}


# ---------- Metadata 过滤解析 ----------


def test_explicit_filters_preserved() -> None:
    result = QueryProcessor().process(
        "retrieval augmented generation",
        filters={"collection": "docs", "doc_type": "pdf"},
    )
    assert result.filters == {"collection": "docs", "doc_type": "pdf"}


def test_inline_filter_parsed_and_stripped() -> None:
    result = QueryProcessor().process("什么是 RAG doc_type:pdf")
    assert result.filters == {"doc_type": "pdf"}
    assert result.dense_query == "什么是 RAG"
    assert "doc_type" not in result.dense_query


def test_inline_filter_fullwidth_colon() -> None:
    result = QueryProcessor().process("RAG doc_type：pdf")
    assert result.filters == {"doc_type": "pdf"}


def test_inline_filter_or_list_value() -> None:
    result = QueryProcessor().process("RAG language:en|zh")
    assert result.filters == {"language": ["en", "zh"]}


def test_inline_filter_strips_trailing_punctuation() -> None:
    result = QueryProcessor().process("show RAG doc_type:pdf.")
    assert result.filters == {"doc_type": "pdf"}


def test_explicit_filters_win_over_inline() -> None:
    result = QueryProcessor().process("RAG collection:books", filters={"collection": "docs"})
    assert result.filters == {"collection": "docs"}


def test_unknown_keyword_not_treated_as_filter() -> None:
    result = QueryProcessor().process("see the notes")
    assert result.filters == {}
    assert result.dense_query == "see the notes"


def test_url_not_misparsed_as_filter() -> None:
    result = QueryProcessor().process("how to use https://example.com")
    assert result.filters == {}
    assert "https" in result.keywords
    assert "example.com" in result.dense_query


def test_custom_filter_keys() -> None:
    proc = QueryProcessor(filter_keys={"author"})
    assert proc.process("RAG author:alice").filters == {"author": "alice"}
    # 自定义键集合替换内置集合，默认键不再生效
    assert proc.process("RAG doc_type:pdf").filters == {}


def test_filter_only_query() -> None:
    result = QueryProcessor().process("doc_type:pdf")
    assert result.filters == {"doc_type": "pdf"}
    assert result.keywords == []
    assert result.dense_query == ""


# ---------- 边界行为 ----------


def test_empty_query() -> None:
    result = QueryProcessor().process("")
    assert result.original_query == ""
    assert result.keywords == []
    assert result.sparse_terms == {}
    assert result.filters == {}


def test_whitespace_query() -> None:
    result = QueryProcessor().process("   ")
    assert result.keywords == []
    assert result.dense_query == ""


def test_stopwords_only_query() -> None:
    result = QueryProcessor().process("the and of")
    assert result.keywords == []
    assert result.sparse_terms == {}
    assert result.dense_query == "the and of"


def test_custom_stopwords() -> None:
    proc = QueryProcessor(stopwords={"rag"})
    assert proc.process("rag system").keywords == ["system"]


def test_input_not_mutated() -> None:
    query = "RAG doc_type:pdf"
    QueryProcessor().process(query)
    assert query == "RAG doc_type:pdf"
