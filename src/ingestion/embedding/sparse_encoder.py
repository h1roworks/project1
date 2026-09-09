"""稀疏向量编码（C9：SparseEncoder）。

对 chunks 建立 BM25 所需的词项统计：分词 → 词频（term frequency），
输出每个 chunk 的关键词权重结构 ``{term: tf}``（term weights），
供 C11 BM25Indexer 消费以计算 IDF / 构建倒排索引。

对齐 DEV_SPEC 3.1.2 "双路编码（Dense + Sparse）"：
- Dense Embeddings 捕捉语义关联，Sparse Embeddings（BM25 关键词权重）捕捉精确匹配；
- 本模块只做"词项统计"，IDF 与倒排索引的构建落在 C11。

输出契约（C11 消费约定）：
- ``encode(chunks)`` 返回与输入等长、顺序对齐的 ``list[dict[str, int]]``；
- ``result[i]`` 为 ``chunks[i]`` 的 ``{term: 词频}``（int，兼容 ChunkRecord.sparse_vector 的 float）；
- 空文本 / 无词项文本返回空 dict ``{}``；
- 文档长度可由 ``sum(tf.values())`` 得出，无需单独输出。
"""

from __future__ import annotations

from typing import Any

from core.tokenizer import DEFAULT_STOPWORDS, tokenize_text


class SparseEncoder:
    """稀疏向量编码器：把文本转成 ``{term: tf}`` 关键词权重结构（纯本地计算）。"""

    name = "sparse_encoder"

    def __init__(self, stopwords: set[str] | None = None) -> None:
        """初始化。

        Args:
            stopwords: 需要过滤的停用词集合；None 使用内置英文停用词，
                传空集合 ``set()`` 则不过滤任何词。
        """
        self.stopwords = (
            set(stopwords) if stopwords is not None else set(DEFAULT_STOPWORDS)
        )

    def tokenize(self, text: str) -> list[str]:
        """把文本拆成词项序列（复用 Core 层共享分词，见 ``core.tokenizer``）。"""
        return tokenize_text(text, self.stopwords)

    def term_freqs(self, text: str) -> dict[str, int]:
        """统计单个文本的词频：``{term: 出现次数}``。空文本返回 ``{}``。"""
        freqs: dict[str, int] = {}
        for term in self.tokenize(text):
            freqs[term] = freqs.get(term, 0) + 1
        return freqs

    def encode(self, chunks: list[Any], trace: Any = None) -> list[dict[str, int]]:
        """对 chunks 批量编码，返回与输入等长、顺序对齐的 ``{term: tf}`` 结构。

        Args:
            chunks: 待编码的 Chunk 列表（读取 ``.text``）。
            trace: 预留的 TraceContext（阶段 F 落地；本地计算无需调用外部服务）。

        Returns:
            ``len(chunks)`` 条词频结构，``result[i]`` 对应 ``chunks[i]``。
        """
        return [self.term_freqs(c.text) for c in chunks]
