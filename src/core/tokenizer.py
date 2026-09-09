"""共享文本分词工具（D1：查询端与索引端复用同一套分词规则）。

索引端（C9 SparseEncoder / C11 BM25Indexer）与查询端（D1 QueryProcessor
关键词提取）必须使用**同一套**分词规则，否则稀疏检索时查询词项无法命中
索引词项。故把分词正则与默认停用词下沉为 Core 层共享工具。

分词规则（与 C9 SparseEncoder 原行为保持一致）：
- 统一小写化；
- 英文单词/数字按 ``[a-z0-9]+`` 切分，连续汉字按单字切分（``[一-鿿]``）；
- 标点、空白、连字符等均作为分隔符；
- 输出时过滤停用词（默认英文虚词集合）。
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")

DEFAULT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "could", "did", "do", "does", "for", "from", "had",
        "has", "have", "he", "her", "his", "how", "i", "if", "in", "into",
        "is", "it", "its", "may", "might", "of", "on", "or", "our", "shall",
        "she", "should", "so", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "to", "was", "we", "were", "what",
        "when", "where", "which", "who", "will", "with", "would", "you",
    }
)


def tokenize_text(text: str, stopwords: set[str] | None = None) -> list[str]:
    """把文本拆成词项序列：小写 → 提取英文词/数字 + 单个汉字 → 去停用词。

    Args:
        text: 待分词文本。
        stopwords: 需要过滤的停用词集合；None 使用内置英文停用词，
            传空集合 ``set()`` 则不过滤任何词。

    Returns:
        按出现顺序的词项列表（小写，已过滤停用词）。
    """
    sw = set(stopwords) if stopwords is not None else set(DEFAULT_STOPWORDS)
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in sw]
