"""查询预处理（D1：QueryProcessor）。

接收已由上游（Client/MCP Host）完成会话补全与指代消歧的独立查询
（Standalone Query），产出结构化的 ``ProcessedQuery``，供 D2~D7 检索链路消费：

- **关键词提取**：与索引端（C9 SparseEncoder）共用同一套分词规则
  （``core.tokenizer.tokenize_text``），去停用词、去重、保序；
- **查询扩展**：可配置同义词/别名映射，扩展项仅并入**稀疏检索**（加权 0.8），
  稠密检索保持单次（``dense_query`` 默认取剥离过滤约束后的查询文本）；
- **Metadata 过滤解析**：把结构化约束（``collection``/``doc_type``/``language``/
  ``access_level``/``time_range``）解析为通用 ``filters``。支持显式传入
  ``filters`` 与查询文本内联 ``key:value`` 两种来源，显式传入优先。

对齐 DEV_SPEC 3.1.2 "Query Processing (查询预处理)"。
"""

from __future__ import annotations

import re
from typing import Any

from core.tokenizer import tokenize_text
from core.types import ProcessedQuery

# 匹配查询文本中的 ``key:value`` 约束 token（value 以 | 分隔表示 OR 列表）。
_FILTER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])(?P<key>[a-zA-Z_]+)\s*[:：]\s*(?P<value>[^\s,，;；]+)"
)

# 词项权重：原始关键词更高权重以抑制同义词带来的语义漂移（DEV_SPEC 3.1.2）。
_ORIGINAL_TERM_WEIGHT = 1.0
_SYNONYM_TERM_WEIGHT = 0.8


class QueryProcessor:
    """查询预处理：关键词提取 + 同义词扩展 + Metadata 过滤解析（纯本地规则）。

    Args:
        synonym_dict: 同义词/别名映射 ``{关键词: [同义词, ...]}``；扩展项并入
            稀疏检索词项（权重 0.8）。None/空 dict 表示不做扩展。
        filter_keys: 允许从查询文本解析为 ``filters`` 的元数据键集合；
            None 使用内置集合（collection/doc_type/language/access_level/time_range）。
        stopwords: 关键词提取时过滤的停用词集合；None 使用内置英文停用词，
            传空集合 ``set()`` 则不过滤任何词。
    """

    name = "query_processor"

    DEFAULT_FILTER_KEYS = (
        "collection",
        "doc_type",
        "language",
        "access_level",
        "time_range",
    )

    def __init__(
        self,
        synonym_dict: dict[str, list[str]] | None = None,
        filter_keys: set[str] | None = None,
        stopwords: set[str] | None = None,
    ) -> None:
        self.synonym_dict = dict(synonym_dict or {})
        self.filter_keys = (
            set(filter_keys)
            if filter_keys is not None
            else set(self.DEFAULT_FILTER_KEYS)
        )
        self._stopwords = stopwords

    def process(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
    ) -> ProcessedQuery:
        """处理查询，返回 ``ProcessedQuery``。

        Args:
            query: 用户原始查询（Standalone Query）。
            filters: 显式结构化过滤条件（如 MCP 工具传入的 collection），
                优先级高于查询文本内联 ``key:value`` 约束。
            trace: 预留的 TraceContext（阶段 F 落地；纯本地规则无需外部调用）。
        """
        original = query if isinstance(query, str) else ""
        inline_filters, remainder = self._extract_filters(original)
        merged_filters = dict(filters or {})
        for key, value in inline_filters.items():
            merged_filters.setdefault(key, value)

        keywords = self._extract_keywords(remainder)
        sparse_terms = self._build_sparse_terms(keywords)

        return ProcessedQuery(
            original_query=original,
            keywords=keywords,
            sparse_terms=sparse_terms,
            dense_query=remainder,
            filters=merged_filters,
            method="rule",
        )

    # ---------- 内部实现 ----------

    def _extract_filters(self, text: str) -> tuple[dict[str, Any], str]:
        """从查询文本解析 ``key:value`` 约束并剥离，返回 ``(filters, 剩余文本)``。

        未知键或空值的 token 保持原文不动，避免误伤普通文本。
        """
        found: dict[str, Any] = {}

        def _replace(match: re.Match[str]) -> str:
            key = match.group("key")
            if key not in self.filter_keys:
                return match.group(0)
            value = match.group("value").rstrip(".,，。;；!！?？")
            parts = [p for p in value.split("|") if p]
            if not parts:
                return match.group(0)
            found[key] = parts if len(parts) > 1 else parts[0]
            return ""

        cleaned = _FILTER_TOKEN_RE.sub(_replace, text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return found, cleaned

    def _extract_keywords(self, text: str) -> list[str]:
        """提取关键词：与索引端一致的分词 → 去停用词 → 去重保序。"""
        seen: set[str] = set()
        keywords: list[str] = []
        for term in tokenize_text(text, self._stopwords):
            if term not in seen:
                seen.add(term)
                keywords.append(term)
        return keywords

    def _build_sparse_terms(self, keywords: list[str]) -> dict[str, float]:
        """构建稀疏检索词项及权重：原始关键词 1.0 + 同义词扩展 0.8。"""
        sparse_terms = {kw: _ORIGINAL_TERM_WEIGHT for kw in keywords}
        for kw in keywords:
            for alias in self.synonym_dict.get(kw, []):
                for term in tokenize_text(alias, self._stopwords):
                    sparse_terms.setdefault(term, _SYNONYM_TERM_WEIGHT)
        return sparse_terms
