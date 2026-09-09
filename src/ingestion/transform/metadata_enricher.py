"""语义元数据增强（C6：MetadataEnricher）。

两级加工，逐级可回退（对齐 DEV_SPEC 3.1.1 "语义元数据注入"）：
1. **规则增强**（确定性、幂等）：为每个 Chunk 生成 ``title``（小标题）、``summary``
   （摘要）与 ``tags``（主题标签）的兜底结果，保证元数据字段始终非空。
2. **可选 LLM 增强**（核心）：若配置 ``enrich_with_llm`` 启用，调用 LLM 对 Chunk
   做语义级生成（读取 ``config/prompts/metadata_enrichment.txt``，要求返回严格 JSON）。
   LLM 不可用/失败/返回空时静默回退到规则结果并记录 ``metadata.llm_error``，
   绝不阻塞摄取流水线。

**不变式**：Transform 只改写 ``chunk.metadata``；``id``/``text``/``start_offset`` /
``end_offset`` / ``source_ref`` 保持不变。规则增强幂等：同一输入重复执行产出相同结果。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.settings import TransformSettings
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

# src/ingestion/transform/metadata_enricher.py → 上溯 3 层得到仓库根目录
_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "prompts" / "metadata_enrichment.txt"
)

# 词元抽取：英文词（≥2 字符）/ 中文连续串（≥2 字）
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[一-鿿]{2,}")

# 常见停用词：既不是主题也不是检索关键词，标签抽取时跳过
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "this", "that", "from",
    "have", "has", "had", "not", "but", "its", "our", "you", "your", "all",
    "can", "will", "into", "about", "than", "then", "them", "they", "which",
    "what", "when", "where", "who", "whom", "how", "also", "their", "there",
    "been", "being", "one", "two", "more", "most", "some", "such", "only",
    "very", "just", "out", "over", "other", "these", "those", "each", "any",
    "could", "would", "should", "may", "might", "must", "shall", "off", "under",
    "——", "可以", "以及", "或者", "还是", "但是", "而且", "如果", "因为", "所以",
    "这个", "那个", "这些", "那些", "对于", "关于", "根据", "通过", "进行", "已经",
    "之后", "之前", "同时", "仍然", "一直", "就是", "而是", "并且", "其中",
}

# 行首 Markdown 标题标记（# 或 ## ...）
_HEADING_MARKER_RE = re.compile(r"^#{1,6}\s+")

# 句末标点：summary 取第一句时在此截断
_SENTENCE_END_RE = re.compile(r"[.。！!?？]")

_TITLE_MAX = 80  # 规则 title 最长字符数
_SUMMARY_MAX = 200  # 规则 summary 最长字符数
_DEFAULT_TAGS_MAX = 5  # 规则 tags 最多数量


class MetadataEnricher(BaseTransform):
    """为 Chunk 注入 title/summary/tags 语义元数据（规则兜底 + 可选 LLM 增强）。"""

    name = "metadata_enricher"

    def __init__(
        self,
        settings: TransformSettings | None = None,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """初始化。

        Args:
            settings: TransformSettings；也可传完整 Settings（自动取 ``.transform``
                子段，LLM 配置取 ``.llm``）；缺省用默认值（LLM 增强关闭）。
            llm: 可注入的 LLM 实例（测试传 Fake）；为 None 时从 settings.llm 懒创建。
            prompt_path: metadata_enrichment prompt 文件路径；测试中可注入替代文本。
        """
        settings = settings or TransformSettings()
        self._settings = settings.transform if hasattr(settings, "transform") else settings
        self._llm_settings = getattr(settings, "llm", None)
        self.llm = llm
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self._prompt_template: str | None = None

    def transform(self, chunk: Chunk, trace: Any = None) -> Chunk:
        title, summary, tags = _rule_enrich(chunk.text)

        metadata = dict(chunk.metadata)
        metadata["enriched"] = True
        metadata["enriched_by_llm"] = False

        if self._settings.enrich_with_llm:
            try:
                llm = self._get_llm()
                if llm is not None:
                    title, summary, tags = self._llm_enrich(llm, chunk.text, trace)
                    metadata["enriched_by_llm"] = True
            except Exception as exc:  # noqa: BLE001 - LLM 创建/调用失败一律降级为规则结果
                metadata["llm_error"] = str(exc)

        metadata["title"] = title
        metadata["summary"] = summary
        metadata["tags"] = tags
        return replace(chunk, metadata=metadata)

    # ---------- 可选 LLM 增强 ----------

    def _get_llm(self) -> BaseLLM | None:
        """返回 LLM 实例；未注入且 settings 无 llm 配置时返回 None（静默跳过）。"""
        if self.llm is None:
            llm_settings = self._llm_settings
            if llm_settings is not None and getattr(llm_settings, "provider", ""):
                self.llm = LLMFactory.create(llm_settings)
        return self.llm

    def _llm_enrich(self, llm: BaseLLM, text: str, trace: Any) -> tuple[str, str, list[str]]:
        prompt = self._build_prompt(text)
        response = llm.chat([{"role": "user", "content": prompt}], trace=trace)
        result = (response.content or "").strip()
        if not result:
            raise ValueError("LLM 返回空文本")
        return _parse_enriched(result)

    def _build_prompt(self, text: str) -> str:
        if self._prompt_template is None:
            path = self.prompt_path or Path(self._settings.prompt_path or _DEFAULT_PROMPT_PATH)
            self._prompt_template = path.read_text(encoding="utf-8")
        return f"{self._prompt_template}\n\n文本块内容：\n{text}"


# ---------- 规则增强 ----------


def _rule_enrich(text: str) -> tuple[str, str, list[str]]:
    """确定性规则生成 title/summary/tags，保证输出非空（LLM 降级兜底）。"""
    content = text.strip()
    if not content:
        return "Untitled Chunk", "（空内容）", ["empty"]
    return _rule_title(content), _rule_summary(content), _rule_tags(content)


def _rule_title(text: str) -> str:
    """取首个非空行作为 title；若是 Markdown 标题则去掉井号前缀，超长截断。"""
    for line in text.split("\n"):
        line = _HEADING_MARKER_RE.sub("", line.strip())
        if line:
            return _truncate(line, _TITLE_MAX)
    return "Untitled Chunk"


def _rule_summary(text: str) -> str:
    """取首句作为 summary（到第一个句末标点截止）；无句末标点则取整个文本。"""
    match = _SENTENCE_END_RE.search(text)
    summary = text[: match.end()] if match else text
    summary = _truncate(summary.strip(), _SUMMARY_MAX)
    return summary or "（无摘要）"


def _rule_tags(text: str, max_tags: int = _DEFAULT_TAGS_MAX) -> list[str]:
    """抽取去重后的主题标签；全为停用词时回退到 title 词元，再兜底 "general"。"""
    tags = _extract_tokens(text, max_tags)
    if tags:
        return tags
    title = _rule_title(text)
    tags = _extract_tokens(title, max_tags)
    return tags or ["general"]


def _extract_tokens(text: str, max_tags: int) -> list[str]:
    """按出现顺序抽取非停用词词元，去重并限数。"""
    seen: set[str] = set()
    tags: list[str] = []
    for word in _TOKEN_RE.findall(text):
        key = word.lower()
        if key in _STOPWORDS or key in seen:
            continue
        seen.add(key)
        tags.append(word)
        if len(tags) >= max_tags:
            break
    return tags


def _truncate(text: str, limit: int) -> str:
    """超长截断并加省略号；不切断字符（Unicode 安全）。"""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


# ---------- LLM 输出解析 ----------


def _parse_enriched(response: str) -> tuple[str, str, list[str]]:
    """把 LLM 的严格 JSON 输出解析为 (title, summary, tags)；非法则抛 ValueError。"""
    start, end = response.find("{"), response.rfind("}")
    if start == -1 or end < start:
        raise ValueError("LLM 输出中未找到 JSON 对象")
    try:
        data = json.loads(response[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM 输出不是合法 JSON: {exc}") from exc

    title = str(data.get("title") or "").strip()
    summary = str(data.get("summary") or "").strip()
    raw_tags = data.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = [str(t).strip() for t in raw_tags if str(t).strip()]

    if not title or not summary or not tags:
        raise ValueError("LLM 输出缺少必要字段 (title/summary/tags)")
    return title, summary, tags
