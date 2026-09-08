"""Chunk 智能重组 / 去噪（C5：ChunkRefiner）。

两级加工，逐级可回退（对齐 DEV_SPEC 3.1.1 "智能重组"）：
1. **规则去噪**（确定性、幂等）：剔除页码/换页符等噪声、规整空白、合并被物理切断
   的续行（段内合并，遇标题/列表/表格/代码块等结构行止步），确保每个 Chunk 是
   自包含的语义单元。
2. **可选 LLM 二次加工**：若配置 ``refine_with_llm`` 启用，调用 LLM 对片段做语义级
   精炼（读取 ``config/prompts/chunk_refinement.txt``）。LLM 不可用/失败/返回空时
   静默回退到规则结果并记录 ``metadata.llm_error``，绝不阻塞摄取流水线。

**不变式**：Transform 只改写 ``chunk.text`` 与 ``metadata``；``id`` 作为该 chunk
在文档中的稳定锚点（文档 ID + 序号 + 原文内容哈希）保持不变，``start_offset`` /
``end_offset`` 仍指向原文位置，仅作溯源参考。
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.settings import TransformSettings
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

# src/ingestion/transform/chunk_refiner.py → 上溯 3 层得到仓库根目录
_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "prompts" / "chunk_refinement.txt"
)

# 句末标点：以这些结尾的行视为完整句子，不再与下一行合并
_SENTENCE_END_RE = re.compile(r"[.。!！?？]$")

# 结构行（标题/列表/引用/表格行）：段内合并必须在此止步，避免破坏 Markdown 结构
_HARD_BREAK_RE = re.compile(r"^(\s*[-*+]\s|\s*\d+[.)]\s|#{1,6}\s|>\s|\|)")

# 整行为页码/页眉脚等噪声的启发式模式（PDF 解析常见残留）
_PAGE_NOISE_PATTERNS = [
    re.compile(r"^[-—]\s*\d+\s*[-—]$"),  # "- 3 -" / "— 4 —"
    re.compile(r"^第\s*\d+\s*页$"),  # "第 3 页"
    re.compile(r"^page\s*\d+(\s*of\s*\d+)?$", re.IGNORECASE),  # "Page 2 of 10"
    re.compile(r"^\d+\s*/\s*\d+$"),  # "3 / 10"
]


class ChunkRefiner(BaseTransform):
    """对 Chunk 做规则去噪（+ 可选 LLM 二次加工），产出自包含的语义单元。"""

    name = "chunk_refiner"

    def __init__(
        self,
        settings: TransformSettings | None = None,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """初始化。

        Args:
            settings: TransformSettings；也可传完整 Settings（自动取 ``.transform``
                子段，LLM 配置取 ``.llm``）；缺省用默认值（refine=True, LLM 关闭）。
            llm: 可注入的 LLM 实例（测试传 Fake）；为 None 时从 settings.llm 懒创建。
            prompt_path: chunk_refinement prompt 文件路径；测试中可注入替代文本。
        """
        settings = settings or TransformSettings()
        self._settings = settings.transform if hasattr(settings, "transform") else settings
        self._llm_settings = getattr(settings, "llm", None)
        self.llm = llm
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self._prompt_template: str | None = None

    def transform(self, chunk: Chunk, trace: Any = None) -> Chunk:
        cleaned = _rule_denoise(chunk.text)
        if cleaned.strip():
            new_text = cleaned
            denoised = cleaned != chunk.text
        else:
            new_text = chunk.text  # 整块都是噪声时保留原文，避免清空内容
            denoised = False

        metadata = dict(chunk.metadata)
        metadata["refined"] = True
        metadata["denoised"] = denoised
        metadata["refined_by_llm"] = False

        if self._settings.refine_with_llm:
            llm = self._get_llm()
            if llm is not None:
                try:
                    rewritten = self._llm_refine(llm, new_text, trace)
                    if rewritten:
                        new_text = rewritten
                        metadata["refined_by_llm"] = True
                except Exception as exc:  # noqa: BLE001 - LLM 失败降级为规则结果
                    metadata["llm_error"] = str(exc)

        return replace(chunk, text=new_text, metadata=metadata)

    # ---------- 可选 LLM 二次加工 ----------

    def _get_llm(self) -> BaseLLM | None:
        """返回 LLM 实例；未注入且 settings 无 llm 配置时返回 None（静默跳过）。"""
        if self.llm is None:
            llm_settings = self._llm_settings
            if llm_settings is not None and getattr(llm_settings, "provider", ""):
                self.llm = LLMFactory.create(llm_settings)
        return self.llm

    def _llm_refine(self, llm: BaseLLM, text: str, trace: Any) -> str:
        prompt = self._build_prompt(text)
        response = llm.chat([{"role": "user", "content": prompt}], trace=trace)
        result = (response.content or "").strip()
        if not result:
            raise ValueError("LLM 返回空文本")
        return result

    def _build_prompt(self, text: str) -> str:
        if self._prompt_template is None:
            path = self.prompt_path or Path(self._settings.prompt_path or _DEFAULT_PROMPT_PATH)
            self._prompt_template = path.read_text(encoding="utf-8")
        return (
            f"{self._prompt_template}\n\n原始文本块：\n{text}\n\n"
            "请只输出精炼后的文本块内容（不要任何解释或代码围栏）。"
        )


# ---------- 规则去噪 ----------


def _rule_denoise(text: str) -> str:
    """确定性规则去噪：换页符归一化 → 剔除页码噪声 → 段内续行合并 → 规整空白。"""
    text = text.replace("\x0c", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not _is_page_noise_line(ln)]
    merged = _merge_continuation_lines(lines)
    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _is_page_noise_line(line: str) -> bool:
    return any(p.search(line) for p in _PAGE_NOISE_PATTERNS)


def _is_hard_break(line: str) -> bool:
    """结构行（标题/列表/引用/表格行）：段内合并必须在此止步。"""
    return bool(_HARD_BREAK_RE.match(line))


def _ends_sentence(line: str) -> bool:
    return bool(_SENTENCE_END_RE.search(line.rstrip()))


def _merge_continuation_lines(lines: list[str]) -> list[str]:
    """把被物理切断的续行合并进同一段，遇空行/结构行/代码围栏则止步。

    代码围栏（``` / ~~~）内的行原样保留，避免破坏代码。结果幂等：同一输入
    再次执行产出相同输出。
    """
    result: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            result.append(line)
            i += 1
            continue

        if not stripped or in_fence or _is_hard_break(line) or _ends_sentence(line):
            result.append(line)
            i += 1
            continue

        # 不以句末标点结尾的普通行：与后续非空、非结构行合并为一段
        buffer = line
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            nxt_stripped = nxt.strip()
            if nxt_stripped.startswith("```") or nxt_stripped.startswith("~~~"):
                break
            if not nxt_stripped or _is_hard_break(nxt):
                break
            buffer = f"{buffer.rstrip()} {nxt_stripped}"
            if _ends_sentence(nxt):
                j += 1
                break
            j += 1
        result.append(buffer)
        i = j
    return result
