"""LLM 重排实现：读取 config/prompts/rerank.txt 构造 prompt，让 LLM 输出
按相关性降序排列的候选 id 列表，据此重排候选集。

输出被严格结构化校验（JSON 数组 + 字符串 id）；任何失败都抛出
``LLMRerankError``（可读的回退信号），供 Core 层 D6 捕获后回退融合排名。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory

# src/libs/reranker/llm_reranker.py → 上溯 3 层得到仓库根目录
_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[3] / "config" / "prompts" / "rerank.txt"


class LLMRerankError(RuntimeError):
    """LLM 重排失败的可读错误（D6 fallback 的触发信号）。"""


class LLMReranker(BaseReranker):
    backend = "llm"

    def __init__(
        self,
        settings: Any,
        prompt_path: str | Path | None = None,
        llm: Any = None,
    ) -> None:
        """初始化 LLM 重排器。

        Args:
            settings: RerankSettings（或携带 ``llm`` 配置的完整 Settings）。
            prompt_path: rerank prompt 文件路径；测试中可注入替代文本。
            llm: 可注入的 LLM 实例（mock）；为 None 时从 ``settings.llm`` 懒创建。
        """
        self.settings = settings
        self.prompt_path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
        self.llm = llm
        try:
            self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMRerankError(
                f"无法读取 rerank prompt 文件 {self.prompt_path}: {exc}"
            ) from exc

    def _get_llm(self) -> Any:
        if self.llm is None:
            llm_settings = getattr(self.settings, "llm", None)
            if llm_settings is None:
                raise LLMRerankError(
                    "LLM Reranker 缺少 LLM：请注入 llm 或在 settings.llm 中配置"
                )
            self.llm = LLMFactory.create(llm_settings)
        return self.llm

    def _build_rerank_prompt(
        self, query: str, candidates: list[RerankCandidate]
    ) -> str:
        lines = [
            f"ID: {c.id}\nText: {c.text}" for c in candidates
        ]
        return (
            f"{self.prompt_template}\n\nQuery: {query}\n\nCandidates:\n\n"
            + "\n\n".join(lines)
            + "\n\n只输出按相关性降序排列的 JSON 数组（候选 ID 列表）。"
        )

    def _parse_ranked_ids(self, response_text: str) -> list[str]:
        """解析并严格校验 LLM 输出为 JSON 字符串数组（ranked ids）。"""
        text = response_text.strip()
        if text.startswith("```"):  # 容忍模型把 JSON 包在 markdown 围栏里
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMRerankError(
                f"LLM 重排输出不是合法 JSON（前 200 字符）: {response_text[:200]}"
            ) from exc
        if not isinstance(parsed, list):
            raise LLMRerankError(
                f"LLM 重排输出必须是 JSON 数组（ranked ids），"
                f"得到 {type(parsed).__name__}"
            )
        if not all(isinstance(x, str) for x in parsed):
            raise LLMRerankError("LLM 重排输出数组的每个元素必须是字符串 id")
        return parsed

    def _map_results_to_candidates(
        self,
        ranked_ids: list[str],
        candidates: list[RerankCandidate],
    ) -> list[RerankCandidate]:
        id_to_candidate = {c.id: c for c in candidates}
        reranked: list[RerankCandidate] = []
        seen: set[str] = set()
        for rank, cid in enumerate(ranked_ids):
            if cid in seen or cid not in id_to_candidate:
                continue  # 重复 id 或不在候选集内的 id 直接跳过
            cand = id_to_candidate[cid]
            metadata = dict(cand.metadata)
            metadata["rerank_rank"] = rank
            reranked.append(
                RerankCandidate(
                    id=cand.id,
                    text=cand.text,
                    score=float(len(ranked_ids) - rank),
                    metadata=metadata,
                )
            )
            seen.add(cid)
        # 未被 LLM 提到的候选保持原顺序排在末尾（不丢数据）
        reranked.extend(c for c in candidates if c.id not in seen)
        return reranked

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any = None,
    ) -> list[RerankCandidate]:
        if not isinstance(query, str) or not query:
            raise ValueError("query 必须是非空字符串")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates 必须是非空 list")
        if len(candidates) == 1:
            return list(candidates)

        try:
            prompt = self._build_rerank_prompt(query, candidates)
            response = self._get_llm().chat(
                [{"role": "user", "content": prompt}], trace=trace
            )
            ranked_ids = self._parse_ranked_ids(response.content)
        except LLMRerankError:
            raise
        except Exception as exc:  # noqa: BLE001 - LLM 调用失败统一包装为回退信号
            raise LLMRerankError(f"LLM 重排调用失败: {exc}") from exc

        return self._map_results_to_candidates(ranked_ids, candidates)


RerankerFactory.register("llm", LLMReranker)
