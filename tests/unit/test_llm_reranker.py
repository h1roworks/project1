"""B7.7: LLM Reranker 测试（注入 mock LLM，不触网）。

覆盖 prompt 构建、重排顺序、遗留候选、markdown 围栏容错、严格 schema 校验、
失败回退信号与工厂路由。
"""

import pytest

from core.settings import RerankSettings
from libs.llm.base_llm import ChatResponse
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.llm_reranker import LLMRerankError, LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class MockLLM:
    def __init__(self, content: str = "[]", exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[list[dict]] = []

    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return ChatResponse(content=self.content, model="mock")


def make_settings(**overrides) -> RerankSettings:
    base = {"provider": "llm", "enabled": True, "top_m": 10, "model": ""}
    base.update(overrides)
    return RerankSettings(**base)


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="苹果怎么种植", score=0.3),
        RerankCandidate(id="b", text="香蕉的产地", score=0.9),
        RerankCandidate(id="c", text="橙子的营养", score=0.5),
    ]


def make_reranker(content="[]", exc=None, prompt_path=None) -> LLMReranker:
    return LLMReranker(make_settings(), prompt_path=prompt_path, llm=MockLLM(content, exc))


# ---------- 工厂路由 ----------

def test_factory_creates_llm_reranker() -> None:
    r = RerankerFactory.create(make_settings())
    assert isinstance(r, LLMReranker)


# ---------- 重排逻辑 ----------

def test_rerank_reorders_by_llm_output() -> None:
    r = make_reranker('["c", "a", "b"]')
    result = r.rerank("水果种植", make_candidates())
    assert [c.id for c in result] == ["c", "a", "b"]
    assert result[0].score > result[1].score > result[2].score


def test_rerank_keeps_leftover_candidates() -> None:
    r = make_reranker('["a"]')
    result = r.rerank("水果种植", make_candidates())
    # 未被 LLM 提到的 b、c 保持原顺序排在末尾
    assert [c.id for c in result] == ["a", "b", "c"]


def test_markdown_fenced_json_accepted() -> None:
    r = make_reranker('```json\n["c", "a", "b"]\n```')
    assert [c.id for c in r.rerank("q", make_candidates())] == ["c", "a", "b"]


def test_duplicate_and_unknown_ids_skipped() -> None:
    r = make_reranker('["c", "c", "zzz", "a"]')
    assert [c.id for c in r.rerank("q", make_candidates())] == ["c", "a", "b"]


# ---------- prompt 构建 ----------

def test_prompt_path_injection(tmp_path) -> None:
    prompt = tmp_path / "rerank_custom.txt"
    prompt.write_text("INJECTED_RERANK_PROMPT", encoding="utf-8")
    r = make_reranker('["b", "a", "c"]', prompt_path=prompt)
    r.rerank("水果种植", make_candidates())
    messages = r.llm.calls[0]
    assert "INJECTED_RERANK_PROMPT" in messages[0]["content"]
    assert "苹果怎么种植" in messages[0]["content"]  # 候选文本进入 prompt


# ---------- 严格 schema 校验 ----------

def test_invalid_json_raises() -> None:
    r = make_reranker("这不是 JSON")
    with pytest.raises(LLMRerankError, match="JSON"):
        r.rerank("q", make_candidates())


def test_non_array_json_raises() -> None:
    r = make_reranker('{"ranked": ["a"]}')
    with pytest.raises(LLMRerankError, match="数组"):
        r.rerank("q", make_candidates())


def test_non_string_ids_raise() -> None:
    r = make_reranker("[1, 2]")
    with pytest.raises(LLMRerankError, match="字符串"):
        r.rerank("q", make_candidates())


# ---------- 失败回退信号 ----------

def test_llm_failure_raises_fallback_signal() -> None:
    r = make_reranker(exc=ConnectionError("llm down"))
    with pytest.raises(LLMRerankError, match="LLM 重排调用失败"):
        r.rerank("q", make_candidates())


def test_missing_llm_raises_fallback_signal() -> None:
    r = LLMReranker(make_settings())  # 未注入 llm，settings 也没有 .llm
    with pytest.raises(LLMRerankError, match="缺少 LLM"):
        r.rerank("q", make_candidates())


# ---------- 输入校验 ----------

def test_invalid_query_raises() -> None:
    r = make_reranker('["a"]')
    with pytest.raises(ValueError, match="query"):
        r.rerank("", make_candidates())


def test_invalid_candidates_raise() -> None:
    r = make_reranker('["a"]')
    with pytest.raises(ValueError, match="candidates"):
        r.rerank("q", [])
