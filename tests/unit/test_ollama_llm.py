"""B7.2: Ollama LLM 测试（注入 Fake client，不触网）。"""

import pytest

from core.settings import LLMSettings
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM


class FakeOllamaClient:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[dict] = []

    def chat(self, model, messages, options=None, **kwargs) -> dict:
        self.calls.append({"model": model, "messages": messages, "options": options})
        if self.exc:
            raise self.exc
        return {
            "model": model,
            "message": {"role": "assistant", "content": "本地回复"},
            "prompt_eval_count": 3,
            "eval_count": 5,
        }


def make_settings(**overrides) -> LLMSettings:
    base = {
        "provider": "ollama",
        "model": "qwen2.5",
        "api_key": "",
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    base.update(overrides)
    return LLMSettings(**base)


def test_chat_success() -> None:
    fake = FakeOllamaClient()
    llm = OllamaLLM(make_settings(), client=fake)
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "本地回复"
    assert resp.model == "qwen2.5"
    call = fake.calls[0]
    assert call["model"] == "qwen2.5"
    assert call["options"]["temperature"] == 0.2
    assert call["options"]["num_predict"] == 512  # max_tokens 映射为 Ollama 的 num_predict


def test_chat_rejects_invalid_messages() -> None:
    llm = OllamaLLM(make_settings(), client=FakeOllamaClient())
    with pytest.raises(ValueError, match="messages"):
        llm.chat("oops")


def test_connection_failure_readable() -> None:
    llm = OllamaLLM(
        make_settings(),
        client=FakeOllamaClient(exc=ConnectionError("connection refused")),
    )
    with pytest.raises(RuntimeError, match="Ollama 调用失败") as exc_info:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "ConnectionError" in str(exc_info.value)
    # 错误信息不泄露敏感配置
    assert "api_key" not in str(exc_info.value)


def test_factory_creates() -> None:
    llm = LLMFactory.create(make_settings())
    assert isinstance(llm, OllamaLLM)
