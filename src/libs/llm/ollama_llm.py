"""Ollama LLM：本地 Ollama 服务的 BaseLLM 实现。

消息格式与 OpenAI 一致（``[{"role": "user", "content": "..."}]``）。
``ollama`` 包在首次真正调用时才导入（懒加载），测试中可注入 Fake client。
"""

from __future__ import annotations

from typing import Any

from libs.llm.base_llm import BaseLLM, ChatResponse, usage_to_dict
from libs.llm.llm_factory import LLMFactory


class OllamaLLM(BaseLLM):
    provider = "ollama"
    _default_host = "http://localhost:11434"

    def __init__(self, settings: Any, client: Any = None) -> None:
        self.settings = settings
        self.model = settings.model
        self._host = settings.base_url or self._default_host
        self._client = client  # 测试注入的 Fake client；None 时懒加载真实客户端

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._host)
        return self._client

    def _validate_messages(self, messages: Any) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"messages 必须是非空 list（provider={self.provider}）")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"每条消息必须是包含 role/content 字段的 dict"
                    f"（provider={self.provider}）"
                )

    def chat(
        self,
        messages: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self._validate_messages(messages)
        options: dict[str, Any] = {
            "temperature": kwargs.pop("temperature", self.settings.temperature),
        }
        max_tokens = kwargs.pop("max_tokens", self.settings.max_tokens)
        if max_tokens:
            options["num_predict"] = max_tokens  # Ollama 用 num_predict 表示生成上限

        try:
            resp = self._get_client().chat(
                model=kwargs.pop("model", self.model),
                messages=messages,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001 - 连接失败/超时统一包装为可读错误
            raise RuntimeError(
                f"Ollama 调用失败 (provider={self.provider}, "
                f"错误类型={type(exc).__name__}): {exc}"
            ) from exc

        usage = {
            "prompt_tokens": resp.get("prompt_eval_count"),
            "completion_tokens": resp.get("eval_count"),
        }
        return ChatResponse(
            content=resp["message"]["content"],
            model=self.model,
            usage=usage_to_dict(usage),
            raw=resp,
        )


LLMFactory.register("ollama", OllamaLLM)
