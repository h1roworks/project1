"""OpenAI 兼容协议 LLM 的共享实现基类。

OpenAI / Azure / DeepSeek / DashScope 都遵循 OpenAI chat completions 协议，
差异只在客户端构造参数。本模块把 chat 调用、消息校验与错误包装抽成公共基类，
各 provider 子类只需实现 ``_build_client()``。
"""

from __future__ import annotations

from typing import Any

from libs.llm.base_llm import BaseLLM, ChatResponse, usage_to_dict


class OpenAICompatLLM(BaseLLM):
    """基于 OpenAI SDK 的兼容协议实现基类。

    构造时可注入 ``http_client``：测试中传入包装 ``httpx.MockTransport`` 的
    客户端即可 mock HTTP，不触发真实网络；生产环境为 None 时由 SDK 自动创建
    真实客户端（懒加载，首次 chat 时才建立）。
    """

    _default_base_url = "https://api.openai.com/v1"
    _timeout = 60.0

    def __init__(self, settings: Any, http_client: Any = None) -> None:
        self.settings = settings
        self.model = settings.model or settings.deployment_name
        self._http_client = http_client
        self._client: Any = None

    def _build_client(self) -> Any:
        """按 provider 构造 OpenAI/Azure SDK 客户端（由子类实现）。"""
        raise NotImplementedError

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
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
        request: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.settings.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.settings.max_tokens),
        }
        request.update(kwargs)  # 其余 provider 特有参数透传

        try:
            resp = self._get_client().chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - 统一包装为含 provider 与错误类型的可读错误
            raise RuntimeError(
                f"LLM 调用失败 (provider={self.provider}, "
                f"错误类型={type(exc).__name__}): {exc}"
            ) from exc

        return ChatResponse(
            content=(resp.choices[0].message.content or ""),
            model=getattr(resp, "model", "") or self.model,
            usage=usage_to_dict(getattr(resp, "usage", None)),
            raw=resp,
        )
