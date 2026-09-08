"""DashScope LLM：阿里云百炼（DashScope）的 BaseLLM 实现。

DashScope 提供 OpenAI 兼容协议（``compatible-mode``），因此复用
``OpenAICompatLLM``，仅默认 base_url 不同。
"""

from __future__ import annotations

from typing import Any

from libs.llm._openai_compat import OpenAICompatLLM
from libs.llm.llm_factory import LLMFactory


class DashScopeLLM(OpenAICompatLLM):
    provider = "dashscope"
    _default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _build_client(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or self._default_base_url,
            timeout=self._timeout,
            http_client=self._http_client,
        )


LLMFactory.register("dashscope", DashScopeLLM)
