"""OpenAI 兼容 Vision LLM：支持 OpenAI 官方与 DashScope 等兼容服务。

阿里云百炼（provider=dashscope）通过 compatible-mode 提供 OpenAI 兼容的
多模态协议（如 qwen-vl-plus），因此同一个实现类注册到 openai 与 dashscope 两个名字。
"""

from __future__ import annotations

from typing import Any

from libs.llm._openai_vision_compat import OpenAICompatVisionLLM
from libs.llm.llm_factory import LLMFactory


class OpenAIVisionLLM(OpenAICompatVisionLLM):
    provider = "openai"
    _default_base_url = "https://api.openai.com/v1"

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url or self._default_base_url,
                timeout=self._timeout,
                http_client=self._http_client,
            )
        return self._client


LLMFactory.register_vision("openai", OpenAIVisionLLM)
LLMFactory.register_vision("dashscope", OpenAIVisionLLM)
