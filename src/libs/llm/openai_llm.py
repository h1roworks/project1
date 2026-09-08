"""OpenAI LLM：OpenAI 官方 API 的 BaseLLM 实现。"""

from __future__ import annotations

from typing import Any

from libs.llm._openai_compat import OpenAICompatLLM
from libs.llm.llm_factory import LLMFactory


class OpenAILLM(OpenAICompatLLM):
    provider = "openai"
    _default_base_url = "https://api.openai.com/v1"

    def _build_client(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or self._default_base_url,
            timeout=self._timeout,
            http_client=self._http_client,
        )


LLMFactory.register("openai", OpenAILLM)
