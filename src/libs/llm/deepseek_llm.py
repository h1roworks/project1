"""DeepSeek LLM：DeepSeek API（OpenAI 兼容协议）的 BaseLLM 实现。"""

from __future__ import annotations

from typing import Any

from libs.llm._openai_compat import OpenAICompatLLM
from libs.llm.llm_factory import LLMFactory


class DeepSeekLLM(OpenAICompatLLM):
    provider = "deepseek"
    _default_base_url = "https://api.deepseek.com"

    def _build_client(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or self._default_base_url,
            timeout=self._timeout,
            http_client=self._http_client,
        )


LLMFactory.register("deepseek", DeepSeekLLM)
