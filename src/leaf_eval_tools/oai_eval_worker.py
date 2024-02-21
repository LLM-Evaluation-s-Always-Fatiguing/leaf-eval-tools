from typing import Literal, Optional, Union

from jinja2 import Template
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import Field

from leaf_ai_backends.openai import OpenAIBackend, OpenAIBackendConfig, OpenAIClientConfig, AzureOpenAIClientConfig

from ._base import EvalTool, EvalToolConfig


class CustomOpenAIClientConfig(OpenAIClientConfig):
    chat_model: Literal[
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
    ] = Field(default="gpt-4-0125-preview")


class CustomAzureOpenAIClientConfig(AzureOpenAIClientConfig):
    chat_model: str = Field(default=...)


class CustomOpenAIBackendConfig(OpenAIBackendConfig):
    client_config: Union[
        CustomOpenAIClientConfig,
        CustomAzureOpenAIClientConfig,
    ] = Field(default=..., union_mode="smart")


class OaiEvalWorkerConfig(EvalToolConfig):
    ai_backend_config: CustomOpenAIBackendConfig = Field(default=...)
    temperature: float = Field(default=0.25)
    max_tokens: int = Field(default=256)
    response_format: Literal["text", "json_object"] = Field(default="text")


class OaiEvalWorker(EvalTool):
    config_obj = OaiEvalWorkerConfig
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config)
        self.ai_backend = OpenAIBackend(self.config.ai_backend_config)

    async def __call__(
        self,
        prompt_template: str,
        value_dict: dict,
        system_template: Optional[str] = None,
        system_value_dict: Optional[dict] = None
    ) -> str:
        if not system_value_dict:
            system_template = None
        messages = []
        if system_template:
            tpl = Template(system_template)
            messages.append(ChatCompletionSystemMessageParam(role="system", content=tpl.render(system_value_dict)))
        tpl = Template(prompt_template)
        messages.append(ChatCompletionUserMessageParam(role="user", content=tpl.render(value_dict)))

        result = (
            await self.ai_backend.async_client.chat.completions.create(
                messages=messages,
                model=self.config.ai_backend_config.chat_model,
                max_tokens=self.config.max_tokens,
                response_format={"type": self.config.response_format},
                temperature=self.config.temperature,
            )
        ).choices[0].message.content

        return result


__all__ = [
    "OaiEvalWorkerConfig",
    "OaiEvalWorker"
]
