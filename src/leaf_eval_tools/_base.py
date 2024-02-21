from abc import abstractmethod
from typing import Any, List

from pydantic import BaseModel


class EvalToolConfig(BaseModel):
    pass


class EvalTool:
    config_cls = EvalToolConfig

    config: config_cls
    activated_metrics: List[str]

    def __init__(self, config: config_cls):
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


__all__ = ["EvalToolConfig", "EvalTool"]
