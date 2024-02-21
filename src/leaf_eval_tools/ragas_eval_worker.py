from typing import Any, Dict, Literal, List, Optional, Union

from datasets import Dataset
from ragas import evaluate
from ragas.evaluation import Result
from ragas.metrics.base import Metric

from ._base import EvalTool, EvalToolConfig


class RagasEvalWorkerConfig(EvalToolConfig):
    pass


class RagasEvalWorker(EvalTool):
    config_cls = RagasEvalWorkerConfig

    config: config_cls
    activated_metrics: List[Metric]

    def __init__(self, config: config_cls, activated_metrics: List[Metric]):
        super().__init__(config=config)
        self.activated_metrics = activated_metrics

    def __call__(self, dataset: Dataset) -> Result:
        return evaluate(dataset, metrics=self.activated_metrics)


__all__ = [
    "RagasEvalWorkerConfig",
    "RagasEvalWorker"
]
