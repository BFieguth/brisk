"""Base class for all evaluators."""
from abc import ABC
from typing import List, Dict, Any, Union

from sklearn import base

from brisk.evaluation.services.bundle import ServiceBundle
from brisk.evaluation.services import get_services

class BaseEvaluator(ABC):
    """Base class to enforce a common interface for all evaluators."""
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.services = get_services()
        self.metric_config = None

    @property
    def metadata(self):
        return self.services.metadata

    @property
    def io(self):
        return self.services.io

    @property
    def utility(self):
        return self.services.utility

    @property
    def logger(self):
        return self.services.logger

    def set_metric_config(self, metric_config):
        self.metric_config = metric_config

    def _generate_metadata(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        is_test: bool
    ) -> Dict[str, Any]:
        """Enforced: generate metadata for output."""
        return self.metadata.get_model(
            models, self.method_name, is_test
        )
