"""Base class for all evaluators."""
from abc import ABC
from typing import List, Dict, Any, Union

from sklearn import base

from brisk.evaluation.services.bundle import ServiceBundle

class BaseEvaluator(ABC):
    """Base class to enforce a common interface for all evaluators."""
    def __init__(self, method_name: str, services: ServiceBundle):
        self.method_name = method_name
        self.services = services

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

    @property
    def metric_config(self):
        return self.services.metric_config

    def _generate_metadata(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        is_test: bool
    ) -> Dict[str, Any]:
        """Enforced: generate metadata for output."""
        return self.metadata.get_metadata(
            models, self.method_name, is_test
        )
