"""Base class for all evaluators."""
from abc import ABC
from typing import List, Dict, Any, Union, Optional

from sklearn import base

# from brisk.services import get_services

class BaseEvaluator(ABC):
    """Base class to enforce a common interface for all evaluators."""
    def __init__(self, method_name: str, description: str):
        self.method_name = method_name
        self.description = description
        # self.services = get_services()
        self.services: Optional[Any] = None
        self.metric_config = None

    def set_services(self, services):
        """Set the services bundle for this evaluator."""
        self.services = services

    @property
    def metadata(self):
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.metadata

    @property
    def io(self):
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.io

    @property
    def utility(self):
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.utility

    @property
    def logger(self):
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.logger
    
    @property
    def reporting(self):
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.reporting

    # @property
    # def metadata(self):
    #     return self.services.metadata

    # @property
    # def io(self):
    #     return self.services.io

    # @property
    # def utility(self):
    #     return self.services.utility

    # @property
    # def logger(self):
    #     return self.services.logger
    
    # @property
    # def reporting(self):
    #     return self.services.reporting

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
