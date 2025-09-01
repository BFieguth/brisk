"""Base class for all evaluators."""
from abc import ABC
from typing import List, Dict, Any, Union, Optional

from sklearn import base

class BaseEvaluator(ABC):
    """Base class to enforce a common interface for all evaluators.
    
    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    services : ServiceBundle
        The global services bundle
    metric_config : MetricManager
        The metric configuration manager
    metadata : MetadataService
        The metadata service
    io : IOService
        The I/O service
    utility : UtilityService
        The utility service
    logger : LoggingService
        The logging servicem
    reporting : ReportingService
        The reporting service
    """
    def __init__(self, method_name: str, description: str):
        self.method_name = method_name
        self.description = description
        self.services: Optional[Any] = None
        self.metric_config = None
        self.primary_color = "#1175D5" # Blue
        self.secondary_color = "#00A878" # Green
        self.accent_color = "#DE6B48" # Orange

    def set_services(self, services) -> None:
        """Set the services bundle for this evaluator.

        Parameters
        ----------
        services : ServiceBundle
            The global services bundle
        """
        self.services = services

    def set_colors(self, colors: dict) -> None:
        """Set colors from PlotSettings.
        
        Parameters
        ----------
        colors : dict
            Dictionary with 'primary_color', 'secondary_color', 'accent_color'
        """
        self.primary_color = colors.get("primary_color", self.primary_color)
        self.secondary_color = colors.get("secondary_color", self.secondary_color)  
        self.accent_color = colors.get("accent_color", self.accent_color)

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

    def set_metric_config(self, metric_config) -> None:
        """Set the metric configuration for this evaluator.

        Parameters
        ----------
        metric_config : MetricManager
            The metric configuration manager
        """
        self.metric_config = metric_config

    def _generate_metadata(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        is_test: bool
    ) -> Dict[str, Any]:
        """Enforced: generate metadata for output.

        Parameters
        ----------
        models : Union[base.BaseEstimator, List[base.BaseEstimator]]
            The model or list of models to generate metadata for
        is_test : bool
            Whether the model is a test model
        """
        return self.metadata.get_model(
            models, self.method_name, is_test
        )
