"""Base class for all evaluators.

This module provides the BaseEvaluator abstract base class that defines the
common interface for all evaluators in the Brisk framework. It provides
access to services, color management, and metadata generation functionality.
"""
import abc
from typing import List, Dict, Any, Union, Optional

from sklearn import base

class BaseEvaluator(abc.ABC):
    """Base class to enforce a common interface for all evaluators.
    
    Provides a common interface and shared functionality for all evaluators
    in the Brisk framework. Includes access to services, color management,
    and metadata generation capabilities.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    services : ServiceBundle or None
        The global services bundle (set via set_services)
    metric_config : MetricManager or None
        The metric configuration manager (set via set_metric_config)
    primary_color : str
        Primary color for plots and visualizations (default: "#1175D5")
    secondary_color : str
        Secondary color for plots and visualizations (default: "#00A878")
    accent_color : str
        Accent color for plots and visualizations (default: "#DE6B48")

    Notes
    -----
    This abstract base class provides the foundation for all evaluators in
    the Brisk framework. It ensures consistent access to services and
    provides common functionality for color management and metadata generation.

    Subclasses must implement their specific evaluation logic while following
    the interface defined by this base class.

    Examples
    --------
    Create a custom evaluator:
        >>> class CustomEvaluator(BaseEvaluator):
        ...     def __init__(self):
        ...         super().__init__("custom", "Custom evaluation method")
        ...     
        ...     def evaluate(self, model, X, y):
        ...         # Custom evaluation logic
        ...         pass
    """

    def __init__(self, method_name: str, description: str):
        """Initialize BaseEvaluator with method name and description.

        Parameters
        ----------
        method_name : str
            The name of the evaluator
        description : str
            The description of the evaluator output

        Notes
        -----
        The constructor initializes the evaluator with its identifying
        information and sets up default color scheme. Services and metric
        configuration must be set separately using set_services() and
        set_metric_config() methods.
        """
        self.method_name = method_name
        self.description = description
        self.services: Optional[Any] = None
        self.metric_config = None
        self.primary_color = "#1175D5" # Blue
        self.secondary_color = "#00A878" # Green
        self.accent_color = "#DE6B48" # Orange

    def set_services(self, services) -> None:
        """Set the services bundle for this evaluator.

        Configures the evaluator with access to the global services bundle,
        which provides access to metadata, I/O, utility, logging, and
        reporting services.

        Parameters
        ----------
        services : ServiceBundle
            The global services bundle

        Returns
        -------
        None

        Notes
        -----
        This method must be called before using any service properties
        (metadata, io, utility, logger, reporting). The services bundle
        provides access to all shared functionality in the Brisk framework.
        """
        self.services = services

    def set_colors(self, colors: dict) -> None:
        """Set colors from PlotSettings.
        
        Updates the evaluator's color scheme with values from the plot
        settings. This allows for consistent theming across all evaluators.

        Parameters
        ----------
        colors : dict
            Dictionary with 'primary_color', 'secondary_color', 'accent_color'
            keys. Missing keys will use current values.

        Returns
        -------
        None

        Notes
        -----
        This method is typically called during evaluator initialization
        to apply the global color theme. Only provided color keys are
        updated; missing keys retain their current values.
        """
        self.primary_color = colors.get("primary_color", self.primary_color)
        self.secondary_color = colors.get(
            "secondary_color", self.secondary_color
        )
        self.accent_color = colors.get("accent_color", self.accent_color)

    @property
    def metadata(self):
        """Access to the metadata service.

        Returns
        -------
        MetadataService
            The metadata service for generating model metadata

        Raises
        ------
        RuntimeError
            If services have not been set via set_services()

        Notes
        -----
        This property provides access to the metadata service, which is
        used for generating metadata about models and evaluation results.
        """
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.metadata

    @property
    def io(self):
        """Access to the I/O service.

        Returns
        -------
        IOService
            The I/O service for file operations

        Raises
        ------
        RuntimeError
            If services have not been set via set_services()

        Notes
        -----
        This property provides access to the I/O service, which handles
        file reading, writing, and other I/O operations.
        """
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.io

    @property
    def utility(self):
        """Access to the utility service.

        Returns
        -------
        UtilityService
            The utility service for common operations

        Raises
        ------
        RuntimeError
            If services have not been set via set_services()

        Notes
        -----
        This property provides access to the utility service, which provides
        common utility functions and operations used across the framework.
        """
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.utility

    @property
    def logger(self):
        """Access to the logging service.

        Returns
        -------
        LoggingService
            The logging service for logging operations

        Raises
        ------
        RuntimeError
            If services have not been set via set_services()

        Notes
        -----
        This property provides access to the logging service, which handles
        all logging operations throughout the framework.
        """
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.logger

    @property
    def reporting(self):
        """Access to the reporting service.

        Returns
        -------
        ReportingService
            The reporting service for report generation

        Raises
        ------
        RuntimeError
            If services have not been set via set_services()

        Notes
        -----
        This property provides access to the reporting service, which handles
        report generation and formatting.
        """
        if self.services is None:
            raise RuntimeError("Services not set. Call set_services() first.")
        return self.services.reporting

    def set_metric_config(self, metric_config) -> None:
        """Set the metric configuration for this evaluator.

        Configures the evaluator with access to the metric configuration
        manager, which provides access to evaluation metrics.

        Parameters
        ----------
        metric_config : MetricManager
            The metric configuration manager

        Returns
        -------
        None

        Notes
        -----
        This method must be called before the evaluator can access metrics.
        The metric configuration manager provides access to all available
        evaluation metrics and their configurations.
        """
        self.metric_config = metric_config

    def _generate_metadata(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        is_test: bool
    ) -> Dict[str, Any]:
        """Generate metadata for output.

        Generates metadata for the given model(s) using the metadata service.
        This method provides a standardized way to generate metadata for
        evaluation outputs.

        Parameters
        ----------
        models : Union[base.BaseEstimator, List[base.BaseEstimator]]
            The model or list of models to generate metadata for
        is_test : bool
            Whether the model is a test model

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata about the model(s)

        Notes
        -----
        This method delegates to the metadata service to generate appropriate
        metadata for the given model(s). The metadata includes information
        about the model type, parameters, and other relevant details.
        """
        return self.metadata.get_model(
            models, self.method_name, is_test
        )
