"""Provides the MetricManager class for managing evaluation metrics.

This module defines the MetricManager class, which manages metrics used for
model evaluation. It supports both accessing metric functions and their
corresponding scoring callables, with support for built-in and custom metrics.
"""
from typing import Callable, List, Dict, Any

from brisk.evaluation import metric_wrapper
from brisk.defaults import regression_metrics, classification_metrics

class MetricManager:
    """A class to manage scoring metrics.

    Provides access to various scoring metrics, allowing retrieval by either
    their full names or common abbreviations. Supports both built-in and
    custom metrics with comprehensive name resolution and configuration export.

    Parameters
    ----------
    *metric_wrappers : MetricWrapper
        Instances of MetricWrapper for each metric to include

    Attributes
    ----------
    _metrics_by_name : dict
        Dictionary mapping metric names to MetricWrapper instances
    _abbreviations_to_name : dict
        Dictionary mapping metric abbreviations to full names
    _display_name_to_name : dict
        Dictionary mapping metric display names to full names

    Notes
    -----
    The MetricManager provides a unified interface for accessing metrics
    regardless of whether they are built-in or custom. It maintains
    mappings for name resolution and supports both function and scorer
    retrieval patterns.

    Examples
    --------
    Create a metric manager with built-in metrics:
        >>> from brisk.defaults import regression_metrics
        >>> manager = MetricManager(*regression_metrics.REGRESSION_METRICS)

    Get a metric function:
        >>> metric_func = manager.get_metric("mean_squared_error")

    Get a scorer callable:
        >>> scorer = manager.get_scorer("mse")  # Using abbreviation
    """

    def __init__(self, *metric_wrappers):
        """Initialize MetricManager with metric wrappers.

        Parameters
        ----------
        *metric_wrappers : MetricWrapper
            Instances of MetricWrapper for each metric to include

        Notes
        -----
        The constructor processes all provided metric wrappers and sets up
        the internal mappings for name resolution.
        """
        self._metrics_by_name = {}
        self._abbreviations_to_name = {}
        self._display_name_to_name = {}
        for wrapper in metric_wrappers:
            self._add_metric(wrapper)

    def _add_metric(self, wrapper: metric_wrapper.MetricWrapper) -> None:
        """Add a new metric wrapper to the manager.

        Adds a metric wrapper to the manager and updates all internal
        mappings. If a metric with the same name already exists, it
        replaces the old one and cleans up old mappings.

        Parameters
        ----------
        wrapper : MetricWrapper
            Metric wrapper to add

        Notes
        -----
        This method handles the replacement of existing metrics by cleaning
        up old abbreviation and display name mappings before adding the
        new metric wrapper.
        """
        # Remove old abbreviation
        if wrapper.name in self._metrics_by_name:
            old_wrapper = self._metrics_by_name[wrapper.name]
            if (old_wrapper.abbr
                and old_wrapper.abbr in self._abbreviations_to_name
                ):
                del self._abbreviations_to_name[old_wrapper.abbr]
            if (old_wrapper.display_name
                and old_wrapper.display_name in self._display_name_to_name
                ):
                del self._display_name_to_name[old_wrapper.display_name]

        self._metrics_by_name[wrapper.name] = wrapper
        if wrapper.abbr:
            self._abbreviations_to_name[wrapper.abbr] = wrapper.name
        if wrapper.display_name:
            self._display_name_to_name[wrapper.display_name] = wrapper.name

    def _resolve_identifier(self, identifier: str) -> str:
        """Resolve a metric identifier to its full name.

        Attempts to resolve the given identifier by checking full names,
        abbreviations, and display names in that order.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric

        Returns
        -------
        str
            Full metric name

        Raises
        ------
        ValueError
            If metric identifier is not found in any mapping

        Notes
        -----
        The resolution order is:
        1. Full name lookup
        2. Abbreviation lookup
        3. Display name lookup
        """
        if identifier in self._metrics_by_name:
            return identifier
        if identifier in self._abbreviations_to_name:
            return self._abbreviations_to_name[identifier]
        if identifier in self._display_name_to_name:
            return self._display_name_to_name[identifier]
        raise ValueError(f"Metric '{identifier}' not found.")

    def get_metric(self, identifier: str) -> Callable:
        """Retrieve a metric function by name or abbreviation.

        Gets the metric function with its parameters configured for the
        current context.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric

        Returns
        -------
        Callable
            The metric function with parameters configured

        Raises
        ------
        ValueError
            If metric identifier is not found

        Notes
        -----
        The returned function includes any parameters that were set
        via set_split_metadata or other configuration methods.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].get_func_with_params()

    def get_scorer(self, identifier: str) -> Callable:
        """Retrieve a scoring callable by name or abbreviation.

        Gets the scorer callable that can be used with scikit-learn's
        cross-validation and other evaluation functions.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric

        Returns
        -------
        Callable
            The scoring callable

        Raises
        ------
        ValueError
            If metric identifier is not found

        Notes
        -----
        The scorer is suitable for use with scikit-learn's make_scorer
        and cross-validation functions.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].scorer

    def get_name(self, identifier: str) -> str:
        """Retrieve a metric's display name.

        Gets the formatted display name for the metric, which is
        typically used for reporting and visualization.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric

        Returns
        -------
        str
            The formatted display name

        Raises
        ------
        ValueError
            If metric identifier is not found

        Notes
        -----
        The display name is typically a human-readable version of the
        metric name suitable for reports and plots.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].display_name

    def list_metrics(self) -> List[str]:
        """Get list of available metric names.

        Returns all metric names that are currently registered in the
        manager.

        Returns
        -------
        List[str]
            List of available metric names

        Notes
        -----
        This returns the full names of all registered metrics, not
        abbreviations or display names.
        """
        return list(self._metrics_by_name.keys())

    def set_split_metadata(self, split_metadata: Dict[str, Any]) -> None:
        """Set the split_metadata for all metrics.

        Updates all registered metrics with the provided split metadata,
        which may be used by metrics that need information about the
        data splits.

        Parameters
        ----------
        split_metadata : Dict[str, Any]
            Metadata to set for all metrics

        Returns
        -------
        None

        Notes
        -----
        This method propagates the split metadata to all registered
        metric wrappers, allowing them to use this information in
        their calculations.
        """
        for wrapper in self._metrics_by_name.values():
            wrapper.set_params(split_metadata=split_metadata)

    def is_higher_better(self, identifier: str) -> bool:
        """Determine if a higher value is better for this metric.

        Checks whether higher values of the metric indicate better
        performance, which is important for optimization and comparison.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric

        Returns
        -------
        bool
            True if a higher value is better for this metric, False otherwise

        Raises
        ------
        ValueError
            If metric identifier is not found

        Notes
        -----
        This information is crucial for optimization algorithms and
        model selection, as it determines the direction of optimization.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].greater_is_better

    def export_params(self) -> List[Dict[str, Any]]:
        """Export metric configuration data for rerun functionality.
        
        Detects built-in metric collections and exports custom metrics with
        their function definitions to enable exact rerun functionality.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of metric configurations that can be used to recreate the
            MetricManager

        Notes
        -----
        This method intelligently detects whether metrics are from built-in
        collections or are custom metrics. For built-in collections, it
        exports collection references. For custom metrics, it exports the
        full configuration including function definitions.

        The exported configuration can be used to recreate the exact same
        MetricManager instance, enabling reproducible experiments.
        """
        regression_names = {
            wrapper.name for wrapper in regression_metrics.REGRESSION_METRICS
        }
        classification_names = {
            wrapper.name
            for wrapper in classification_metrics.CLASSIFICATION_METRICS
        }

        found_regression_metrics = set()
        found_classification_metrics = set()
        custom_metrics = []

        for name, wrapper in self._metrics_by_name.items():
            if name in regression_names:
                found_regression_metrics.add(name)
            elif name in classification_names:
                found_classification_metrics.add(name)
            else:
                custom_metrics.append(wrapper.export_config())

        result = []
        if found_regression_metrics == regression_names:
            result.append({
                "type": "builtin_collection",
                "collection": "brisk.REGRESSION_METRICS"
            })
        elif found_regression_metrics:
            for name in found_regression_metrics:
                result.append({
                    "type": "builtin_metric",
                    "collection": "brisk.REGRESSION_METRICS",
                    "name": name
                })

        if found_classification_metrics == classification_names:
            result.append({
                "type": "builtin_collection",
                "collection": "brisk.CLASSIFICATION_METRICS"
            })
        elif found_classification_metrics:
            for name in found_classification_metrics:
                result.append({
                    "type": "builtin_metric",
                    "collection": "brisk.CLASSIFICATION_METRICS",
                    "name": name
                })

        for custom_config in custom_metrics:
            custom_config["type"] = "custom_metric"
            result.append(custom_config)

        return result
