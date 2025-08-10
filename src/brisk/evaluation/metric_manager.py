"""Provides the MetricManager class for managing evaluation metrics.

This module defines the MetricManager class, which manages metrics used for 
model evaluation. It supports both accessing metric functions and their 
corresponding scoring callables.
"""
from typing import Callable, List, Dict, Any

from brisk.evaluation import metric_wrapper

class MetricManager:
    """A class to manage scoring metrics.

    Provides access to various scoring metrics, allowing retrieval by either 
    their full names or common abbreviations.

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
    """
    def __init__(self, *metric_wrappers):
        self._metrics_by_name = {}
        self._abbreviations_to_name = {}
        self._display_name_to_name = {}
        for wrapper in metric_wrappers:
            self._add_metric(wrapper)

    def _add_metric(self, wrapper: metric_wrapper.MetricWrapper) -> None:
        """Add a new metric wrapper to the manager.

        Parameters
        ----------
        wrapper : MetricWrapper
            Metric wrapper to add
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

        Parameters
        ----------
        identifier : str
            Full name or abbreviation of the metric

        Returns
        -------
        str
            Full metric name

        Raises
        ------
        ValueError
            If metric identifier is not found
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

        Parameters
        ----------
        identifier : str
            Full name or abbreviation of the metric

        Returns
        -------
        callable
            The metric function

        Raises
        ------
        ValueError
            If metric is not found
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].get_func_with_params()

    def get_scorer(self, identifier: str) -> Callable:
        """Retrieve a scoring callable by name or abbreviation.

        Parameters
        ----------
        identifier : str
            Full name or abbreviation of the metric

        Returns
        -------
        callable
            The scoring callable

        Raises
        ------
        ValueError
            If scoring callable is not found
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].scorer

    def get_name(self, identifier: str) -> str:
        """Retrieve a metric's display name.

        Parameters
        ----------
        identifier : str
            Full name or abbreviation of the metric

        Returns
        -------
        str
            The formatted display name

        Raises
        ------
        ValueError
            If metric is not found
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].display_name

    def list_metrics(self) -> List[str]:
        """Get list of available metric names.

        Returns
        -------
        list of str
            List of available metric names
        """
        return list(self._metrics_by_name.keys())

    def set_split_metadata(self, split_metadata: Dict[str, Any]) -> None:
        """Set the split_metadata for all metrics.

        Parameters
        ----------
        split_metadata : dict
            Metadata to set for all metrics
        """
        for wrapper in self._metrics_by_name.values():
            wrapper.set_params(split_metadata=split_metadata)

    def is_higher_better(self, identifier: str) -> bool:
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].greater_is_better
