"""Provides the MetricManager class for managing and retrieving evaluation 
metrics.

Exports:
    - MetricManager: A class to manage metrics used for model evaluation. It 
        supports both accessing the metric functions and the corresponding 
        scoring callables.
"""
from typing import Callable, List

from brisk.utility import MetricWrapper

class MetricManager:
    """A class to manage scoring metrics.

    This class provides access to various scoring metrics for regression tasks, 
    allowing retrieval by either their full names or common abbreviations.

    Attributes:
        scoring_metrics (dict): A dictionary storing the available metrics and 
            their corresponding callables.
    """
    def __init__(self, *metric_wrappers):
        """Initializes the MetricManager with a set of MetricWrapper instances.

        Args:
            metric_wrappers: Instances of MetricWrapper for each metric to 
            include.
        """
        self._metrics_by_name = {}
        self._abbreviations_to_name = {}
        for wrapper in metric_wrappers:
            self._add_metric(wrapper)

    def _add_metric(self, wrapper: MetricWrapper.MetricWrapper):
        # Remove old abbreviation
        if wrapper.name in self._metrics_by_name:
            old_wrapper = self._metrics_by_name[wrapper.name]
            if (old_wrapper.abbr
                and old_wrapper.abbr in self._abbreviations_to_name
                ):
                del self._abbreviations_to_name[old_wrapper.abbr]

        self._metrics_by_name[wrapper.name] = wrapper
        if wrapper.abbr:
            self._abbreviations_to_name[wrapper.abbr] = wrapper.name

    def _resolve_identifier(self, identifier: str) -> str:
        if identifier in self._metrics_by_name:
            return identifier
        if identifier in self._abbreviations_to_name:
            return self._abbreviations_to_name[identifier]
        raise ValueError(f"Metric '{identifier}' not found.")

    def get_metric(self, identifier: str) -> Callable:
        """Retrieve a metric function by its full name or abbreviation.

        Args:
            identifier (str): The full name or abbreviation of the metric.

        Returns:
            Callable: The metric function.

        Raises:
            ValueError: If the metric is not found.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].get_func_with_params()

    def get_scorer(self, identifier: str) -> Callable:
        """Retrieve a scoring callable by its full name or abbreviation.

        Args:
            identifier (str): The full name or abbreviation of the metric.

        Returns:
            Callable: The scoring callable.

        Raises:
            ValueError: If the scoring callable is not found.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].scorer

    def get_name(self, identifier: str) -> str:
        """Retrieve a metrics name, formatted for plots/tables, by its full 
        name or abbreviation.

        Args:
            identifier (str): The full name or abbreviation of the metric.

        Returns:
            str: The display name.

        Raises:
            ValueError: If the metric is not found.
        """
        name = self._resolve_identifier(identifier)
        return self._metrics_by_name[name].display_name

    def list_metrics(self) -> List[str]:
        return list(self._metrics_by_name.keys())
