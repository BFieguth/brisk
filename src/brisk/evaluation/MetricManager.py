"""Provides the MetricManager class for managing and retrieving evaluation metrics.

Exports:
    - MetricManager: A class to manage metrics used for model evaluation. It 
        supports both accessing the metric functions and the corresponding 
        scoring callables.
"""
from typing import Callable

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
            metric_wrappers: Instances of MetricWrapper for each metric to include.
        """
        self.metrics = {}
        for wrapper in metric_wrappers:
            self.metrics[wrapper.name] = wrapper
            if wrapper.abbr:
                self.metrics[wrapper.abbr] = wrapper

    def get_metric(self, name_or_abbr: str) -> Callable:
        """Retrieve a metric function by its full name or abbreviation.

        Args:
            name_or_abbr (str): The full name or abbreviation of the metric.

        Returns:
            Callable: The metric function.

        Raises:
            ValueError: If the metric is not found.
        """
        if name_or_abbr in self.metrics:
            return self.metrics[name_or_abbr].func
        raise ValueError(f"Metric function '{name_or_abbr}' not found.")

    def get_scorer(self, name_or_abbr: str) -> Callable:
        """Retrieve a scoring callable by its full name or abbreviation.

        Args:
            name_or_abbr (str): The full name or abbreviation of the metric.

        Returns:
            Callable: The scoring callable.

        Raises:
            ValueError: If the scoring callable is not found.
        """
        if name_or_abbr in self.metrics:
            return self.metrics[name_or_abbr].scorer       
        raise ValueError(f"Scoring callable '{name_or_abbr}' not found.")

    def get_name(self, name_or_abbr: str) -> str:
        """Retrieve a metrics name, formatted for plots/tables, by its full name or abbreviation.

        Args:
            name_or_abbr (str): The full name or abbreviation of the metric.

        Returns:
            str: The display name.

        Raises:
            ValueError: If the metric is not found.
        """
        if name_or_abbr in self.metrics:
            return self.metrics[name_or_abbr].display_name     
        raise ValueError(f"Metric '{name_or_abbr}' not found.")
    