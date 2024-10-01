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
    def __init__(self, *metric_dicts):
        """Initializes the MetricManager with a set of metrics.

        Args:
            include_regression (bool): Whether to include regression metrics. 
                Defaults to True.
        """
        self.scoring_metrics = {}
        for metric_dict in metric_dicts:
            self.scoring_metrics.update(metric_dict)

    def get_metric(self, name_or_abbr: str) -> Callable:
        """Retrieve a metric function by its full name or abbreviation.

        Args:
            name_or_abbr (str): The full name or abbreviation of the metric.

        Returns:
            Callable: The metric function.

        Raises:
            ValueError: If the metric is not found.
        """
        if name_or_abbr in self.scoring_metrics:
            return self.scoring_metrics[name_or_abbr]['func']
        
        for full_name, details in self.scoring_metrics.items():
            if details.get("abbr") == name_or_abbr:
                return details['func']
        
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
        if name_or_abbr in self.scoring_metrics:
            return self.scoring_metrics[name_or_abbr]['scorer']
        
        for full_name, details in self.scoring_metrics.items():
            if details.get("abbr") == name_or_abbr:
                return details['scorer']
        
        raise ValueError(f"Scoring callable '{name_or_abbr}' not found.")

    def get_name(self, name_or_abbr: str) -> Callable:
        """Retrieve a metrics name, formatted for plots/tables, by its full name or abbreviation.

        Args:
            name_or_abbr (str): The full name or abbreviation of the metric.

        Returns:
            Callable: The scoring callable.

        Raises:
            ValueError: If the scoring callable is not found.
        """
        if name_or_abbr in self.scoring_metrics:
            return self.scoring_metrics[name_or_abbr]['display_name']
        
        for full_name, details in self.scoring_metrics.items():
            if details.get("abbr") == name_or_abbr:
                return details['display_name']
        
        raise ValueError(f"Scoring callable '{name_or_abbr}' not found.")
    