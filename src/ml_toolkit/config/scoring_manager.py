"""
Dictionaries to import and access a variety of scoring methods
"""
from typing import Callable

import numpy as np 
import scipy
import sklearn.metrics as metrics
import sklearn.metrics._regression as regression


class ScoringManager:
    """A class to manage scoring metrics for different tasks.

    This class provides access to various scoring metrics for regression 
    tasks by either their full names or common abbreviations.
    """
    def __init__(self, include_regression: bool = True):
        """Initializes the scoring manager with the specified metrics.

        Args:
            include_regression (bool): Whether to include regression metrics.
        """
        self.scoring_metrics = {}

        if include_regression:
            self._add_regression_metrics()

    def _add_regression_metrics(self):
        """Add commmon regression metrics and abbreviations to scoring manager."""
        CCC = metrics.make_scorer(self.__concordance_correlation_coefficient)

        REGRESSION_SCORING = {
            "explained_variance_score": {
                "func": regression.explained_variance_score
            },
            "max_error": {
                "func": regression.max_error
            },
            "mean_absolute_error": {
                "abbr": "MAE",
                "func": regression.mean_absolute_error
            },
            "mean_absolute_percentage_error": {
                "abbr": "MAPE",
                "func": regression.mean_absolute_percentage_error
            },
            "mean_pinball_loss": {
                "func": regression.mean_pinball_loss
            },
            "mean_squared_error": {
                "abbr": "MSE",
                "func": regression.mean_squared_error
            },
            "mean_squared_log_error": {
                "func": regression.mean_squared_log_error
            },
            "median_absolute_error": {
                "func": regression.median_absolute_error
            },
            "r2_score": {
                "abbr": "R2",
                "func": regression.r2_score
            },
            "root_mean_squared_error": {
                "abbr": "RMSE",
                "func": regression.root_mean_squared_error
            },
            "root_mean_squared_log_error": {
                "func": regression.root_mean_squared_log_error
            },
            "concordance_correlation_coefficient": {
                "abbr": "CCC",
                "func": CCC
            }
        }
        self.scoring_metrics.update(REGRESSION_SCORING)

    def get_scorer(self, name_or_abbr: str) -> Callable:
        """Retrieve a scoring function by full name or abbreviation.
        
        Args:
            name_or_abbr (str): The full name or abbreviation of the scoring metric.
        
        Returns:
            The scoring function or None if not found.
        """
        if name_or_abbr in self.scoring_metrics:
            return self.scoring_metrics[name_or_abbr]['func']
        
        for full_name, details in self.scoring_metrics.items():
            if details.get("abbr") == name_or_abbr:
                return details['func']
        
        raise ValueError(f"Scorer '{name_or_abbr}' not found.")

    # Define additional scoring metrics not included with sklearn
    @staticmethod
    def __concordance_correlation_coefficient(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Lin's Concordance Correlation Coefficient (CCC)

        Args:
            y_true (array-like): The true (observed) values.
            y_pred (array-like): The predicted values.

        Returns:
            float: Concordance Correlation Coefficient between y_true and y_pred.
        """
        corr, _ = scipy.stats.pearsonr(y_true, y_pred)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)
        numerator = 2 * corr * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        return numerator / denominator
