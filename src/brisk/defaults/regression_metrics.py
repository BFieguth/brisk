"""regression_metrics.py

This module defines a collection of regression metrics wrapped in 
MetricWrapper instances for use within the Brisk framework. These metrics 
are sourced from the scikit-learn library and provide various ways to 
evaluate the performance of regression models. Additionally, it includes 
a custom implementation of Lin's Concordance Correlation Coefficient (CCC).
"""

import numpy as np
import scipy
from sklearn.metrics import _regression

from brisk.utility import MetricWrapper

def concordance_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Calculate Lin's Concordance Correlation Coefficient (CCC).

    Args:
        y_true (np.ndarray): The true (observed) values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The Concordance Correlation Coefficient between y_true and y_pred
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


REGRESSION_METRICS = [
    MetricWrapper.MetricWrapper(
        name="explained_variance_score",
        func=_regression.explained_variance_score,
        display_name="Explained Variance Score"
    ),
    MetricWrapper.MetricWrapper(
        name="max_error",
        func=_regression.max_error,
        display_name="Max Error"
    ),
    MetricWrapper.MetricWrapper(
        name="mean_absolute_error",
        func=_regression.mean_absolute_error,
        display_name="Mean Absolute Error",
        abbr="MAE"
    ),
    MetricWrapper.MetricWrapper(
        name="mean_absolute_percentage_error",
        func=_regression.mean_absolute_percentage_error,
        display_name="Mean Absolute Percentage Error",
        abbr="MAPE"
    ),
    MetricWrapper.MetricWrapper(
        name="mean_pinball_loss",
        func=_regression.mean_pinball_loss,
        display_name="Mean Pinball Loss"
    ),
    MetricWrapper.MetricWrapper(
        name="mean_squared_error",
        func=_regression.mean_squared_error,
        display_name="Mean Squared Error",
        abbr="MSE"
    ),
    MetricWrapper.MetricWrapper(
        name="mean_squared_log_error",
        func=_regression.mean_squared_log_error,
        display_name="Mean Squared Log Error"
    ),
    MetricWrapper.MetricWrapper(
        name="median_absolute_error",
        func=_regression.median_absolute_error,
        display_name="Median Absolute Error"
    ),
    MetricWrapper.MetricWrapper(
        name="r2_score",
        func=_regression.r2_score,
        display_name="R2 Score",
        abbr="R2"
    ),
    MetricWrapper.MetricWrapper(
        name="root_mean_squared_error",
        func=_regression.mean_squared_error,
        display_name="Root Mean Squared Error",
        abbr="RMSE"
    ),
    MetricWrapper.MetricWrapper(
        name="root_mean_squared_log_error",
        func=_regression.mean_squared_log_error,
        display_name="Root Mean Squared Log Error"
    ),
    MetricWrapper.MetricWrapper(
        name="concordance_correlation_coefficient",
        func=concordance_correlation_coefficient,
        display_name="Concordance Correlation Coefficient",
        abbr="CCC"
    ),
    MetricWrapper.MetricWrapper(
        name="neg_mean_absolute_error",
        func=_regression.mean_absolute_error,
        display_name="Negative Mean Absolute Error",
        abbr="NegMAE",
        greater_is_better=False
    ),
]
