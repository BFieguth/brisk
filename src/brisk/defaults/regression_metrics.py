"""regression_metrics.py

This module defines a collection of regression metrics wrapped in
MetricWrapper instances for use within the Brisk framework. These metrics
are sourced from the scikit-learn library and provide various ways to
evaluate the performance of regression models. Additionally, it includes
a custom implementation of Lin's Concordance Correlation Coefficient (CCC).
"""
from typing import Dict, Any

import numpy as np
import scipy
from sklearn.metrics import _regression

from brisk.evaluation import metric_wrapper

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


def adjusted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_metadata: Dict[str, Any]
) -> float:
    r2 = _regression.r2_score(y_true, y_pred)
    adjusted_r2 = (1 - (1 - r2) * (len(y_true) - 1) /
                   (len(y_true) - split_metadata["num_features"] - 1))
    return adjusted_r2


REGRESSION_METRICS = [
    metric_wrapper.MetricWrapper(
        name="explained_variance_score",
        func=_regression.explained_variance_score,
        display_name="Explained Variance Score"
    ),
    metric_wrapper.MetricWrapper(
        name="max_error",
        func=_regression.max_error,
        display_name="Max Error"
    ),
    metric_wrapper.MetricWrapper(
        name="mean_absolute_error",
        func=_regression.mean_absolute_error,
        display_name="Mean Absolute Error",
        abbr="MAE"
    ),
    metric_wrapper.MetricWrapper(
        name="mean_absolute_percentage_error",
        func=_regression.mean_absolute_percentage_error,
        display_name="Mean Absolute Percentage Error",
        abbr="MAPE"
    ),
    metric_wrapper.MetricWrapper(
        name="mean_pinball_loss",
        func=_regression.mean_pinball_loss,
        display_name="Mean Pinball Loss"
    ),
    metric_wrapper.MetricWrapper(
        name="mean_squared_error",
        func=_regression.mean_squared_error,
        display_name="Mean Squared Error",
        abbr="MSE"
    ),
    metric_wrapper.MetricWrapper(
        name="mean_squared_log_error",
        func=_regression.mean_squared_log_error,
        display_name="Mean Squared Log Error"
    ),
    metric_wrapper.MetricWrapper(
        name="median_absolute_error",
        func=_regression.median_absolute_error,
        display_name="Median Absolute Error"
    ),
    metric_wrapper.MetricWrapper(
        name="r2_score",
        func=_regression.r2_score,
        display_name="R2 Score",
        abbr="R2"
    ),
    metric_wrapper.MetricWrapper(
        name="root_mean_squared_error",
        func=_regression.root_mean_squared_error,
        display_name="Root Mean Squared Error",
        abbr="RMSE"
    ),
    metric_wrapper.MetricWrapper(
        name="root_mean_squared_log_error",
        func=_regression.mean_squared_log_error,
        display_name="Root Mean Squared Log Error"
    ),
    metric_wrapper.MetricWrapper(
        name="concordance_correlation_coefficient",
        func=concordance_correlation_coefficient,
        display_name="Concordance Correlation Coefficient",
        abbr="CCC"
    ),
    metric_wrapper.MetricWrapper(
        name="neg_mean_absolute_error",
        func=_regression.mean_absolute_error,
        display_name="Negative Mean Absolute Error",
        abbr="NegMAE",
        greater_is_better=False
    ),
    metric_wrapper.MetricWrapper(
        name="adjusted_r2_score",
        func=adjusted_r2_score,
        display_name="Adjusted R2 Score",
        abbr="AdjR2"
    )
]
