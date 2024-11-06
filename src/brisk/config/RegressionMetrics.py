import numpy as np 
import scipy
import sklearn.metrics._regression as regression

from brisk.utility.MetricWrapper import MetricWrapper

def concordance_correlation_coefficient(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """Calculate Lin's Concordance Correlation Coefficient (CCC).

    Args:
        y_true (np.ndarray): The true (observed) values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The Concordance Correlation Coefficient between y_true and y_pred.
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
    MetricWrapper(
        name="explained_variance_score",
        func=regression.explained_variance_score,
        display_name="Explained Variance Score"
    ),
    MetricWrapper(
        name="max_error",
        func=regression.max_error,
        display_name="Max Error"
    ),
    MetricWrapper(
        name="mean_absolute_error",
        func=regression.mean_absolute_error,
        display_name="Mean Absolute Error",
        abbr="MAE"
    ),
    MetricWrapper(
        name="mean_absolute_percentage_error",
        func=regression.mean_absolute_percentage_error,
        display_name="Mean Absolute Percentage Error",
        abbr="MAPE"
    ),
    MetricWrapper(
        name="mean_pinball_loss",
        func=regression.mean_pinball_loss,
        display_name="Mean Pinball Loss"
    ),
    MetricWrapper(
        name="mean_squared_error",
        func=regression.mean_squared_error,
        display_name="Mean Squared Error",
        abbr="MSE"
    ),
    MetricWrapper(
        name="mean_squared_log_error",
        func=regression.mean_squared_log_error,
        display_name="Mean Squared Log Error"
    ),
    MetricWrapper(
        name="median_absolute_error",
        func=regression.median_absolute_error,
        display_name="Median Absolute Error"
    ),
    MetricWrapper(
        name="r2_score",
        func=regression.r2_score,
        display_name="R2 Score",
        abbr="R2"
    ),
    MetricWrapper(
        name="root_mean_squared_error",
        func=regression.mean_squared_error,
        display_name="Root Mean Squared Error",
        abbr="RMSE"
    ),
    MetricWrapper(
        name="root_mean_squared_log_error",
        func=regression.mean_squared_log_error,
        display_name="Root Mean Squared Log Error"
    ),
    MetricWrapper(
        name="concordance_correlation_coefficient",
        func=concordance_correlation_coefficient,
        display_name="Concordance Correlation Coefficient",
        abbr="CCC"
    ),
    MetricWrapper(
        name="neg_mean_absolute_error",
        func=regression.mean_absolute_error,
        display_name="Negative Mean Absolute Error",
        abbr="NegMAE",
        greater_is_better=False
    ),
]
