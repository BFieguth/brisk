import numpy as np 
import scipy
import sklearn.metrics as metrics
import sklearn.metrics._regression as regression

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


REGRESSION_SCORING = {
    "explained_variance_score": {
        "func": regression.explained_variance_score,
        "scorer": metrics.make_scorer(
            regression.explained_variance_score
            )
    },
    "max_error": {
        "func": regression.max_error,
        "scorer": metrics.make_scorer(regression.max_error)
    },
    "mean_absolute_error": {
        "abbr": "MAE",
        "func": regression.mean_absolute_error,
        "scorer": metrics.make_scorer(regression.mean_absolute_error)
    },
    "mean_absolute_percentage_error": {
        "abbr": "MAPE",
        "func": regression.mean_absolute_percentage_error,
        "scorer": metrics.make_scorer(
            regression.mean_absolute_percentage_error
            )
    },
    "mean_pinball_loss": {
        "func": regression.mean_pinball_loss,
        "scorer": metrics.make_scorer(regression.mean_pinball_loss)
    },
    "mean_squared_error": {
        "abbr": "MSE",
        "func": regression.mean_squared_error,
        "scorer": metrics.make_scorer(regression.mean_squared_error)
    },
    "mean_squared_log_error": {
        "func": regression.mean_squared_log_error,
        "scorer": metrics.make_scorer(regression.mean_squared_log_error)
    },
    "median_absolute_error": {
        "func": regression.median_absolute_error,
        "scorer": metrics.make_scorer(regression.median_absolute_error)
    },
    "r2_score": {
        "abbr": "R2",
        "func": regression.r2_score,
        "scorer": metrics.make_scorer(regression.r2_score)
    },
    "root_mean_squared_error": {
        "abbr": "RMSE",
        "func": regression.root_mean_squared_error,
        "scorer": metrics.make_scorer(
            regression.root_mean_squared_error
            )
    },
    "root_mean_squared_log_error": {
        "func": regression.root_mean_squared_log_error,
        "scorer": metrics.make_scorer(
            regression.root_mean_squared_log_error
            )
    },
    "concordance_correlation_coefficient": {
        "abbr": "CCC",
        "func": concordance_correlation_coefficient,
        "scorer": metrics.make_scorer(
            concordance_correlation_coefficient
            )
    }
}