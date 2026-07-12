"""Default scikit-learn algorithm and metric registrations.

Builds the built-in classification/regression algorithm and metric
collections on top of the sklearn adapter wrappers
(``model_adapter.SklearnAlgorithmWrapper`` and ``metric_adapter.SklearnMetricWrapper``). This is the
ports-and-adapters replacement for the registrations in ``brisk.defaults``.

Exposes
-------
CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS
    Lists of ``model_adapter.SklearnAlgorithmWrapper`` instances.
CLASSIFICATION_METRICS, REGRESSION_METRICS
    Lists of ``metric_adapter.SklearnMetricWrapper`` instances.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy
from sklearn import (ensemble, linear_model, metrics, naive_bayes, neighbors,
                     neural_network, svm, tree)
from sklearn.metrics import _regression

from brisk.adapters.sklearn import metric_adapter, model_adapter


def concordance_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Lin's Concordance Correlation Coefficient (CCC).

    Parameters
    ----------
    y_true : np.ndarray
        The true (observed) values.
    y_pred : np.ndarray
        The predicted values.

    Returns
    -------
    float
        The Concordance Correlation Coefficient between y_true and y_pred.
    """
    corr, _ = scipy.stats.pearsonr(y_true, y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * corr * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float(numerator / denominator)


def adjusted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_metadata: dict[str, Any],
) -> float:
    """Calculate the adjusted R^2 score.

    Parameters
    ----------
    y_true : np.ndarray
        The true (observed) values.
    y_pred : np.ndarray
        The predicted values.
    split_metadata : dict[str, Any]
        Metadata for the current split; must contain ``num_features``.

    Returns
    -------
    float
        The adjusted R^2 score.
    """
    r2 = _regression.r2_score(y_true, y_pred)
    adjusted_r2 = (
        1 - (1 - r2) * (len(y_true) - 1)
        / (len(y_true) - split_metadata["num_features"] - 1)
    )
    return float(adjusted_r2)


CLASSIFICATION_ALGORITHMS: list[model_adapter.SklearnAlgorithmWrapper] = [
    model_adapter.SklearnAlgorithmWrapper(
        name="logistic",
        display_name="Logistic Regression",
        algorithm_class=linear_model.LogisticRegression,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "penalty": [None, "l2", "l1", "elasticnet"],
            "l1_ratio": list(np.arange(0.1, 1, 0.1)),
            "C": list(np.arange(1, 30, 0.5)),
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="svc",
        display_name="Support Vector Classification",
        algorithm_class=svm.SVC,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "kernel": ["linear", "rbf", "sigmoid"],
            "C": list(np.arange(1, 30, 0.5)),
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="knn_classifier",
        display_name="k-Nearest Neighbours Classifier",
        algorithm_class=neighbors.KNeighborsClassifier,
        hyperparam_grid={
            "n_neighbors": list(range(1, 5, 2)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(5, 50, 5)),
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="dtc",
        display_name="Decision Tree Classifier",
        algorithm_class=tree.DecisionTreeClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": list(range(5, 25, 5)) + [None],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="rf_classifier",
        display_name="Random Forest Classifier",
        algorithm_class=ensemble.RandomForestClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "n_estimators": list(range(20, 160, 20)),
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": list(range(5, 25, 5)) + [None],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        algorithm_class=naive_bayes.GaussianNB,
        hyperparam_grid={
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="ridge_classifier",
        display_name="Ridge Classifier",
        algorithm_class=linear_model.RidgeClassifier,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)},
    ),
]

REGRESSION_ALGORITHMS: list[model_adapter.SklearnAlgorithmWrapper] = [
    model_adapter.SklearnAlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=linear_model.LinearRegression,
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=linear_model.Ridge,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)},
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="lasso",
        display_name="LASSO Regression",
        algorithm_class=linear_model.Lasso,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)},
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="bridge",
        display_name="Bayesian Ridge Regression",
        algorithm_class=linear_model.BayesianRidge,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "alpha_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "alpha_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "lambda_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "lambda_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="elasticnet",
        display_name="Elastic Net Regression",
        algorithm_class=linear_model.ElasticNet,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={
            "alpha": np.logspace(-3, 0, 100),
            "l1_ratio": list(np.arange(0.1, 1, 0.1)),
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="dtr",
        display_name="Decision Tree Regression",
        algorithm_class=tree.DecisionTreeRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "criterion": [
                "friedman_mse", "absolute_error", "poisson", "squared_error"
            ],
            "max_depth": list(range(5, 25, 5)) + [None],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="rf",
        display_name="Random Forest",
        algorithm_class=ensemble.RandomForestRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "n_estimators": list(range(20, 160, 20)),
            "criterion": [
                "friedman_mse", "absolute_error", "poisson", "squared_error"
            ],
            "max_depth": list(range(5, 25, 5)) + [None],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="svr",
        display_name="Support Vector Regression",
        algorithm_class=svm.SVR,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "kernel": ["linear", "rbf", "sigmoid"],
            "C": list(np.arange(1, 30, 0.5)),
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="mlp",
        display_name="Multi-Layer Perceptron Regression",
        algorithm_class=neural_network.MLPRegressor,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            "hidden_layer_sizes": [
                (100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "invscaling", "adaptive"],
        },
    ),
    model_adapter.SklearnAlgorithmWrapper(
        name="knn",
        display_name="K-Nearest Neighbour Regression",
        algorithm_class=neighbors.KNeighborsRegressor,
        hyperparam_grid={
            "n_neighbors": list(range(1, 5, 2)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(5, 50, 5)),
        },
    ),
]


CLASSIFICATION_METRICS: list[metric_adapter.SklearnMetricWrapper] = [
    metric_adapter.SklearnMetricWrapper(
        name="accuracy",
        func=metrics.accuracy_score,
        display_name="Accuracy",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="precision",
        func=metrics.precision_score,
        display_name="Precision",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="recall",
        func=metrics.recall_score,
        display_name="Recall",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="f1_score",
        func=metrics.f1_score,
        display_name="F1 Score",
        abbr="f1",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="balanced_accuracy",
        func=metrics.balanced_accuracy_score,
        display_name="Balanced Accuracy",
        abbr="bal_acc",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="top_k_accuracy",
        func=metrics.top_k_accuracy_score,
        display_name="Top-k Accuracy Score",
        abbr="top_k",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="log_loss",
        func=metrics.log_loss,
        display_name="Log Loss",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="roc_auc",
        func=metrics.roc_auc_score,
        display_name="Area Under the Receiver Operating Characteristic Curve",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="brier",
        func=metrics.brier_score_loss,
        display_name="Brier Score Loss",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="roc",
        func=metrics.roc_curve,
        display_name="Receiver Operating Characteristic",
        greater_is_better=True,
    ),
]

REGRESSION_METRICS: list[metric_adapter.SklearnMetricWrapper] = [
    metric_adapter.SklearnMetricWrapper(
        name="explained_variance_score",
        func=_regression.explained_variance_score,
        display_name="Explained Variance Score",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="max_error",
        func=_regression.max_error,
        display_name="Max Error",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="mean_absolute_error",
        func=_regression.mean_absolute_error,
        display_name="Mean Absolute Error",
        abbr="MAE",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="mean_absolute_percentage_error",
        func=_regression.mean_absolute_percentage_error,
        display_name="Mean Absolute Percentage Error",
        abbr="MAPE",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="mean_pinball_loss",
        func=_regression.mean_pinball_loss,
        display_name="Mean Pinball Loss",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="mean_squared_error",
        func=_regression.mean_squared_error,
        display_name="Mean Squared Error",
        abbr="MSE",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="mean_squared_log_error",
        func=_regression.mean_squared_log_error,
        display_name="Mean Squared Log Error",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="median_absolute_error",
        func=_regression.median_absolute_error,
        display_name="Median Absolute Error",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="r2_score",
        func=_regression.r2_score,
        display_name="R2 Score",
        abbr="R2",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="root_mean_squared_error",
        func=_regression.root_mean_squared_error,
        display_name="Root Mean Squared Error",
        abbr="RMSE",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="root_mean_squared_log_error",
        func=_regression.mean_squared_log_error,
        display_name="Root Mean Squared Log Error",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="concordance_correlation_coefficient",
        func=concordance_correlation_coefficient,
        display_name="Concordance Correlation Coefficient",
        abbr="CCC",
        greater_is_better=True,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="neg_mean_absolute_error",
        func=_regression.mean_absolute_error,
        display_name="Negative Mean Absolute Error",
        abbr="NegMAE",
        greater_is_better=False,
    ),
    metric_adapter.SklearnMetricWrapper(
        name="adjusted_r2_score",
        func=adjusted_r2_score,
        display_name="Adjusted R2 Score",
        abbr="AdjR2",
        greater_is_better=True,
    ),
]
