"""classification_metrics.py

This module defines a collection of classification metrics wrapped in 
MetricWrapper instances for use within the Brisk framework. These metrics 
are sourced from the scikit-learn library and provide various ways to 
evaluate the performance of classification models.
"""

from sklearn import metrics

from brisk.utility import MetricWrapper

CLASSIFICATION_METRICS = [
    MetricWrapper.MetricWrapper(
        name="accuracy",
        func=metrics.accuracy_score,
        display_name="Accuracy"
    ),
    MetricWrapper.MetricWrapper(
        name="precision",
        func=metrics.precision_score,
        display_name="Precision"
    ),
    MetricWrapper.MetricWrapper(
        name="recall",
        func=metrics.recall_score,
        display_name="Recall"
    ),
    MetricWrapper.MetricWrapper(
        name="f1_score",
        func=metrics.f1_score,
        display_name="F1 Score",
        abbr="f1"
    ),
    MetricWrapper.MetricWrapper(
        name="balanced_accuracy",
        func=metrics.balanced_accuracy_score,
        display_name="Balanced Accuracy",
        abbr="bal_acc"
    ),
    MetricWrapper.MetricWrapper(
        name="top_k_accuracy",
        func=metrics.top_k_accuracy_score,
        display_name="Top-k Accuracy Score",
        abbr="top_k"
    ),
    MetricWrapper.MetricWrapper(
        name="log_loss",
        func=metrics.log_loss,
        display_name="Log Loss"
    ),
    MetricWrapper.MetricWrapper(
        name="roc_auc",
        func=metrics.roc_auc_score,
        display_name="Area Under the Receiver Operating Characteristic Curve"
    ),
    MetricWrapper.MetricWrapper(
        name="brier",
        func=metrics.brier_score_loss,
        display_name="Brier Score Loss"
    ),
    MetricWrapper.MetricWrapper(
        name="roc",
        func=metrics.roc_curve,
        display_name="Receiver Operating Characteristic"
    ),
]
