"""classification_metrics.py

This module defines a collection of classification metrics wrapped in
MetricWrapper instances for use within the Brisk framework. These metrics
are sourced from the scikit-learn library and provide various ways to
evaluate the performance of classification models.
"""

from sklearn import metrics

from brisk.evaluation import metric_wrapper

CLASSIFICATION_METRICS = [
    metric_wrapper.MetricWrapper(
        name="accuracy",
        func=metrics.accuracy_score,
        display_name="Accuracy",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="precision",
        func=metrics.precision_score,
        display_name="Precision",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="recall",
        func=metrics.recall_score,
        display_name="Recall",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="f1_score",
        func=metrics.f1_score,
        display_name="F1 Score",
        abbr="f1",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="balanced_accuracy",
        func=metrics.balanced_accuracy_score,
        display_name="Balanced Accuracy",
        abbr="bal_acc",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="top_k_accuracy",
        func=metrics.top_k_accuracy_score,
        display_name="Top-k Accuracy Score",
        abbr="top_k",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="log_loss",
        func=metrics.log_loss,
        display_name="Log Loss",
        greater_is_better=False
    ),
    metric_wrapper.MetricWrapper(
        name="roc_auc",
        func=metrics.roc_auc_score,
        display_name="Area Under the Receiver Operating Characteristic Curve",
        greater_is_better=True
    ),
    metric_wrapper.MetricWrapper(
        name="brier",
        func=metrics.brier_score_loss,
        display_name="Brier Score Loss",
        greater_is_better=False
    ),
    metric_wrapper.MetricWrapper(
        name="roc",
        func=metrics.roc_curve,
        display_name="Receiver Operating Characteristic",
        greater_is_better=True
    ),
]
