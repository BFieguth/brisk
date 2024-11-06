import sklearn.metrics as metrics

from brisk.utility.MetricWrapper import MetricWrapper

CLASSIFICATION_METRICS = [
    MetricWrapper(
        name="accuracy",
        func=metrics.accuracy_score,
        display_name="Accuracy"
    ),
    MetricWrapper(
        name="precision",
        func=metrics.precision_score,
        display_name="Precision"
    ),
    MetricWrapper(
        name="recall",
        func=metrics.recall_score,
        display_name="Recall"
    ),
    MetricWrapper(
        name="f1_score",
        func=metrics.f1_score,
        display_name="F1 Score",
        abbr="f1"
    ),
    MetricWrapper(
        name="balanced_accuracy",
        func=metrics.balanced_accuracy_score,
        display_name="Balanced Accuracy",
        abbr="bal_acc"
    ),
    MetricWrapper(
        name="top_k_accuracy",
        func=metrics.top_k_accuracy_score,
        display_name="Top-k Accuracy Score",
        abbr="top_k"
    ),
    MetricWrapper(
        name="log_loss",
        func=metrics.log_loss,
        display_name="Log Loss"
    ),
    MetricWrapper(
        name="roc_auc",
        func=metrics.roc_auc_score,
        display_name="Area Under the Receiver Operating Characteristic Curve"
    ),
    MetricWrapper(
        name="brier",
        func=metrics.brier_score_loss,
        display_name="Brier Score Loss"
    ),
    MetricWrapper(
        name="roc",
        func=metrics.roc_curve,
        display_name="Receiver Operating Characteristic"
    ),
]
