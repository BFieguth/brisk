import sklearn.metrics as metrics

CLASSIFICATION_METRICS = {
    "accuracy": {
        "display_name": "Accuracy",
        "func": metrics.accuracy_score,
        "scorer": metrics.make_scorer(
            metrics.accuracy_score
        ),
    },
    "precision": {
        "display_name": "Precision",
        "func": metrics.precision_score,
        "scorer": metrics.make_scorer(
            metrics.precision_score
        ),
    },
    "recall": {
        "display_name": "Recall",
        "func": metrics.recall_score,
        "scorer": metrics.make_scorer(
            metrics.recall_score
        ),
    },
    "f1_score": {
        "display_name": "F1 Score",
        "abbr": "f1",
        "func": metrics.f1_score,
        "scorer": metrics.make_scorer(
            metrics.f1_score
        ),
    },
    "balanced_accuracy": {
        "display_name": "Balanced Accuracy",
        "abbr": "bal_acc",
        "func": metrics.balanced_accuracy_score,
        "scorer": metrics.make_scorer(
            metrics.balanced_accuracy_score
        ),
    },
    "top_k_accuracy": {
        "display_name": "Top-k Accuracy Score",
        "abbr": "top_k",
        "func": metrics.top_k_accuracy_score,
        "scorer": metrics.make_scorer(
            metrics.top_k_accuracy_score
        ),
    },
    "log_loss": {
        "display_name": "Log Loss",
        "func": metrics.log_loss,
        "scorer": metrics.make_scorer(
            metrics.log_loss
        ),
    },
    "roc_auc": {
        "display_name": "Area Under the Reciever Operating Characteristic Curve",
        "func": metrics.roc_auc_score,
        "scorer": metrics.make_scorer(
            metrics.roc_auc_score
        ),
    },
    "brier": {
        "display_name": "Brier Score Loss",
        "func": metrics.brier_score_loss,
        "scorer": metrics.make_scorer(
            metrics.brier_score_loss
        ),
    },
    "roc": {
        "display_name": "Receiver Operating Characteristic",
        "func": metrics.roc_curve,
        "scorer": metrics.make_scorer(
            metrics.roc_curve
        ),
    },
}