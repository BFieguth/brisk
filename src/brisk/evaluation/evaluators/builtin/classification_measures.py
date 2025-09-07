"""Evaluators to calculate measures for classification problems.

This module provides built-in evaluators specifically designed for
classification problems. These evaluators calculate performance measures
and metrics that are relevant for classification tasks.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics

from brisk.evaluation.evaluators import measure_evaluator

class ConfusionMatrix(measure_evaluator.MeasureEvaluator):
    """Calculate a confusion matrix for a classification model.

    This evaluator generates confusion matrices for classification models,
    providing detailed information about the model's performance across
    different classes. The confusion matrix shows the count of true
    positives, false positives, true negatives, and false negatives
    for each class.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    services : ServiceBundle or None
        The global services bundle
    metric_config : MetricManager or None
        The metric configuration manager

    Notes
    -----
    The confusion matrix is a fundamental tool for evaluating classification
    model performance. It provides a detailed breakdown of prediction
    accuracy across all classes, making it easy to identify which classes
    the model predicts well and which it struggles with.

    The evaluator automatically determines the unique labels from the
    true target values and creates a square matrix where rows represent
    true classes and columns represent predicted classes.

    Examples
    --------
    Use the confusion matrix evaluator:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_confusion_matrix")
        >>> evaluator.evaluate(model, X, y, "confusion_matrix_results")
    """

    def evaluate(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Generate and save a confusion matrix.

        Executes the complete evaluation workflow for generating a
        confusion matrix. This includes generating predictions, calculating
        the confusion matrix, and saving the results with metadata.

        Parameters
        ----------
        model : Any
            Trained classification model with predict method
        X : np.ndarray
            The input features for evaluation
        y : np.ndarray
            The true target values
        filename : str
            The name of the output file (without extension)

        Returns
        -------
        None

        Notes
        -----
        This method overrides the base evaluate method to provide
        classification-specific evaluation workflow. It generates
        predictions using the model and calculates the confusion matrix
        with appropriate class labels.

        The results are saved as JSON with metadata for later analysis
        and reporting.
        """
        prediction = self._generate_prediction(model, X)
        results = self._calculate_measures(prediction, y)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_json(results, filename, metadata)
        self._log_results(results)

    def _calculate_measures(
        self,
        prediction: pd.Series,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate a confusion matrix.

        Calculates the confusion matrix for the given predictions and
        true labels. The matrix shows the count of correct and incorrect
        predictions for each class.

        Parameters
        ----------
        prediction : pd.Series
            The predicted target values from the model
        y : np.ndarray
            The true target values

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - confusion_matrix: 2D list representing the confusion matrix
            - labels: List of unique class labels in order

        Notes
        -----
        The confusion matrix is calculated using scikit-learn's
        confusion_matrix function with labels determined from the
        unique values in the true target array. The matrix is converted
        to a list format for JSON serialization.

        The labels are sorted to ensure consistent ordering across
        different evaluations.
        """
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, prediction, labels=labels).tolist()
        data = {
            "confusion_matrix": cm,
            "labels": labels
            }
        return data

    def _log_results(self, data: Dict[str, Any]) -> None:
        """Log the results of the confusion matrix to console.

        Displays the confusion matrix in a formatted table format
        for easy reading and analysis. The table shows class labels
        as both row and column headers with prediction counts in
        the corresponding cells.

        Parameters
        ----------
        data : Dict[str, Any]
            The confusion matrix data containing:
            - confusion_matrix: 2D list of prediction counts
            - labels: List of class labels

        Returns
        -------
        None

        Notes
        -----
        The logged table format makes it easy to quickly assess
        model performance across different classes. The format
        shows true classes as rows and predicted classes as columns,
        making it straightforward to identify misclassification patterns.

        The table is formatted with proper spacing and alignment
        for readability in console output.
        """
        header = " " * 10 + " ".join(
            f"{label:>10}" for label in data["labels"]
        ) + "\n"
        rows = [f"{label:>10} " + " ".join(f"{count:>10}" for count in row)
                for label, row in zip(data["labels"], data["confusion_matrix"])]
        table = header + "\n".join(rows)
        confusion_log = f"Confusion Matrix:\n{table}"
        self.services.logger.logger.info(confusion_log)
