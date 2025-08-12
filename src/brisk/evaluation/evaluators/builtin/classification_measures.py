"""Evaluators to calculate measures for classification problems."""
from typing import Any, Dict

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics

from brisk.evaluation.evaluators import measure_evaluator

class ConfusionMatrix(measure_evaluator.MeasureEvaluator):
    """Calculate a confusion matrix for a classification model."""
    def evaluate(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Generate and save a confusion matrix.

        Parameters
        ----------
        model : Any
            Trained classification model with predict method
        X : ndarray
            The input features.
        y : ndarray
            The true target values.
        filename : str
            The name of the output file (without extension).
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

        Parameters
        ----------
        prediction (pd.Series): 
            The predicted target values.
        y (np.ndarray): 
            The true target values.

        Returns
        -------
        Dict[str, Any]: 
            A dictionary containing the confusion matrix and labels.
        """
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, prediction, labels=labels).tolist()
        data = {
            "confusion_matrix": cm,
            "labels": labels
            }
        return data

    def _log_results(self, data: Dict[str, Any]):
        """Log the results of the confusion matrix to console.

        Parameters
        ----------
        data : Dict[str, Any]
            The data to log

        Returns
        -------
        None
        """
        header = " " * 10 + " ".join(
            f"{label:>10}" for label in data["labels"]
        ) + "\n"
        rows = [f"{label:>10} " + " ".join(f"{count:>10}" for count in row)
                for label, row in zip(data["labels"], data["confusion_matrix"])]
        table = header + "\n".join(rows)
        confusion_log = f"Confusion Matrix:\n{table}"
        self.services.logger.logger.info(confusion_log)
