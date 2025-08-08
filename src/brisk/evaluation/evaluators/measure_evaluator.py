"""
Base class for all evaluators that calculate measures of model performance.
"""
from abc import abstractmethod
from typing import List, Dict, Any, Tuple

from sklearn import base
import pandas as pd

from brisk.evaluation.evaluators.base import BaseEvaluator

class MeasureEvaluator(BaseEvaluator):
    """Template for evaluators that calcualte measures of model performance."""
    def evaluate(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str
    ) -> Dict[str, Any]:
        """Template for all measure methods to follow."""
        predictions = self._generate_prediction(model, X)
        results = self._calculate_measures(predictions, y, metrics)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_json(results, filename, metadata)
        self._log_results(results, filename)

    def _save_json(
        self,
        data: Dict[str, Any],
        filename: str,
        metadata: Dict[str, Any]
    ) -> str:
        """ENFORCED: Save JSON with metadata."""
        output_path = self.services.io.output_dir / f"{filename}.json"
        self.io.save_to_json(data, output_path, metadata)
        return str(output_path)

    def _generate_prediction(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame # pylint: disable=C0103
    ) -> pd.Series:
        """Default prediction generation - can be overridden."""
        return model.predict(X)

    @abstractmethod
    def _calculate_measures(
        self,
        predictions: pd.Series,
        y_true: pd.Series,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Must implement this method to calculate something."""
        pass

    # @abstractmethod
    def report(self, results: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
        """Default reporting - can be overridden.
        
        By default reporting assumes that the keys of dict are column headers
        and values are lists corresponding to each row. If the results are in a
        different format, this method should be overridden to return the list of
        column headers and a nested list of rows.
        """
        columns = [key for key in results.keys() if key != "_metadata"]
        rows = [row for key in columns for row in results[key]]
        return columns, rows
        # pass

    def _log_results(self, results: Dict[str, float], filename: str):
        """Default logging - can be overridden."""
        scores_log = "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])
        self.services.logger.logger.info(
            f"Results:\n{scores_log}\n Saved to '{filename}.json'."
        )
