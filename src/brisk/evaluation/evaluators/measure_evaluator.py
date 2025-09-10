"""Base class for all evaluators that calculate measures of model performance.

This module provides the MeasureEvaluator abstract base class for
evaluators that calculate performance measures and metrics for machine
learning models. It provides a template for implementing model evaluation
methods with standardized prediction generation, metric calculation, and
result handling.
"""
import abc
from typing import List, Dict, Any, Tuple

from sklearn import base
import pandas as pd

from brisk.evaluation.evaluators import base as base_eval

class MeasureEvaluator(base_eval.BaseEvaluator):
    """Template for evaluators that calculate measures of model performance.

    Abstract base class for evaluators that calculate performance measures
    and metrics for machine learning models. Provides a standardized workflow
    for model evaluation including prediction generation, metric calculation,
    metadata handling, and result saving.

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
    primary_color : str
        Primary color for plots and visualizations
    secondary_color : str
        Secondary color for plots and visualizations
    accent_color : str
        Accent color for plots and visualizations

    Notes
    -----
    This abstract base class provides a template for implementing model
    performance evaluation methods. Subclasses must implement the
    _calculate_measures method to define the specific metric calculation logic.

    The class handles the complete evaluation workflow:
    1. Generate predictions using the model
    2. Calculate measures using _calculate_measures method
    3. Generate metadata for the results
    4. Save results to JSON file with metadata
    5. Log the results

    Examples
    --------
    Create a custom measure evaluator:
        >>> class CustomMeasureEvaluator(MeasureEvaluator):
        ...     def __init__(self):
        ...         super().__init__("custom_measure", "a custom measure")
        ...     
        ...     def _calculate_measures(self, predictions, y_true, metrics):
        ...         # Custom measure calculation logic
        ...         return {"custom_metric": 0.85}
    """

    def evaluate(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str
    ) -> None:
        """Template for all measure methods to follow.
        
        Executes the complete evaluation workflow for model performance
        measures. This method orchestrates the evaluation process by generating
        predictions, calculating measures, and handling result processing.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained model to evaluate
        X : pd.DataFrame
            The input data for evaluation
        y : pd.Series
            The true target values
        metrics : List[str]
            The list of metric names to calculate
        filename : str
            The name of the file to save the results to (without extension)

        Returns
        -------
        None

        Notes
        -----
        This method provides the standard workflow for model evaluation:
        1. Generate predictions using the model
        2. Calculate measures using _calculate_measures
        3. Generate metadata for the results
        4. Save results to JSON file
        5. Log the results

        The method delegates the actual measure calculation to the
        _calculate_measures method, which must be implemented by subclasses.
        """
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
        """Save JSON with metadata.

        Saves the evaluation results to a JSON file with associated metadata.
        This method provides standardized result saving functionality.

        Parameters
        ----------
        data : Dict[str, Any]
            The data to save (typically the evaluation results)
        filename : str
            The name of the file to save the results to (without extension)
        metadata : Dict[str, Any]
            The metadata to include with the saved data

        Returns
        -------
        str
            The path to the saved file

        Notes
        -----
        The method saves the data to the output directory specified in
        the services configuration. The filename is automatically given
        a .json extension.
        """
        output_path = self.services.io.output_dir / f"{filename}.json"
        self.io.save_to_json(data, output_path, metadata)
        return str(output_path)

    def _generate_prediction(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame # pylint: disable=C0103
    ) -> pd.Series:
        """Default prediction generation - can be overridden.

        Generates predictions using the provided model. This method provides
        default prediction generation that can be overridden by subclasses
        for custom prediction logic.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained model to use for prediction
        X : pd.DataFrame
            The input data for prediction

        Returns
        -------
        pd.Series
            The model predictions

        Notes
        -----
        The default implementation uses the model's predict method.
        Subclasses can override this method to implement custom prediction
        logic, such as using predict_proba for probability predictions
        or applying additional post-processing.
        """
        return model.predict(X)

    @abc.abstractmethod
    def _calculate_measures(
        self,
        predictions: pd.Series,
        y_true: pd.Series,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Must implement this method to calculate something.
        
        Abstract method that must be implemented by subclasses to define
        the specific measure calculation logic. This is where the actual
        metric computation takes place.

        Parameters
        ----------
        predictions : pd.Series
            The model predictions
        y_true : pd.Series
            The true target values
        metrics : List[str]
            The list of metric names to calculate

        Returns
        -------
        Dict[str, float]
            Dictionary containing the calculated measures with metric names
            as keys and their values as values

        Notes
        -----
        This method must be implemented by all subclasses. It should
        contain the specific logic for calculating the requested metrics
        using the provided predictions and true values.
        """
        pass

    def report(
        self,
        results: Dict[str, Any]
    ) -> Tuple[List[str], List[List[Any]]]:
        """Default reporting - can be overridden.
        
        Converts evaluation results into a format suitable for reporting.
        By default, assumes that the keys of the results dictionary are
        column headers and values are lists corresponding to each row.

        Parameters
        ----------
        results : Dict[str, Any]
            The results of the evaluation

        Returns
        -------
        Tuple[List[str], List[List[Any]]]
            A tuple containing:
            - List of column headers
            - Nested list of rows (each row is a list of values)

        Notes
        -----
        The default implementation assumes a specific results format where
        keys are column headers and values are lists of row data. If the
        results are in a different format, this method should be overridden
        to return the appropriate column headers and row data.

        The metadata key is automatically excluded from the column headers.
        """
        columns = [key for key in results.keys() if key != "_metadata"]
        rows = [row for key in columns for row in results[key]]
        return columns, rows

    def _log_results(self, results: Dict[str, float], filename: str) -> None:
        """Default logging - can be overridden.

        Logs the evaluation results in a standardized format.
        This method provides default logging functionality that can be
        overridden by subclasses for custom logging behavior.

        Parameters
        ----------
        results : Dict[str, float]
            The results of the evaluation to log
        filename : str
            The name of the file where results were saved

        Returns
        -------
        None

        Notes
        -----
        The default implementation logs all key-value pairs from the results
        dictionary in a formatted manner. Subclasses can override this
        method to provide custom logging behavior.

        The logging format shows each metric name and its value with
        4 decimal places precision.
        """
        scores_log = "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])
        self.services.logger.logger.info(
            f"Results:\n{scores_log}\n Saved to '{filename}.json'."
        )
