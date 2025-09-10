"""Base class for all evaluators that calculate measures on a dataset.

This module provides the DatasetMeasureEvaluator abstract base class for
evaluators that calculate measures or statistics on datasets. It provides
a template for implementing dataset-level evaluation methods with standardized
data handling, metadata generation, and result saving.
"""
import abc
from typing import List, Dict, Any

import pandas as pd

from brisk.evaluation.evaluators import base

class DatasetMeasureEvaluator(base.BaseEvaluator):
    """Template for dataset evaluators that calculate measures or plot data.

    Abstract base class for evaluators that calculate measures or statistics
    on datasets. Provides a standardized workflow for dataset evaluation
    including data processing, metadata generation, result saving, and logging.

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
    This abstract base class provides a template for implementing dataset-level
    evaluation methods. Subclasses must implement the _calculate_measures method
    to define the specific evaluation logic.

    The class handles the complete evaluation workflow:
    1. Calculate measures using the implemented _calculate_measures method
    2. Generate metadata for the results
    3. Save results to JSON file with metadata
    4. Log the results

    Examples
    --------
    Create a custom dataset measure evaluator:
        >>> class CustomDatasetEvaluator(DatasetMeasureEvaluator):
        ...     def __init__(self):
        ...         super().__init__("custom_dataset", "Custom evaluation")
        ...     
        ...     def _calculate_measures(self, train_data, test_data, features):
        ...         # Custom measure calculation logic
        ...         return {"custom_metric": 0.85}
    """

    def evaluate(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str],
        filename: str,
        dataset_name: str,
        group_name: str,
    ) -> Dict[str, Any]:
        """Template for all measure methods to follow.

        Executes the complete evaluation workflow for dataset measures.
        This method orchestrates the evaluation process by calling the
        abstract _calculate_measures method and handling result processing.

        Parameters
        ----------
        train_data : pd.DataFrame or pd.Series
            The training data for evaluation
        test_data : pd.DataFrame or pd.Series
            The testing data for evaluation
        feature_names : List[str]
            The names of the features in the dataset
        filename : str
            The name of the file to save the results to (without extension)
        dataset_name : str
            The name of the dataset being evaluated
        group_name : str
            The name of the experiment group

        Returns
        -------
        Dict[str, Any]
            The results of the evaluation containing calculated measures

        Notes
        -----
        This method provides the standard workflow for dataset evaluation:
        1. Calculate measures using _calculate_measures
        2. Generate metadata for the results
        3. Save results to JSON file
        4. Log the results
        5. Return the calculated measures

        The method delegates the actual measure calculation to the
        _calculate_measures method, which must be implemented by subclasses.
        """
        results = self._calculate_measures(train_data, test_data, feature_names)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_json(results, filename, metadata)
        self._log_results(results, filename)
        return results

    @abc.abstractmethod
    def _calculate_measures(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Must implement this method to calculate something.

        Abstract method that must be implemented by subclasses to define
        the specific measure calculation logic. This is where the actual
        evaluation computation takes place.

        Parameters
        ----------
        train_data : pd.DataFrame or pd.Series
            The training data for evaluation
        test_data : pd.DataFrame or pd.Series
            The testing data for evaluation
        feature_names : List[str]
            The names of the features in the dataset

        Returns
        -------
        Dict[str, float]
            Dictionary containing the calculated measures with their names
            as keys and numeric values as values

        Notes
        -----
        This method must be implemented by all subclasses. It should
        contain the specific logic for calculating the desired measures
        or statistics on the provided dataset.
        """
        pass

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

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for output.

        Generates metadata for the evaluation results using the metadata
        service. This provides standardized metadata for all dataset
        measure evaluations.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset being evaluated
        group_name : str
            The name of the experiment group

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata about the evaluation

        Notes
        -----
        The metadata includes information about the evaluator method,
        dataset name, and experiment group, providing context for
        the evaluation results.
        """
        return self.metadata.get_dataset(
            self.method_name, dataset_name, group_name
        )

    def _log_results(self, results: Dict[str, float], filename: str) -> None:
        """Default logging - can be overridden.

        Logs the evaluation results in a standardized format. This method
        provides default logging functionality that can be overridden by
        subclasses for custom logging behavior.

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
        The default implementation logs all float values from the results
        dictionary in a formatted manner. Subclasses can override this
        method to provide custom logging behavior.

        The logging format shows each measure name and its value with
        4 decimal places precision.
        """
        scores_log = "\n".join([
            f"{k}: {v:.4f}"
            for result in results.values()
            for k, v in result.items()
            if isinstance(v, float)
        ])
        self.services.logger.logger.info(
            f"Results:\n{scores_log}\n Saved to '{filename}.json'."
        )
