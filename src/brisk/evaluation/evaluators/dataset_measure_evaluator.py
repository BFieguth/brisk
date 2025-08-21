"""Base class for all evaluators that calculate measures on a dataset."""
from abc import abstractmethod
from typing import List, Dict, Any

import pandas as pd

from brisk.evaluation.evaluators.base import BaseEvaluator

class DatasetMeasureEvaluator(BaseEvaluator):
    """Template for dataset evaluators that calculate measures or plot data."""
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

        Parameters
        ----------
        train_data : pd.DataFrame | pd.Series
            The training data
        test_data : pd.DataFrame | pd.Series
            The testing data
        feature_names : List[str]
            The names of the features
        filename : str
            The name of the file to save the results to
        dataset_name : str
            The name of the dataset
        group_name : str
            The name of the experiment group

        Returns
        -------
        Dict[str, Any]
            The results of the evaluation
        """
        results = self._calculate_measures(train_data, test_data, feature_names)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_json(results, filename, metadata)
        self._log_results(results, filename)
        return results

    @abstractmethod
    def _calculate_measures(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Must implement this method to calculate something.

        Parameters
        ----------
        train_data : pd.DataFrame | pd.Series
            The training data
        test_data : pd.DataFrame | pd.Series
            The testing data
        feature_names : List[str]
            The names of the features

        Returns
        -------
        Dict[str, float]
            The results of the evaluation
        """
        pass

    def _save_json(
        self,
        data: Dict[str, Any],
        filename: str,
        metadata: Dict[str, Any]
    ) -> str:
        """ENFORCED: Save JSON with metadata.

        Parameters
        ----------
        data : Dict[str, Any]
            The data to save
        filename : str
            The name of the file to save the results to
        metadata : Dict[str, Any]
            The metadata to save

        Returns
        -------
        str
            The path to the saved file
        """
        output_path = self.services.io.output_dir / f"{filename}.json"
        self.io.save_to_json(data, output_path, metadata)
        return str(output_path)

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str
    ) -> Dict[str, Any]:
        """Enforced: generate metadata for output.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset
        group_name : str
            The name of the experiment group

        Returns
        -------
        Dict[str, Any]
            The metadata for the output file
        """
        return self.metadata.get_dataset(
            self.method_name, dataset_name, group_name
        )

    def _log_results(self, results: Dict[str, float], filename: str):
        """Default logging - can be overridden.

        Parameters
        ----------
        results : Dict[str, float]
            The results of the evaluation
        filename : str
            The name of the file to save the results to

        Returns
        -------
        None
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
