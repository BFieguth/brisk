"""Store and analyze data splits created by DataManager.

This module defines the DataSplitInfo class, which is responsible for storing
and analyzing data related to the training and testing splits of datasets within
the Brisk framework. The DataSplitInfo class provides methods for calculating
descriptive statistics for both continuous and categorical features, as well as
visualizing the distributions of these features through various plots.

Examples
--------
>>> from brisk.data.data_split_info import DataSplitInfo
>>> data_info = DataSplitInfo(X_train, X_test, y_train, y_test,
...                          filename="dataset.csv", scaler=my_scaler)
>>> data_info.save_distribution("output_directory")
"""

import os
from typing import Any, List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from brisk.evaluation.evaluators.registry import EvaluatorRegistry
from brisk.evaluation.evaluators.builtin import register_dataset_evaluators
from brisk.services import get_services

class DataSplitInfo:
    """Store and analyze features and labels of training and testing splits.

    This class provides methods for calculating descriptive statistics for both
    continuous and categorical features, as well as visualizing the
    distributions of these features through various plots.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training features
    X_test : pd.DataFrame
        The testing features
    y_train : pd.Series
        The training labels
    y_test : pd.Series
        The testing labels
    group_index_train : dict
        Index of the groups for the training split
    group_index_test : dict
        Index of the groups for the testing split
    split_key : tuple
        The split key (group_name, dataset_name, table_name)
    split_index : int
        The split index in DataSplits container
    scaler : object, optional
        The scaler used for this split
    continuous_features : list of str, optional
        List of continuous feature names
    categorical_features : list of str, optional
        List of categorical feature names
    scaler : object, optional
        The fitted scaler used for this split

    Attributes
    ----------
    group_name : str
        The name of the experiment group
    dataset_name : str
        The name of the dataset
    table_name : str
        The name of the table
    features : list of str or None
        The order of input features
    split_index : int
        The split index in DataSplits container
    services : ServiceBundle
        The global services bundle
    X_train : pd.DataFrame
        The training features
    X_test : pd.DataFrame
        The testing features
    y_train : pd.Series
        The training labels
    y_test : pd.Series
        The testing labels
    group_index_train : dict
        Index of the groups for the training split
    group_index_test : dict
        Index of the groups for the testing split
    registry : EvaluatorRegistry
        The evaluator registry with evaluators for datasets
    categorical_features : list of str
        List of categorical features present in the training dataset
    continuous_features : list of str
        List of continuous features derived from the training dataset
    scaler : object or None
        The scaler used for this split

    Notes
    -----
    The class automatically detects categorical features if not provided.
    Statistics are calculated for both continuous and categorical features
    during initialization.
    """
    def __init__(
        self,
        X_train: pd.DataFrame, # pylint: disable=C0103
        X_test: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        y_test: pd.Series,
        group_index_train: Dict[str, np.array] | None,
        group_index_test: Dict[str, np.array] | None,
        split_key: Tuple[str, str, str],
        split_index: int,
        scaler: Optional[Any] = None,
        categorical_features: Optional[List[str]] = None,
        continuous_features: Optional[List[str]] = None
    ):
        self.group_name = split_key[0]
        self.file_name = split_key[1]
        self.table_name = split_key[2]
        if self.table_name is not None:
            self.dataset_name = f"{self.file_name}_{self.table_name}"
        else:
            self.dataset_name = self.file_name
        self.features = []
        self.split_index = split_index

        self.services = get_services()
        self.services.io.set_output_dir(Path(
            os.path.join(
                self.services.io.results_dir,
                self.group_name,
                self.dataset_name,
                f"split_{split_index}",
                "split_distribution"
            )
        ))

        self.X_train = X_train.copy(deep=True) # pylint: disable=C0103
        self.X_test = X_test.copy(deep=True) # pylint: disable=C0103
        self.y_train = y_train.copy(deep=True)
        self.y_test = y_test.copy(deep=True)
        self.group_index_train = group_index_train
        self.group_index_test = group_index_test

        theme = self.services.utility.get_theme()
        self.registry = EvaluatorRegistry()
        register_dataset_evaluators(self.registry, theme)
        for evaluator in self.registry.evaluators.values():
            evaluator.set_services(self.services)

        self.categorical_features = []
        self.continuous_features = []
        self._set_features(
            X_train.columns, categorical_features, continuous_features
        )
        self.scaler = scaler
        self.evaluate_data_split()

    def evaluate_data_split(self):
        """Evaluate distribution of features in the train and test splits.

        This method calculates descriptive statistics for both continuous and 
        categorical features in the training and testing splits. It also 
        generates plots including histograms, boxplots, pie plots, and 
        correlation matrices

        The method uses the evaluator registry to get the appropriate evaluators 
        for the dataset and then calls the evaluate method for each evaluator.

        Returns
        -------
        None
        """
        self.services.reporting.set_context(
            self.group_name, self.dataset_name, self.split_index, self.features,
            None
        )
        try:
            self.services.logger.logger.info(
                "Calculating stats for continuous features in %s split.", 
                self.dataset_name
            )
            evaluator = self.registry.get("brisk_continuous_statistics")
            evaluator.evaluate(
                self.X_train, self.X_test, self.continuous_features,
                "continuous_stats", self.group_name, self.dataset_name
            )

            self.services.logger.logger.info(
                "Calculating stats for categorical features in %s split.", 
                self.dataset_name
                )
            evaluator = self.registry.get("brisk_categorical_statistics")
            evaluator.evaluate(
                self.X_train, self.X_test, self.categorical_features,
                "categorical_stats", self.group_name, self.dataset_name
            )
            for feature in self.continuous_features:
                evaluator = self.registry.get("brisk_histogram_boxplot")
                evaluator.plot(
                    self.X_train[feature], self.X_test[feature],
                    feature, f"hist_box_plot/{feature}_hist_box_plot",
                    self.dataset_name, self.group_name
                )
            for feature in self.categorical_features:
                evaluator = self.registry.get("brisk_pie_plot")
                evaluator.plot(
                    self.X_train[feature], self.X_test[feature],
                    feature, f"pie_plot/{feature}_pie_plot",
                    self.dataset_name, self.group_name
                )
            evaluator = self.registry.get("brisk_correlation_matrix")
            evaluator.plot(
                self.X_train, self.continuous_features,
                "correlation_matrix",
                self.dataset_name, self.group_name
            )
        finally:
            self.services.reporting.clear_context()

    def _detect_categorical_features(self) -> List[str]:
        """Detect possible categorical features in the dataset.

        Checks datatype and if less than 5% of the columns have unique values.

        Returns
        -------
        List[str]
            Names of detected categorical features

        Notes
        -----
        Features are considered categorical if they are:
        - Object dtype
        - Category dtype
        - Boolean dtype
        - Have less than 5% unique values
        """
        combined_data = pd.concat([self.X_train, self.X_test], axis=0)
        categorical_features = []

        for column in combined_data.columns:
            series = combined_data[column]
            n_unique = series.nunique()
            n_samples = len(series)

            is_categorical = any([
                series.dtype == "object",
                series.dtype == "category",
                series.dtype == "bool",
                (n_unique / n_samples < 0.05)
            ])

            if is_categorical:
                categorical_features.append(column)

        self.services.logger.logger.info(
            "Detected %d categorical features: %s",
            len(categorical_features),
            categorical_features
        )
        return categorical_features

    def get_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the training features.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            A tuple containing the training features and training labels.
        """
        if self.scaler and self.continuous_features:
            categorical_data = (
                self.X_train[self.categorical_features].copy()
                if self.categorical_features
                else pd.DataFrame(index=self.X_train.index)
            )

            continuous_scaled = pd.DataFrame(
                self.scaler.transform(self.X_train[self.continuous_features]),
                columns=self.continuous_features,
                index=self.X_train.index
            )

            # Concatenate categorical and scaled features, preserving original order
            X_train_scaled = pd.concat(  # pylint: disable=C0103
                [categorical_data, continuous_scaled], axis=1
            )
            # Reorder columns to match original order
            original_order = list(self.X_train.columns)
            X_train_scaled = X_train_scaled[original_order]  # pylint: disable=C0103
            return X_train_scaled, self.y_train

        return self.X_train, self.y_train

    def get_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the testing features.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            A tuple containing the testing features and testing labels.
        """
        if self.scaler and self.continuous_features:
            categorical_data = (
                self.X_test[self.categorical_features].copy()
                if self.categorical_features
                else pd.DataFrame(index=self.X_test.index)
            )

            continuous_scaled = pd.DataFrame(
                self.scaler.transform(self.X_test[self.continuous_features]),
                columns=self.continuous_features,
                index=self.X_test.index
            )

            # Concatenate categorical and scaled features, preserving original order
            X_test_scaled = pd.concat(  # pylint: disable=C0103
                [categorical_data, continuous_scaled], axis=1
            )
            # Reorder columns to match original order
            original_order = list(self.X_test.columns)
            X_test_scaled = X_test_scaled[original_order]  # pylint: disable=C0103
            return X_test_scaled, self.y_test

        return self.X_test, self.y_test

    def get_train_test(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Returns both the training and testing split.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            A tuple containing the training features, testing features, 
            training labels, and testing labels.
        """
        X_train, y_train = self.get_train() # pylint: disable=C0103
        X_test, y_test = self.get_test() # pylint: disable=C0103
        return X_train, X_test, y_train, y_test

    def get_split_metadata(self) -> Dict[str, Any]:
        """Returns the split metadata used in certain metric calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the split metadata.
        """
        return {
            "num_features": len(self.X_train.columns),
            "num_samples": len(self.X_train) + len(self.X_test)
        }

    def _set_features(self, columns, categorical_features, continuous_features):
        if categorical_features is None or len(categorical_features) == 0:
            categorical_features = self._detect_categorical_features()

        self.categorical_features = [
            feature for feature in categorical_features
            if feature in columns
        ]

        if continuous_features is None or len(continuous_features) == 0:
            self.continuous_features = []
        else:
            self.continuous_features = [
                feature for feature in continuous_features
                if (
                    feature in columns and
                    feature not in self.categorical_features
                )
            ]
        
        self.features = self.continuous_features + self.categorical_features
