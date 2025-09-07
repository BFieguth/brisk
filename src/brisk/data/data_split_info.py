"""Store and analyze data splits created by DataManager.

This module defines the DataSplitInfo class, which is responsible for storing
and analyzing data related to the training and testing splits of datasets within
the Brisk framework. The DataSplitInfo class provides methods for calculating
descriptive statistics for both continuous and categorical features, as well as
visualizing the distributions of these features through various plots.

Examples
--------
Get scaled training data:
    >>> X_train_scaled, y_train = data_info.get_train()
"""

import os
import pathlib
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from brisk.evaluation.evaluators import registry_module
from brisk.evaluation.evaluators.builtin import register_dataset_evaluators
from brisk.services import get_services

class DataSplitInfo:
    """Store and analyze features and labels of training and testing splits.

    This class provides methods for calculating descriptive statistics for both
    continuous and categorical features, as well as visualizing the
    distributions of these features through various plots. It handles data
    scaling, feature categorization, and statistical analysis automatically.

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
    group_index_train : Dict[str, np.array] or None
        Index of the groups for the training split
    group_index_test : Dict[str, np.array] or None
        Index of the groups for the testing split
    split_key : Tuple[str, str, str]
        The split key (group_name, dataset_name, table_name)
    split_index : int
        The split index in DataSplits container
    scaler : object, optional
        The fitted scaler used for this split, by default None
    categorical_features : List[str], optional
        List of categorical feature names, by default None
    continuous_features : List[str], optional
        List of continuous feature names, by default None

    Attributes
    ----------
    group_name : str
        The name of the experiment group
    dataset_name : str
        The name of the dataset
    table_name : str
        The name of the table
    features : List[str]
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
    group_index_train : Dict[str, np.array] or None
        Index of the groups for the training split
    group_index_test : Dict[str, np.array] or None
        Index of the groups for the testing split
    registry : EvaluatorRegistry
        The evaluator registry with evaluators for datasets
    categorical_features : List[str]
        List of categorical features present in the training dataset
    continuous_features : List[str]
        List of continuous features derived from the training dataset
    scaler : object or None
        The scaler used for this split

    Notes
    -----
    The class automatically detects categorical features if not provided.
    Statistics are calculated for both continuous and categorical features
    during initialization. The class also handles data scaling when a scaler
    is provided, ensuring that only continuous features are scaled while
    preserving categorical features in their original form.

    Examples
    --------
    Create a basic data split info:
        >>> data_info = DataSplitInfo(
        ...     X_train, X_test, y_train, y_test,
        ...     group_index_train=None, group_index_test=None,
        ...     split_key=("group1", "dataset.csv", None),
        ...     split_index=0
        ... )

    Create with specific feature types:
        >>> data_info = DataSplitInfo(
        ...     X_train, X_test, y_train, y_test,
        ...     group_index_train=None, group_index_test=None,
        ...     split_key=("group1", "dataset.csv", None),
        ...     split_index=0,
        ...     categorical_features=["category1", "category2"],
        ...     continuous_features=["feature1", "feature2"]
        ... )
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
        """Initialize DataSplitInfo with training and testing data.

        Creates a new DataSplitInfo instance with the provided training and
        testing data, along with metadata about the split. Sets up the
        evaluator registry and performs automatic feature categorization.

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
        group_index_train : Dict[str, np.array] or None
            Index of the groups for the training split
        group_index_test : Dict[str, np.array] or None
            Index of the groups for the testing split
        split_key : Tuple[str, str, str]
            The split key (group_name, dataset_name, table_name)
        split_index : int
            The split index in DataSplits container
        scaler : object, optional
            The fitted scaler used for this split, by default None
        categorical_features : List[str], optional
            List of categorical feature names, by default None
        continuous_features : List[str], optional
            List of continuous feature names, by default None

        Notes
        -----
        The initialization process:
        1. Extracts group and dataset information from split_key
        2. Sets up the output directory for plots and statistics
        3. Creates copies of the input data to prevent modifications
        4. Initializes the evaluator registry with dataset evaluators
        5. Categorizes features as categorical or continuous
        6. Performs automatic data split evaluation
        """
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
        self.services.io.set_output_dir(pathlib.Path(
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

        plot_settings = self.services.utility.get_plot_settings()
        self.registry = registry_module.EvaluatorRegistry()
        register_dataset_evaluators(self.registry, plot_settings)
        for evaluator in self.registry.evaluators.values():
            evaluator.set_services(self.services)

        self.categorical_features = []
        self.continuous_features = []
        self._set_features(
            X_train.columns, categorical_features, continuous_features
        )
        self.scaler = scaler
        self.evaluate_data_split()

    def evaluate_data_split(self) -> None:
        """Evaluate distribution of features in the train and test splits.

        This method calculates descriptive statistics for both continuous and 
        categorical features in the training and testing splits. It also 
        generates plots including histograms, boxplots, pie plots, and 
        correlation matrices.

        The method uses the evaluator registry to get the appropriate evaluators 
        for the dataset and then calls the evaluate method for each evaluator.

        Notes
        -----
        The evaluation process includes:
        1. Setting up the reporting context
        2. Calculating statistics for continuous features
        3. Calculating statistics for categorical features
        4. Generating histogram and box plots for continuous features
        5. Generating bar plots for categorical features
        6. Creating correlation matrices for continuous features
        7. Clearing the reporting context

        All plots and statistics are saved to the configured output directory.
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
                evaluator = self.registry.get("brisk_histogram_plot")
                evaluator.plot(
                    self.X_train[feature], self.X_test[feature],
                    feature, f"hist_box_plot/{feature}_hist_box_plot",
                    self.dataset_name, self.group_name
                )
            for feature in self.categorical_features:
                evaluator = self.registry.get("brisk_bar_plot")
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
        Uses a combination of data type analysis and cardinality checks to
        identify categorical features.

        Returns
        -------
        List[str]
            Names of detected categorical features

        Notes
        -----
        Features are considered categorical if they are:
        - Object dtype (string data)
        - Category dtype (pandas categorical data)
        - Boolean dtype (True/False values)
        - Have less than 5% unique values (low cardinality)

        The detection is performed on the combined training and test data
        to ensure consistent feature categorization across splits.

        Examples
        --------
        Detect categorical features automatically:
            >>> data_info = DataSplitInfo(...)
            >>> cat_features = data_info._detect_categorical_features()
            >>> print(cat_features)  # ['category1', 'category2']
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
        """Return the training features and labels.

        Returns the training data with optional scaling applied to continuous
        features. Categorical features are preserved in their original form
        while continuous features are scaled using the fitted scaler.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            A tuple containing the training features and training labels.
            Features are scaled if a scaler is available and continuous
            features are present.

        Notes
        -----
        If a scaler is available and continuous features exist:
        1. Categorical features are kept in their original form
        2. Continuous features are scaled using the fitted scaler
        3. Features are concatenated and reordered to match original order
        4. The original column order is preserved

        If no scaler is available, the original data is returned unchanged.

        Examples
        --------
        Get scaled training data:
            >>> X_train, y_train = data_info.get_train()
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

            # Concatenate categorical and scaled features, keep original order
            X_train_scaled = pd.concat(  # pylint: disable=C0103
                [categorical_data, continuous_scaled], axis=1
            )
            # Reorder columns to match original order
            original_order = list(self.X_train.columns)
            X_train_scaled = X_train_scaled[original_order]  # pylint: disable=C0103
            return X_train_scaled, self.y_train

        return self.X_train, self.y_train

    def get_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return the testing features and labels.

        Returns the testing data with optional scaling applied to continuous
        features. Categorical features are preserved in their original form
        while continuous features are scaled using the fitted scaler.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            A tuple containing the testing features and testing labels.
            Features are scaled if a scaler is available and continuous
            features are present.

        Notes
        -----
        If a scaler is available and continuous features exist:
        1. Categorical features are kept in their original form
        2. Continuous features are scaled using the fitted scaler
        3. Features are concatenated and reordered to match original order
        4. The original column order is preserved

        If no scaler is available, the original data is returned unchanged.

        Examples
        --------
        Get scaled testing data:
            >>> X_test, y_test = data_info.get_test()
            >>> print(X_test.shape)  # (n_samples, n_features)
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

            # Concatenate categorical and scaled features, keep original order
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
        """Return both the training and testing splits.

        Convenience method that returns both training and testing data
        in a single call. Data is scaled if a scaler is available.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            A tuple containing the training features, testing features, 
            training labels, and testing labels.

        Notes
        -----
        This method is equivalent to calling get_train() and get_test()
        separately, but provides a more convenient interface when both
        splits are needed.

        Examples
        --------
        Get both training and testing data:
            >>> X_train, X_test, y_train, y_test = data_info.get_train_test()
        """
        X_train, y_train = self.get_train() # pylint: disable=C0103
        X_test, y_test = self.get_test() # pylint: disable=C0103
        return X_train, X_test, y_train, y_test

    def get_split_metadata(self) -> Dict[str, Any]:
        """Return the split metadata used in certain metric calculations.

        Provides metadata about the data split that can be used for
        metric calculations and reporting purposes.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the split metadata with keys:
            - num_features: Number of features in the dataset
            - num_samples: Total number of samples (train + test)

        Examples
        --------
        Get split metadata:
            >>> metadata = data_info.get_split_metadata()
            >>> print(f"Features: {metadata['num_features']}")
            >>> print(f"Samples: {metadata['num_samples']}")
        """
        return {
            "num_features": len(self.X_train.columns),
            "num_samples": len(self.X_train) + len(self.X_test)
        }

    def _set_features(
        self,
        columns: List[str],
        categorical_features: Optional[List[str]],
        continuous_features: Optional[List[str]]
    ) -> None:
        """Set categorical and continuous features based on provided lists or
        detection.

        Categorizes features as either categorical or continuous based on
        the provided lists or automatic detection. Ensures that features
        are properly categorized and available in the dataset.

        Parameters
        ----------
        columns : List[str]
            List of all column names in the dataset
        categorical_features : Optional[List[str]]
            List of categorical feature names, or None for auto-detection
        continuous_features : Optional[List[str]]
            List of continuous feature names, or None for auto-detection

        Notes
        -----
        The feature categorization process:
        1. If categorical_features is None or empty, auto-detect categorical
        features
        2. Filter categorical features to only include those present in the
        dataset
        3. If continuous_features is None or empty, set to empty list
        4. Filter continuous features to exclude categorical features
        5. Set the features list to the combined categorical and continuous
        features

        This ensures that all features are properly categorized and available
        for analysis and scaling operations.
        """
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
