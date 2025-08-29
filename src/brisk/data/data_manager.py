"""Provides the DataManager class for creating train-test splits and applying preprocessing.

This module contains the DataManager class, which handles creating train-test
splits for machine learning models and applies preprocessing pipelines. It supports
several splitting strategies such as shuffle, k-fold, and stratified splits, with
optional grouping, and can apply missing data handling, scaling, categorical encoding,
and feature selection preprocessing.

Exports:
    DataManager: A class for configuring and generating train-test splits with
    preprocessing pipelines.
"""

import os
import sqlite3
from typing import Optional, List, Any

import pandas as pd
from sklearn import model_selection

from brisk.data import data_split_info
from brisk.data.preprocessing import (
    BasePreprocessor,
    MissingDataPreprocessor,
    ScalingPreprocessor,
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor,
)

from brisk.data import data_splits
from brisk.services import get_services

class DataManager:
    """A class that handles data splitting logic for creating train-test splits.

    This class allows users to configure different splitting strategies
    (e.g., shuffle, k-fold, stratified) and return train-test splits or
    cross-validation folds. It supports splitting based on groupings and
    includes a completed data preprocessing pipeline


    Parameters
    ----------
    test_size : float, optional
        The proportion of the dataset to allocate to the test set, by default
        0.2
    n_splits : int, optional
        Number of splits for cross-validation, by default 5
    split_method : str, optional
        The method to use for splitting ("shuffle" or "kfold"), by default
        "shuffle"
    group_column : str, optional
        The column to use for grouping (if any), by default None
    stratified : bool, optional
        Whether to use stratified sampling or cross-validation, by default False
    random_state : int, optional
        The random seed for reproducibility, by default None
    problem_type : str, optional
        The type of problem ("classification" or "regression").
        Defaults to "classification".
    algorithm_config : list or AlgorithmCollection of AlgorithmWrapper, optional
        User-provided collection of AlgorithmWrapper objects to use for feature selection.
    preprocessors : List[BasePreprocessor], optional
        List of preprocessor objects to apply to the data in sequence.


    Attributes
    ----------
    test_size : float
        Proportion of dataset allocated to test set
    n_splits : int
        Number of splits for cross-validation
    split_method : str
        Method used for splitting
    group_column : str or None
        Column used for grouping
    stratified : bool
        Whether stratified sampling is used
    random_state : int or None
        Random seed for reproducibility
    problem_type : str
        Type of problem (classification or regression)
    algorithm_config : list of AlgorithmWrapper or None
        List of algorithms to use as feature selection estimators
    preprocessors : List[BasePreprocessor]
        List of preprocessors to apply to the data
    splitter : sklearn.model_selection._BaseKFold
        The initialized scikit-learn splitter object
    _splits : dict
        Cache of previously computed splits
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_splits: int = 5,
        split_method: str = "shuffle",
        group_column: Optional[str] = None,
        stratified: bool = False,
        random_state: Optional[int] = None,
        problem_type: str = "classification",
        algorithm_config=None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
    ):
        self.services = get_services()
        self.test_size = test_size
        self.split_method = split_method
        self.group_column = group_column
        self.stratified = stratified
        self.n_splits = n_splits
        self.random_state = random_state
        self.problem_type = problem_type
        self.algorithm_config = algorithm_config
        self.preprocessors = preprocessors or []
        self._validate_config()
        self.splitter = self._set_splitter()
        self._splits = {}

    def _validate_config(self) -> None:
        """Validates the provided configuration for splitting.

        Raises
        ------
            ValueError
                If invalid split method or incompatible combination of group
                column and stratification is provided.
        """
        valid_split_methods = ["shuffle", "kfold"]
        if self.split_method not in valid_split_methods:
            raise ValueError(
                f"Invalid split_method: {self.split_method}. "
                "Choose 'shuffle' or 'kfold'."
            )

        if (
            self.group_column
            and self.stratified
            and self.split_method == "shuffle"
        ):
            raise ValueError(
                "Group stratified shuffle is not supported. "
                "Use split_method='kfold' for grouped and stratified splits."
            )


        valid_problem_types = ["classification", "regression"]
        if self.problem_type not in valid_problem_types:
            raise ValueError(
                f"Invalid problem_type: {self.problem_type}. "
                "Choose from 'classification' or 'regression'."
            )

    def _set_splitter(self):
        """Selects the appropriate splitter based on the configuration.

        Returns
        -------
        sklearn.model_selection._BaseKFold or
            sklearn.model_selection._Splitter: The initialized splitter
            object based on the configuration.

        Raises
        ------
        ValueError
            If invalid combination of stratified and group_column settings
            is provided.
        """
        if self.split_method == "shuffle":
            if self.group_column and not self.stratified:
                return model_selection.GroupShuffleSplit(
                    n_splits=self.n_splits, test_size=self.test_size,
                    random_state=self.random_state
                    )

            elif self.stratified and not self.group_column:
                return model_selection.StratifiedShuffleSplit(
                    n_splits=self.n_splits, test_size=self.test_size,
                    random_state=self.random_state
                    )

            elif not self.stratified and not self.group_column:
                return model_selection.ShuffleSplit(
                    n_splits=self.n_splits, test_size=self.test_size,
                    random_state=self.random_state
                    )

        elif self.split_method == "kfold":
            if self.group_column and not self.stratified:
                return model_selection.GroupKFold(n_splits=self.n_splits)

            elif self.stratified and not self.group_column:
                return model_selection.StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True if self.random_state else False,
                    random_state=self.random_state,
                )

            elif not self.stratified and not self.group_column:
                return model_selection.KFold(
                    n_splits=self.n_splits,
                    shuffle=True if self.random_state else False,
                    random_state=self.random_state,
                )

            elif self.group_column and self.stratified:
                return model_selection.StratifiedGroupKFold(
                    n_splits=self.n_splits
                )

        raise ValueError(
            "Invalid combination of stratified and group_column for "
            "the specified split method."
        )

    def _apply_preprocessing(
        self,
        X_train: pd.DataFrame,  # pylint: disable=C0103
        X_test: pd.DataFrame,  # pylint: disable=C0103
        y_train: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, List[str],
        List[str], List[str], Optional[Any]
    ]:
        """Apply all configured preprocessors to the data in fixed order:
        Missing Data → Encoding → Scaling → Feature Selection

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series, optional
            Training target values
        feature_names : List[str], optional
            Original feature names
        categorical_features : List[str], optional
            List of categorical feature names

        Returns
        -------
        tuple[
            pd.DataFrame, pd.DataFrame, List[str],
            List[str], List[str], Optional[Any]
        ]
            Transformed training data, transformed test data, updated feature names,
            updated categorical features, updated continuous features,
            and fitted scaler
        """
        if not self.preprocessors:
            # Calculate continuous features (all features except categorical)
            continuous_features = [
                f for f in (feature_names or list(X_train.columns))
                if f not in (categorical_features or [])
            ]
            return X_train, X_test, feature_names or list(X_train.columns), categorical_features or [], continuous_features, None

        current_feature_names = feature_names or list(X_train.columns)
        current_categorical_features = categorical_features or []

        # Get preprocessors by type (fixed order: Missing → Encoding → Scaling → Feature Selection)
        missing_preprocessor = next((p for p in self.preprocessors if isinstance(p, MissingDataPreprocessor)), None)
        encoding_preprocessor = next((p for p in self.preprocessors if isinstance(p, CategoricalEncodingPreprocessor)), None)
        scaling_preprocessor = next((p for p in self.preprocessors if isinstance(p, ScalingPreprocessor)), None)
        feature_selection_preprocessor = next((p for p in self.preprocessors if isinstance(p, FeatureSelectionPreprocessor)), None)

        # Step 1: Apply missing data preprocessing
        X_train, X_test, current_feature_names = self._apply_missing_data_preprocessing(
            X_train, X_test, y_train, current_feature_names, missing_preprocessor
        )

        # Step 2: Apply categorical encoding
        X_train, X_test, current_feature_names, current_categorical_features = self._apply_categorical_encoding(
            X_train, X_test, y_train, current_feature_names, current_categorical_features, encoding_preprocessor
        )

        # Step 3: Apply scaling
        X_train, X_test, fitted_scaler = self._apply_scaling(
            X_train, X_test, y_train, current_feature_names, current_categorical_features, scaling_preprocessor
        )

        # Step 4: Apply feature selection
        X_train, X_test, current_feature_names, current_categorical_features = self._apply_feature_selection(
            X_train, X_test, y_train, current_feature_names, current_categorical_features, feature_selection_preprocessor, fitted_scaler
        )

        # Calculate updated continuous features (all features except categorical)
        updated_continuous_features = [
            f for f in current_feature_names if f not in current_categorical_features
        ]

        return X_train, X_test, current_feature_names, current_categorical_features, updated_continuous_features, fitted_scaler

    def _apply_missing_data_preprocessing(
        self,
        X_train: pd.DataFrame,  # pylint: disable=C0103
        X_test: pd.DataFrame,  # pylint: disable=C0103
        y_train: Optional[pd.Series],
        current_feature_names: List[str],
        missing_preprocessor: Optional[MissingDataPreprocessor]
    ) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Apply missing data preprocessing.
        
        Handles missing values in the dataset using the configured missing data
        preprocessor. This is the first step in the preprocessing pipeline.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target values
        current_feature_names : List[str]
            Current feature names
        missing_preprocessor : Optional[MissingDataPreprocessor]
            Missing data preprocessor instance
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, List[str]]
            Transformed training data, transformed test data, updated feature names
        """
        if missing_preprocessor:
            missing_preprocessor.fit(X_train, y_train)
            X_train = missing_preprocessor.transform(X_train)
            X_test = missing_preprocessor.transform(X_test)
            current_feature_names = missing_preprocessor.get_feature_names(current_feature_names)
        else:
            # Check if there are missing values and no missing data preprocessor
            if X_train.isnull().any().any() or X_test.isnull().any().any():
                raise ValueError(
                    "Missing values detected in the data but no MissingDataPreprocessor provided. "
                    "You must handle missing values before proceeding."
                )
        return X_train, X_test, current_feature_names

    def _apply_categorical_encoding(
        self,
        X_train: pd.DataFrame,  # pylint: disable=C0103
        X_test: pd.DataFrame,  # pylint: disable=C0103
        y_train: Optional[pd.Series],
        current_feature_names: List[str],
        current_categorical_features: List[str],
        encoding_preprocessor: Optional[CategoricalEncodingPreprocessor]
    ) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """Apply categorical encoding preprocessing.
        
        Encodes categorical features using the configured encoding preprocessor.
        Updates categorical features list to include encoded feature names.
        This step must be completed before scaling to ensure categorical
        features are properly identified and excluded from scaling.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target values
        current_feature_names : List[str]
            Current feature names
        current_categorical_features : List[str]
            Current categorical feature names (will be updated with encoded features)
        encoding_preprocessor : Optional[CategoricalEncodingPreprocessor]
            Categorical encoding preprocessor instance
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]
            Transformed training data, transformed test data, updated feature names,
            updated categorical features (including encoded features)
        """
        if current_categorical_features and encoding_preprocessor:
            encoding_preprocessor.fit(X_train, y_train, categorical_features=current_categorical_features)
            X_train = encoding_preprocessor.transform(X_train)
            X_test = encoding_preprocessor.transform(X_test)
            current_feature_names = encoding_preprocessor.get_feature_names(current_feature_names)
            
            # Update categorical features to include encoded features
            updated_categorical_features = []
            for original_cat_feature in current_categorical_features:
                # Find all features that start with the original categorical feature name
                encoded_features = [
                    f for f in current_feature_names
                    if f.startswith(f"{original_cat_feature}_")
                ]
                updated_categorical_features.extend(encoded_features)
            current_categorical_features = updated_categorical_features
        elif current_categorical_features and not encoding_preprocessor:
            # Check if there are categorical features but no encoding preprocessor
            raise ValueError(
                f"Categorical features detected: {current_categorical_features} but no CategoricalEncodingPreprocessor provided. "
                "You must encode categorical features before proceeding."
            )
        return X_train, X_test, current_feature_names, current_categorical_features

    def _apply_scaling(
        self,
        X_train: pd.DataFrame,  # pylint: disable=C0103
        X_test: pd.DataFrame,  # pylint: disable=C0103
        y_train: Optional[pd.Series],
        current_feature_names: List[str],
        current_categorical_features: List[str],
        scaling_preprocessor: Optional[ScalingPreprocessor]
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
        """Apply scaling preprocessing (only to continuous features).
        
        Scales continuous features while excluding categorical features (both original
        and encoded). This step occurs after categorical encoding to ensure
        categorical features are properly identified and excluded from scaling.
        The actual transformation is applied later in DataSplitInfo.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target values
        current_feature_names : List[str]
            Current feature names
        current_categorical_features : List[str]
            Categorical features to exclude from scaling (original + encoded)
        scaling_preprocessor : Optional[ScalingPreprocessor]
            Scaling preprocessor instance
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]
            Training data, test data (unchanged), fitted scaler
        """
        fitted_scaler = None
        if scaling_preprocessor:
            # Exclude categorical features from scaling
            features_to_exclude = current_categorical_features.copy()
            
            # Check if there are any continuous features to scale
            all_features = set(current_feature_names)
            features_to_scale = all_features - set(features_to_exclude)

            if not features_to_scale:
                raise ValueError(
                    "ScalingPreprocessor provided but no continuous features found to scale. "
                    "All features are categorical or encoded categorical features. "
                    "Remove the ScalingPreprocessor or ensure there are continuous features in your data."
                )

            # Apply scaling (only to continuous features)
            scaling_preprocessor.fit(X_train, y_train, categorical_features=features_to_exclude)
            fitted_scaler = scaling_preprocessor.scaler
        return X_train, X_test, fitted_scaler

    def _apply_feature_selection(
        self,
        X_train: pd.DataFrame,  # pylint: disable=C0103
        X_test: pd.DataFrame,  # pylint: disable=C0103
        y_train: Optional[pd.Series],
        current_feature_names: List[str],
        current_categorical_features: List[str],
        feature_selection_preprocessor: Optional[FeatureSelectionPreprocessor],
        fitted_scaler: Optional[Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """Apply feature selection preprocessing.
        
        Selects features using the configured feature selection preprocessor.
        Updates both feature names and categorical features to only include
        selected features. This is the final step in the preprocessing pipeline.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target values
        current_feature_names : List[str]
            Current feature names
        current_categorical_features : List[str]
            Current categorical features (will be filtered to only selected ones)
        feature_selection_preprocessor : Optional[FeatureSelectionPreprocessor]
            Feature selection preprocessor instance
        fitted_scaler : Optional[Any]
            Fitted scaler from previous step (if any)
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]
            Transformed training data, transformed test data, updated feature names,
            updated categorical features (only selected ones)
        """
        if feature_selection_preprocessor:
            # Pass the scaler to feature selection if it exists
            if fitted_scaler:
                feature_selection_preprocessor.scaler = fitted_scaler
            feature_selection_preprocessor.fit(X_train, y_train)
            X_train = feature_selection_preprocessor.transform(X_train)
            X_test = feature_selection_preprocessor.transform(X_test)
            current_feature_names = feature_selection_preprocessor.get_feature_names(current_feature_names)
            
            # Update categorical features to only include those that were selected
            current_categorical_features = [
                f for f in current_categorical_features if f in current_feature_names
            ]
        return X_train, X_test, current_feature_names, current_categorical_features

    def _load_data(
        self,
        data_path: str,
        table_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Loads data from a CSV, Excel file, or SQL database.

        Parameters
        ----------
        data_path : str
            Path to the dataset file
        table_name : str, optional
            Name of the table in SQL database. Required for SQL databases.

        Returns
        -------
        pd.DataFrame: The loaded dataset.

        Raises
        ------
        ValueError
            If file format is unsupported or table_name is missing for SQL
            database.
        """
        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension == ".csv":
            return pd.read_csv(data_path)

        elif file_extension in [".xls", ".xlsx"]:
            return pd.read_excel(data_path)

        elif file_extension in [".db", ".sqlite"]:
            if table_name is None:
                raise ValueError(
                    "For SQL databases, 'table_name' must be provided."
                )

            conn = sqlite3.connect(data_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            return df

        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                "Supported formats are CSV, Excel, and SQL database."
            )

    def split(
        self,
        data_path: str,
        categorical_features: List[str],
        group_name: str,
        filename: str,
        table_name: Optional[str] = None,
    ) -> data_split_info.DataSplitInfo:
        """Splits the data based on the preconfigured splitter.

        Parameters
        ----------
        data_path : str
            Path to the dataset file
        categorical_features : list of str
            List of categorical feature names
        group_name : str
            Name of the group for split caching
        filename : str
            Filename for split caching
        table_name : str, optional
            Name of the table in SQL database, by default None

        Returns
        -------
        DataSplitInfo
            Object containing train/test splits and related information

        Raises
        ------
        ValueError
            If group_name is provided without filename or vice versa
        """
        split_key = (group_name, filename, table_name)

        if split_key in self._splits:
            return self._splits[split_key]

        df = self._load_data(data_path, table_name)
        X = df.iloc[:, :-1]  # pylint: disable=C0103
        y = df.iloc[:, -1]
        groups = df[self.group_column] if self.group_column else None

        if self.group_column:
            X = X.drop(columns=self.group_column)  # pylint: disable=C0103

        feature_names = list(X.columns)

        split_container = data_splits.DataSplits(self.n_splits)
        for split_index, (train_idx, test_idx) in enumerate(
            self.splitter.split(X, y, groups)
            ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] # pylint: disable=C0103
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] # pylint: disable=C0103

            # Apply preprocessing
            X_train, X_test, feature_names, categorical_features, continuous_features, fitted_scaler = self._apply_preprocessing(  # pylint: disable=C0103
                X_train, X_test, y_train, feature_names, categorical_features
            )

            if self.group_column:
                group_index_train = {
                    "values": groups.iloc[train_idx].values.copy(),
                    "indices": train_idx.copy(),
                    "series": groups.iloc[train_idx].copy(),
                }
                group_index_test = {
                    "values": groups.iloc[test_idx].values.copy(),
                    "indices": test_idx.copy(),
                    "series": groups.iloc[test_idx].copy(),
                }
            else:
                group_index_train = None
                group_index_test = None

            split = data_split_info.DataSplitInfo(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                group_index_train=group_index_train,
                group_index_test=group_index_test,
                split_key=split_key,
                split_index=split_index,
                scaler=fitted_scaler,
                categorical_features=categorical_features,
                continuous_features=continuous_features
            )
            split_container.add(split)

        self._splits[split_key] = split_container
        self.services.reporting.add_dataset(group_name, split_container)
        return split_container

    def to_markdown(self) -> str:
        """Creates a markdown representation of the DataManager configuration.

        Returns
        -------
        str: Markdown formatted string describing the configuration.
        """
        config = {
            "test_size": self.test_size,
            "n_splits": self.n_splits,
            "split_method": self.split_method,
            "group_column": self.group_column,
            "stratified": self.stratified,
            "random_state": self.random_state,
            "problem_type": self.problem_type,
            "preprocessors": [type(p).__name__ for p in self.preprocessors] if self.preprocessors else [],
        }

        md = [
            "```python",
            "DataManager Configuration:",
        ]

        for key, value in config.items():
            if value is not None:
                md.append(f"{key}: {value}")

        md.append("```")
        return "\n".join(md)

    def export_data_manager_params(self) -> None:
        """Export a JSON-serializable snapshot of the DataManager init params.
        """
        try:
            json = {
                "params": {
                    "test_size": self.test_size,
                    "n_splits": self.n_splits,
                    "split_method": self.split_method,
                    "group_column": self.group_column,
                    "stratified": self.stratified,
                    "random_state": self.random_state,
                    "problem_type": self.problem_type,
                    "preprocessors": [type(p).__name__ for p in self.preprocessors] if self.preprocessors else [],
                }
            }

            self.services.rerun.add_base_data_manager(json)
        except Exception as e:
            print(f"Warning: Failed to export DataManager params for rerun. {e}")