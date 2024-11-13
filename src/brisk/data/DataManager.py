"""Provides the DataManager class for creating train-test splits.

This module contains the `DataManager` class, which handles creating train-test 
splits for machine learning models. It supports several splitting strategies 
such as shuffle, k-fold, and stratified splits, with optional grouping.

Exports:
    - DataManager: A class for configuring and generating train-test splits or 
        cross-validation folds.
"""

import os
import sqlite3
from typing import Optional, Tuple

import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing

class DataManager:
    """Handles the data splitting logic for creating train-test splits.

    This class allows users to configure different splitting strategies 
    (e.g., shuffle, k-fold, stratified) and return train-test splits or 
    cross-validation folds. Supports splitting based on groupings.

    Attributes:
        test_size (float): The proportion of the dataset to allocate to the test set.
        n_splits (int): The number of splits for cross-validation (if applicable).
        split_method (str): The method used for splitting ("shuffle" or "kfold").
        group_column (Optional[str]): The column used for grouping (if any).
        stratified (bool): Whether to use stratified sampling or cross-validation.
        random_state (Optional[int]): The random seed for reproducibility.
    """
    def __init__(
        self, 
        test_size: float = 0.2, 
        n_splits: int = 5,
        split_method: str = "shuffle", 
        group_column: Optional[str] = None, 
        stratified: bool = False,
        random_state: Optional[int] = None,
        scale_method: Optional[str] = None,
        categorical_features: Optional[list] = []
    ):
        """Initializes the DataManager with custom splitting strategies.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test set. Defaults to 0.2.
            n_splits (int): Number of splits for cross-validation. Defaults to 5.
            split_method (str): The method to use for splitting ("shuffle" or "kfold"). Defaults to "shuffle".
            group_column (Optional[str]): The column to use for grouping (if any). Defaults to None.
            stratified (bool): Whether to use stratified sampling or cross-validation. Defaults to False.
            random_state (Optional[int]): The random seed for reproducibility. Defaults to None.
        """
        self.test_size = test_size
        self.split_method = split_method
        self.group_column = group_column
        self.stratified = stratified
        self.n_splits = n_splits
        self.random_state = random_state
        self.scale_method = scale_method
        self.categorical_features = categorical_features

        self._validate_config()
        self.splitter = self._set_splitter()

    def _validate_config(self) -> None:
        """Validates the provided configuration for splitting.

        Raises:
            ValueError: If an invalid split method or incompatible combination of
                        group column and stratification is provided.
        """
        valid_split_methods = ["shuffle", "kfold"]
        if self.split_method not in valid_split_methods:
            raise ValueError(
                f"Invalid split_method: {self.split_method}. "
                "Choose 'shuffle' or 'kfold'."
                )

        if (self.group_column and 
            self.stratified and 
            self.split_method == "shuffle"
            ):
            raise ValueError(
                "Group stratified shuffle is not supported. "
                "Use split_method='kfold' for grouped and stratified splits."
                )
        
        valid_scale_methods = [
            "standard", "minmax", "robust", "maxabs", "normalizer", None
            ]
        if self.scale_method not in valid_scale_methods:
            raise ValueError(
                f"Invalid scale_method: {self.scale_method}."
                "Choose from standard, minmax, robust, maxabs, normalizer"
                )

    def _set_splitter(self):
        """Selects and returns the appropriate splitter based on the configuration.

        Returns:
            sklearn.model_selection._BaseKFold or sklearn.model_selection._Splitter: 
                The initialized splitter object based on the configuration.

        Raises:
            ValueError: If an invalid combination of stratified and group_column 
                        settings is provided.
        """
        if self.split_method == "shuffle":
            if self.group_column and not self.stratified:
                return model_selection.GroupShuffleSplit(
                    n_splits=1, test_size=self.test_size, 
                    random_state=self.random_state
                    )
            
            elif self.stratified and not self.group_column:
                return model_selection.StratifiedShuffleSplit(
                    n_splits=1, test_size=self.test_size, 
                    random_state=self.random_state
                    )
            
            elif not self.stratified and not self.group_column:
                return model_selection.ShuffleSplit(
                    n_splits=1, test_size=self.test_size, 
                    random_state=self.random_state
                    )
            
        elif self.split_method == "kfold":
            if self.group_column and not self.stratified:
                return model_selection.GroupKFold(n_splits=self.n_splits)
            
            elif self.stratified and not self.group_column:
                return model_selection.StratifiedKFold(
                    n_splits=self.n_splits, 
                    shuffle=True if self.random_state else False, 
                    random_state=self.random_state
                    )
            
            elif not self.stratified and not self.group_column:
                return model_selection.KFold(
                    n_splits=self.n_splits, 
                    shuffle=True if self.random_state else False,
                    random_state=self.random_state
                    )
            
            elif self.group_column and self.stratified:
                return model_selection.StratifiedGroupKFold(
                    n_splits=self.n_splits
                    )
            
        raise ValueError(
            "Invalid combination of stratified and group_column for "
            "the specified split method."
            )

    def _set_scaler(self):
        if self.scale_method == "standard":
            return preprocessing.StandardScaler()

        if self.scale_method == "minmax":
            return preprocessing.MinMaxScaler()

        if self.scale_method == "robust":
            return preprocessing.RobustScaler()

        if self.scale_method == "maxabs":
            return preprocessing.MaxAbsScaler()

        if self.scale_method == "normalizer":
            return preprocessing.Normalizer()
        
        else:
            return None

    def _load_data(
        self, 
        data_path: str, 
        table_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Loads data from a CSV, Excel file, or SQL database.

        Args:
            data_path (str): Path to the dataset.
            table_name (Optional[str]): Name of the table in the SQL database 
                (if applicable). Defaults to None.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            ValueError: If the file format is unsupported or if table_name is 
                missing for an SQL database.
        """
        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension == '.csv':
            return pd.read_csv(data_path)
        
        elif file_extension in ['.xls', '.xlsx']:
            return pd.read_excel(data_path)
        
        elif file_extension in ['.db', '.sqlite']:
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
        table_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data based on the preconfigured splitter.

        Args:
            data_path (str): Path to the dataset.
            table_name (Optional[str]): Name of the table in the SQL database 
                (if applicable). Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
                A tuple containing:
                - X_train (pd.DataFrame): The training features.
                - X_test (pd.DataFrame): The testing features.
                - y_train (pd.Series): The training target variable.
                - y_test (pd.Series): The testing target variable.
        """
        df = self._load_data(data_path, table_name)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        groups = df[self.group_column] if self.group_column else None
        
        if self.group_column:
            X = X.drop(columns=self.group_column)

        if self.categorical_features:
            continuous_features = [
                col for col in X.columns if col not in self.categorical_features
                ]
        else:
            continuous_features = X.columns

        feature_names = list(X.columns)

        train_idx, test_idx = next(self.splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if self.scale_method:
            scaler = self._set_scaler()
            scaler.fit(X_train[continuous_features])

        return X_train, X_test, y_train, y_test, scaler, feature_names
