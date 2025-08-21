"""Provides the DataManager class for creating train-test splits.

This module contains the DataManager class, which handles creating train-test
splits for machine learning models. It supports several splitting strategies
such as shuffle, k-fold, and stratified splits, with optional grouping.

Exports:
    DataManager: A class for configuring and generating train-test splits or 
    cross-validation folds.
"""
import os
import sqlite3
from typing import Optional, List

import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

from brisk.data import data_split_info
from brisk.data import data_splits
from brisk.services import get_services

class DataManager:
    """A class that handles data splitting logic for creating train-test splits.

    This class allows users to configure different splitting strategies 
    (e.g., shuffle, k-fold, stratified) and return train-test splits or 
    cross-validation folds. It supports splitting based on groupings and 
    includes options for data scaling.

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
    scale_method : str, optional
        The method to use for scaling ("standard", "minmax", "robust", "maxabs", 
        "normalizer"), by default None

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
    scale_method : str or None
        Method used for scaling features
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
        scale_method: Optional[str] = None,
    ):
        self.services = get_services()
        self.test_size = test_size
        self.split_method = split_method
        self.group_column = group_column
        self.stratified = stratified
        self.n_splits = n_splits
        self.random_state = random_state
        self.scale_method = scale_method
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
                f"Invalid scale_method: {self.scale_method}. "
                "Choose from standard, minmax, robust, maxabs, normalizer"
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
        X = df.iloc[:, :-1] # pylint: disable=C0103
        y = df.iloc[:, -1]
        groups = df[self.group_column] if self.group_column else None

        if self.group_column:
            X = X.drop(columns=self.group_column) # pylint: disable=C0103

        feature_names = list(X.columns)

        split_container = data_splits.DataSplits(self.n_splits)
        for split_index, (train_idx, test_idx) in enumerate(
            self.splitter.split(X, y, groups)
            ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] # pylint: disable=C0103
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if self.group_column:
                group_index_train = {
                    "values": groups.iloc[train_idx].values.copy(),
                    "indices": train_idx.copy(),
                    "series": groups.iloc[train_idx].copy()
                }
                group_index_test = {
                    "values": groups.iloc[test_idx].values.copy(),
                    "indices": test_idx.copy(),
                    "series": groups.iloc[test_idx].copy()
                }
            else:
                group_index_train = None
                group_index_test = None

            scaler = None
            if self.scale_method:
                scaler = self._set_scaler()

            split = data_split_info.DataSplitInfo(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                split_key=split_key,
                split_index=split_index,
                scaler=scaler,
                features=feature_names,
                categorical_features=categorical_features,
                group_index_train=group_index_train,
                group_index_test=group_index_test
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
            "scale_method": self.scale_method,
        }

        md = [
            "```python",
            "DataManager Configuration:",
        ]

        for key, value in config.items():
            if (value is not None
                and (isinstance(value, list) and value
                or not isinstance(value, list))
                ):
                md.append(f"{key}: {value}")

        md.append("```")
        return "\n".join(md)
