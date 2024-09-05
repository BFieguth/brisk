import os
import sqlite3

import pandas as pd
from sklearn.model_selection import (KFold, GroupKFold, StratifiedKFold, 
                                     ShuffleSplit, GroupShuffleSplit, 
                                     StratifiedShuffleSplit, StratifiedGroupKFold)

class DataSplitter:
    """Handles the data splitting logic for train/test splits."""
    def __init__(
        self, 
        test_size=0.2, 
        split_method="shuffle", 
        group_column=None, 
        stratified=False,
        n_splits=5,
        random_state=None
    ):
        """
        Initializes the data splitter with custom splitting strategies.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test set.
            group_column (str): The name or index of the column for groups.
            stratified (bool): Whether to use stratified sampling or cross-validation.
            split_method (str): The method to use for splitting ("shuffle" or "kfold").
            n_splits (int): The number of folds (only for kfold methods).
            random_state (int): The random seed for reproducibility.
        """
        self.test_size = test_size
        self.split_method = split_method
        self.group_column = group_column
        self.stratified = stratified
        self.n_splits = n_splits
        self.random_state = random_state

        self.__validate_config()
        self.splitter = self.__set_splitter()

    def __validate_config(self):
        """Validates the provided configuration is compatible."""
        valid_split_methods = ["shuffle", "kfold"]
        if self.split_method not in valid_split_methods:
            raise ValueError(
                f"Invalid split_method: {self.split_method}. Choose 'shuffle' or 'kfold'."
                )

        if self.group_column and self.stratified and self.split_method == "shuffle":
            raise ValueError(
                "Group stratified shuffle is not supported. Use split_method='kfold' for grouped and stratified splits."
                )

    def __set_splitter(self):
        """Selects and returns the appropriate splitter based on the configuration."""
        if self.split_method == "shuffle":
            if self.group_column and not self.stratified:
                return GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            
            elif self.stratified and not self.group_column:
                return StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            
            elif not self.stratified and not self.group_column:
                return ShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            
            else:
                raise ValueError("Invalid combination of stratified and group_column for shuffle method.")
            
        elif self.split_method == "kfold":
            if self.group_column and not self.stratified:
                return GroupKFold(n_splits=self.n_splits)
            
            elif self.stratified and not self.group_column:
                return StratifiedKFold(
                    n_splits=self.n_splits, 
                    shuffle=True if self.random_state else False, 
                    random_state=self.random_state
                    )
            
            elif not self.stratified and not self.group_column:
                return KFold(
                    n_splits=self.n_splits, 
                    shuffle=True if self.random_state else False,
                    random_state=self.random_state)
            
            elif self.group_column and self.stratified:
                return StratifiedGroupKFold(n_splits=self.n_splits)
            
            else:
                raise ValueError(
                    "Invalid combination of stratified and group_column "
                    "for kfold method."
                    )

    def __load_data(self, data_path, table_name):
        """
        Helper method to load data from a CSV, Excel, or SQL database table.

        Args:
            data_path (str): Path to the data file (CSV, Excel, or SQL database).
            table_name (str, optional): The table name to load from a SQL database. Required for SQL databases.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            ValueError: If the file format is unsupported or table_name is not provided for a SQL database.
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

    def split(self, data_path, table_name=None):
        """
        Splits the data based on the preconfigured splitter.

        Args:
            data_path (str): Path to the dataset.

        Returns:
            X_train, X_test, y_train, y_test: Train/test splits of the data or cross-validation splits.
        """
        df = self.__load_data(data_path, table_name)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        groups = df[self.group_column] if self.group_column else None

        if isinstance(
            self.splitter, 
            (ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit)
            ):
            train_idx, test_idx = next(self.splitter.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            train_idx, test_idx = next(self.splitter.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        return X_train, X_test, y_train, y_test
