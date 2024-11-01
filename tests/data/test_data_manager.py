import unittest.mock as mock

import pandas as pd
import pytest
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit, 
    GroupKFold, StratifiedKFold, KFold, StratifiedGroupKFold
    )

from brisk.data.DataManager import DataManager


class TestDataManager:
    """Test class for DataManager."""

    @pytest.fixture
    def mock_data(self):
        """Mock data as a pandas DataFrame."""
        return pd.DataFrame({
            'feature1': range(10),
            'feature2': range(10, 20),
            'target': [0, 1] * 5
        })

    @pytest.fixture
    def data_splitter(self):
        """Fixture to initialize DataManager."""
        return DataManager(test_size=0.2, split_method="shuffle")

    def test_initialization(self, data_splitter):
        """
        Test that the DataManager is initialized correctly.
        """
        assert data_splitter.test_size == 0.2
        assert data_splitter.split_method == "shuffle"
        assert data_splitter.n_splits == 5

    def test_validate_config_invalid_split_method(self):
        """
        Test that an invalid split method raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid split_method"):
            DataManager(split_method="invalid_method")

    def test_validate_config_invalid_group_stratified(self):
        """
        Test that an invalid combination of group_column and stratified raises an error.
        """
        with pytest.raises(
            ValueError, match="Group stratified shuffle is not supported"
            ):
            DataManager(
                split_method="shuffle", group_column="group", stratified=True
                )

    @mock.patch("pandas.read_csv")
    def test_load_data_csv(self, mock_read_csv, mock_data):
        """
        Test loading data from a CSV file.
        """
        mock_read_csv.return_value = mock_data
        splitter = DataManager()
        df = splitter._load_data("data.csv")

        mock_read_csv.assert_called_once_with("data.csv")
        assert df.equals(mock_data)

    @mock.patch("pandas.read_excel")
    def test_load_data_excel(self, mock_read_excel, mock_data):
        """
        Test loading data from an Excel file.
        """
        mock_read_excel.return_value = mock_data
        splitter = DataManager()
        df = splitter._load_data("data.xlsx")

        mock_read_excel.assert_called_once_with("data.xlsx")
        assert df.equals(mock_data)

    @mock.patch("sqlite3.connect")
    @mock.patch("pandas.read_sql")
    def test_load_data_sql(self, mock_read_sql, mock_connect, mock_data):
        """
        Test loading data from an SQL database.
        """
        mock_read_sql.return_value = mock_data
        splitter = DataManager()
        df = splitter._load_data("database.db", "table_name")

        mock_connect.assert_called_once_with("database.db")
        mock_read_sql.assert_called_once_with(
            "SELECT * FROM table_name", mock_connect()
            )
        assert df.equals(mock_data)

    def test_shuffle_without_stratified_or_group(self):
        splitter = DataManager(
            test_size=0.2, split_method="shuffle", group_column=None, 
            stratified=False
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, ShuffleSplit)
        assert splitter_obj.n_splits == 1
        assert splitter_obj.test_size == 0.2

    def test_shuffle_with_group_column(self):
        """Test GroupShuffleSplit is selected when group_column is True and not stratified."""
        splitter = DataManager(
            test_size=0.2, split_method="shuffle", group_column="group", 
            stratified=False
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, GroupShuffleSplit)
        assert splitter_obj.n_splits == 1
        assert splitter_obj.test_size == 0.2

    def test_shuffle_with_stratified(self):
        """Test StratifiedShuffleSplit is selected when stratified is True and no group column."""
        splitter = DataManager(
            test_size=0.2, split_method="shuffle", group_column=None, 
            stratified=True
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, StratifiedShuffleSplit)
        assert splitter_obj.n_splits == 1
        assert splitter_obj.test_size == 0.2

    def test_kfold_with_group_column(self):
        """Test GroupKFold is selected when using kfold and group_column is True."""
        splitter = DataManager(
            n_splits=5, split_method="kfold", group_column="group", 
            stratified=False
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, GroupKFold)
        assert splitter_obj.n_splits == 5

    def test_kfold_with_stratified(self):
        """Test StratifiedKFold is selected when using kfold and stratified is True."""
        splitter = DataManager(
            n_splits=5, split_method="kfold", group_column=None, 
            stratified=True, random_state=42
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, StratifiedKFold)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.shuffle is True
        assert splitter_obj.random_state == 42

    def test_kfold_without_stratified_or_group(self):
        """Test KFold is selected when using kfold without stratified and group_column."""
        splitter = DataManager(
            n_splits=5, split_method="kfold", group_column=None, 
            stratified=False, random_state=42
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, KFold)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.shuffle is True
        assert splitter_obj.random_state == 42

    def test_kfold_with_stratified_and_group_column(self):
        """Test StratifiedGroupKFold is selected when using kfold with both stratified and group_column."""
        splitter = DataManager(
            n_splits=5, split_method="kfold", group_column="group", 
            stratified=True
            )
        splitter_obj = splitter._set_splitter()

        assert isinstance(splitter_obj, StratifiedGroupKFold)
        assert splitter_obj.n_splits == 5

    @mock.patch("pandas.read_csv")
    def test_shuffle_split(self, mock_read_csv, mock_data):
        """
        Test the split method using ShuffleSplit.
        """
        mock_read_csv.return_value = mock_data

        splitter = DataManager(
            test_size=0.2, split_method="shuffle", random_state=42
            )

        X_train, X_test, y_train, y_test, scaler, feature_names = splitter.split(
            "data.csv"
            )

        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

        assert len(X_train) + len(X_test) == len(mock_data)
        assert len(y_train) + len(y_test) == len(mock_data)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        assert scaler == None
        assert feature_names == ['feature1', 'feature2']

    @mock.patch("pandas.read_csv")
    def test_kfold_split(self, mock_read_csv, mock_data):
        """
        Test the split method using KFold.
        """
        mock_read_csv.return_value = mock_data
        splitter = DataManager(
            n_splits=2, split_method="kfold", random_state=42
            )
        X_train, X_test, y_train, y_test, scaler, feature_names = splitter.split(
            "data.csv"
            )

        assert len(X_train) == 5
        assert len(X_test) == 5
        assert len(y_train) == 5
        assert len(y_test) == 5

        assert len(X_train) + len(X_test) == len(mock_data)
        assert len(y_train) + len(y_test) == len(mock_data)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        assert scaler == None
        assert feature_names == ['feature1', 'feature2']