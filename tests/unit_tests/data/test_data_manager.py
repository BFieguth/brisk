import importlib

from numpy import int64, float64
import pandas as pd
import pytest
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit, 
    GroupKFold, StratifiedKFold, KFold, StratifiedGroupKFold
    )
from sklearn import preprocessing

from brisk.data.data_manager import DataManager
from brisk.data.data_split_info import DataSplitInfo

@pytest.fixture
def data_manager(mock_brisk_project):
    data_file = mock_brisk_project / "data.py"
    spec = importlib.util.spec_from_file_location(
        "data", str(data_file)
        )
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    return data_module.BASE_DATA_MANAGER


class TestDataManager:
    """Test class for DataManager."""
    def test_initialization(self, data_manager):
        """
        Test that the DataManager is initialized correctly.
        """
        assert data_manager.test_size == 0.2
        assert data_manager.split_method == "shuffle"
        assert data_manager.group_column == None
        assert data_manager.stratified == False
        assert data_manager.n_splits == 5
        assert data_manager.random_state == 42
        assert data_manager.scale_method == None
        assert data_manager._splits == {}

    def test_validate_config_invalid_split_method(self):
        """
        Test that an invalid split method raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid split_method: invalid_method. Choose 'shuffle' or 'kfold'."):
            DataManager(split_method="invalid_method")

    def test_validate_config_invalid_group_stratified(self):
        """
        Test that an invalid combination of group_column and stratified raises an error.
        """
        with pytest.raises(
            ValueError, match="Group stratified shuffle is not supported. Use split_method='kfold' for grouped and stratified splits."
            ):
            DataManager(
                split_method="shuffle", group_column="group", stratified=True
            )

    def test_validate_config_invalid_scaler(self):
        """
        Test that an invalid combination of group_column and stratified raises an error.
        """
        with pytest.raises(
            ValueError, match="Invalid scale_method: fake_scaler. Choose from standard, minmax, robust, maxabs, normalizer"
            ):
            DataManager(
                split_method="shuffle", scale_method="fake_scaler"
            )

    def test_valid_config_no_errors(self):
        """
        Test that a valid configuration does not raise an error.
        """
        DataManager(
            split_method="shuffle", scale_method="standard"
        )
        DataManager(
            split_method="kfold", scale_method="minmax"
        )
        DataManager(
            split_method="shuffle", scale_method="robust", group_column="group"
        )
        DataManager(
            split_method="kfold", scale_method="maxabs", group_column="group"
        )
        DataManager(
            split_method="kfold", scale_method="normalizer", group_column="group", stratified=True
        )

    def test_shuffle_without_stratified_or_group(self):
        data_manager = DataManager(
            test_size=0.2, split_method="shuffle", group_column=None, 
            stratified=False
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, ShuffleSplit)
        assert splitter_obj.n_splits == 1
        assert splitter_obj.test_size == 0.2

    def test_shuffle_with_group_column(self):
        """Test GroupShuffleSplit is selected when group_column is True and not stratified."""
        data_manager = DataManager(
            test_size=0.2, split_method="shuffle", group_column="group", 
            stratified=False
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, GroupShuffleSplit)
        assert splitter_obj.n_splits == 1
        assert splitter_obj.test_size == 0.2

    def test_shuffle_with_stratified(self):
        """Test StratifiedShuffleSplit is selected when stratified is True and no group column."""
        data_manager = DataManager(
            test_size=0.2, split_method="shuffle", group_column=None, 
            stratified=True
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, StratifiedShuffleSplit)
        assert splitter_obj.n_splits == 1
        assert splitter_obj.test_size == 0.2

    def test_kfold_with_group_column(self):
        """Test GroupKFold is selected when using kfold and group_column is True."""
        data_manager = DataManager(
            n_splits=5, split_method="kfold", group_column="group", 
            stratified=False
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, GroupKFold)
        assert splitter_obj.n_splits == 5

    def test_kfold_with_stratified(self):
        """Test StratifiedKFold is selected when using kfold and stratified is True."""
        data_manager = DataManager(
            n_splits=5, split_method="kfold", group_column=None, 
            stratified=True, random_state=42
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, StratifiedKFold)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.shuffle is True
        assert splitter_obj.random_state == 42

    def test_kfold_without_stratified_or_group(self):
        """Test KFold is selected when using kfold without stratified and group_column."""
        data_manager = DataManager(
            n_splits=5, split_method="kfold", group_column=None, 
            stratified=False, random_state=42
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, KFold)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.shuffle is True
        assert splitter_obj.random_state == 42

    def test_kfold_with_stratified_and_group_column(self):
        """Test StratifiedGroupKFold is selected when using kfold with both stratified and group_column."""
        data_manager = DataManager(
            n_splits=5, split_method="kfold", group_column="group", 
            stratified=True
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, StratifiedGroupKFold)
        assert splitter_obj.n_splits == 5

    def test_invalid_combination(self, data_manager):
        """Test ValueError is raised for invalid split method configuration."""
        data_manager.split_method = "invalid_method"
        data_manager.group_column = None
        with pytest.raises(
            ValueError, 
            match="Invalid combination of stratified and group_column for the specified split method."
            ):
            data_manager._set_splitter()

    def test_set_scaler_standard(self, data_manager):
        """
        Test that the standard scaler is selected when scale_method is 'standard'.
        """
        data_manager.scale_method = "standard"
        scaler = data_manager._set_scaler()
        assert isinstance(scaler, preprocessing.StandardScaler)

    def test_set_scaler_minmax(self, data_manager):
        """
        Test that the MinMax scaler is selected when scale_method is 'minmax'.
        """
        data_manager.scale_method = "minmax"
        scaler = data_manager._set_scaler()
        assert isinstance(scaler, preprocessing.MinMaxScaler)

    def test_set_scaler_robust(self, data_manager):
        """
        Test that the Robust scaler is selected when scale_method is 'robust'.
        """
        data_manager.scale_method = "robust"
        scaler = data_manager._set_scaler()
        assert isinstance(scaler, preprocessing.RobustScaler)

    def test_set_scaler_maxabs(self, data_manager):
        """
        Test that the MaxAbs scaler is selected when scale_method is 'maxabs'.
        """
        data_manager.scale_method = "maxabs"
        scaler = data_manager._set_scaler()
        assert isinstance(scaler, preprocessing.MaxAbsScaler)

    def test_set_scaler_normalizer(self, data_manager):
        """
        Test that the Normalizer scaler is selected when scale_method is 'normalizer'.
        """
        data_manager.scale_method = "normalizer"
        scaler = data_manager._set_scaler()
        assert isinstance(scaler, preprocessing.Normalizer)

    def test_set_scaler_invalid(self, data_manager):
        """
        Test that None is returned when an invalid scale_method is provided.
        """
        data_manager.scale_method = "invalid_scaler"
        scaler = data_manager._set_scaler()
        assert scaler is None

    def test_load_data_csv(self, data_manager, tmp_path):
        """
        Test loading data from a CSV file.
        """
        df_regression = data_manager._load_data(tmp_path / "datasets" / "regression.csv")

        # Check dataframe is correct
        assert isinstance(df_regression, pd.DataFrame)
        assert df_regression.shape == (5, 3)
        assert df_regression.columns.tolist() == ['x', 'y', 'target']
        assert df_regression.dtypes.tolist() == [float, float, float]
        # Check values are correct
        assert df_regression.iloc[0].tolist() == [1.0, 2.0, 0.5]
        assert df_regression.iloc[1].tolist() == [2.0, 3.0, 1.3]
        assert df_regression.iloc[2].tolist() == [3.0, 4.0, 0.1]
        assert df_regression.iloc[3].tolist() == [4.0, 5.0, 1.0]
        assert df_regression.iloc[4].tolist() == [5.0, 6.0, 0.8]

        df_classification = data_manager._load_data(tmp_path / "datasets" / "classification.csv")
        # Check dataframe is correct
        assert isinstance(df_classification, pd.DataFrame)
        assert df_classification.shape == (5, 3)
        assert df_classification.columns.tolist() == ['feature1', 'feature2', 'label']
        assert df_classification.dtypes.tolist() == [float, float, object]
        # Check values are correct
        assert df_classification.iloc[0].tolist() == [0.1, 0.2, 'A']
        assert df_classification.iloc[1].tolist() == [0.3, 0.4, 'B']
        assert df_classification.iloc[2].tolist() == [0.5, 0.6, 'A']
        assert df_classification.iloc[3].tolist() == [0.7, 0.8, 'B']
        assert df_classification.iloc[4].tolist() == [0.9, 1.0, 'A']

    def test_load_data_excel(self, data_manager, tmp_path):
        """
        Test loading data from an Excel file.
        """
        df_regression = data_manager._load_data(tmp_path / "datasets" / "regression.xlsx")

        assert isinstance(df_regression, pd.DataFrame)
        # Check dataframe is correct
        assert df_regression.shape == (5, 3)
        assert df_regression.columns.tolist() == ['x', 'y', 'target']
        assert df_regression.dtypes.tolist() == [int64, int64, float64]
        # Check values are correct
        assert df_regression.iloc[0].tolist() == [1, 2, 0.5]
        assert df_regression.iloc[1].tolist() == [2, 3, 1.3]
        assert df_regression.iloc[2].tolist() == [3, 4, 0.1]
        assert df_regression.iloc[3].tolist() == [4, 5, 1.0]
        assert df_regression.iloc[4].tolist() == [5, 6, 0.8]

        df_classification = data_manager._load_data(tmp_path / "datasets" / "classification.xlsx")
        # Check dataframe is correct
        assert isinstance(df_classification, pd.DataFrame)
        assert df_classification.shape == (5, 3)
        assert df_classification.columns.tolist() == ['feature1', 'feature2', 'label']
        assert df_classification.dtypes.tolist() == [float64, float64, object]
        # Check values are correct
        assert df_classification.iloc[0].tolist() == [0.1, 0.2, 'A']
        assert df_classification.iloc[1].tolist() == [0.3, 0.4, 'B']
        assert df_classification.iloc[2].tolist() == [0.5, 0.6, 'A']
        assert df_classification.iloc[3].tolist() == [0.7, 0.8, 'B']
        assert df_classification.iloc[4].tolist() == [0.9, 1.0, 'A']

    def test_load_data_sql(self, data_manager, tmp_path):
        """
        Test loading data from an SQL database.
        """
        df_regression = data_manager._load_data(tmp_path / "datasets" / "test_data.db", "regression")

        assert isinstance(df_regression, pd.DataFrame)
        # Check dataframe is correct
        assert df_regression.shape == (5, 3)
        assert df_regression.columns.tolist() == ['x', 'y', 'target']
        assert df_regression.dtypes.tolist() == [float64, float64, int64]
        # Check values are correct
        assert df_regression.iloc[0].tolist() == [1.0, 2.0, 0]
        assert df_regression.iloc[1].tolist() == [2.0, 3.0, 1]
        assert df_regression.iloc[2].tolist() == [3.0, 4.0, 0]
        assert df_regression.iloc[3].tolist() == [4.0, 5.0, 1]
        assert df_regression.iloc[4].tolist() == [5.0, 6.0, 0]

        df_classification = data_manager._load_data(tmp_path / "datasets" / "test_data.db", "classification")
        # Check dataframe is correct
        assert isinstance(df_classification, pd.DataFrame)
        assert df_classification.shape == (5, 3)
        assert df_classification.columns.tolist() == ['feature1', 'feature2', 'label']
        assert df_classification.dtypes.tolist() == [float64, float64, object]
        # Check values are correct
        assert df_classification.iloc[0].tolist() == [0.1, 0.2, 'A']
        assert df_classification.iloc[1].tolist() == [0.3, 0.4, 'B']
        assert df_classification.iloc[2].tolist() == [0.5, 0.6, 'A']
        assert df_classification.iloc[3].tolist() == [0.7, 0.8, 'B']
        assert df_classification.iloc[4].tolist() == [0.9, 1.0, 'A']

    def test_load_data_sql_no_table(self, data_manager, tmp_path):
        """
        Test loading data from an SQL database.
        """
        with pytest.raises(
            ValueError, match="For SQL databases, 'table_name' must be provided."
            ):
            data_manager._load_data(tmp_path / "datasets" / "test_data.db")

    def test_load_data_unsupported(self, data_manager):
        """
        Test loading data from a CSV file.
        """
        with pytest.raises(ValueError, match="Unsupported file format: "):
            data_manager._load_data("data.parquet")

    def test_shuffle_split(self, data_manager, tmp_path):
        """
        Test the split method using ShuffleSplit.
        """
        split = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            categorical_features=None,
            table_name=None,
            group_name="group1",
            filename="regression"
        )

        # Check data was split correctly
        assert isinstance(split, DataSplitInfo)
        assert split.X_train.shape == (4, 2)
        assert isinstance(split.X_train, pd.DataFrame)
        assert split.X_test.shape == (1, 2)
        assert isinstance(split.X_test, pd.DataFrame)
        assert split.y_train.shape == (4,)
        assert isinstance(split.y_train, pd.Series)
        assert split.y_test.shape == (1,)
        assert isinstance(split.y_test, pd.Series)
        assert split.filename == tmp_path / "datasets" / "regression.csv"
        assert split.scaler == None
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        assert data_manager._splits == {
            "group1_regression": split
        }

    def test_kfold_split(self, data_manager, tmp_path):
        """
        Test the split method using KFold.
        """
        split = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            categorical_features=None,
            table_name=None,
            group_name="group",
            filename="regression"
        )

        # Check data was split correctly
        assert isinstance(split, DataSplitInfo)
        assert split.X_train.shape == (4, 2)
        assert isinstance(split.X_train, pd.DataFrame)
        assert split.X_test.shape == (1, 2)
        assert isinstance(split.X_test, pd.DataFrame)
        assert split.y_train.shape == (4,)
        assert isinstance(split.y_train, pd.Series)
        assert split.y_test.shape == (1,)
        assert isinstance(split.y_test, pd.Series)
        assert split.filename == tmp_path / "datasets" / "regression.csv"
        assert split.scaler == None
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        assert data_manager._splits == {
            "group_regression": split
        }

    def test_shuffle_split_grouped(self, mock_brisk_project, tmp_path):
        """
        Test the split method using ShuffleSplit.
        """
        data_manager = DataManager(
            test_size=0.2, 
            split_method="shuffle", 
            group_column="group", 
            random_state=42
        )
        split = data_manager.split(
            tmp_path / "datasets" / "group.csv",
            categorical_features=None,
            table_name=None,
            group_name="group_data",
            filename="group"
        )

        # Check data was split correctly (remove group column)
        assert isinstance(split, DataSplitInfo)
        assert split.X_train.shape == (3, 2)
        assert isinstance(split.X_train, pd.DataFrame)
        assert split.X_test.shape == (2, 2)
        assert isinstance(split.X_test, pd.DataFrame)
        assert split.y_train.shape == (3,)
        assert isinstance(split.y_train, pd.Series)
        assert split.y_test.shape == (2,)
        assert isinstance(split.y_test, pd.Series)
        assert split.filename == tmp_path / "datasets" / "group.csv"
        assert split.scaler == None
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        assert data_manager._splits == {
            "group_data_group": split
        }

    def test_shuffle_split_categorical(self, data_manager, tmp_path):
        """
        Test the split method using ShuffleSplit.
        """
        split = data_manager.split(
            tmp_path / "datasets" / "categorical.csv",
            categorical_features=["category"],
            table_name=None,
            group_name="group_category",
            filename="categorical"
        )

        # Check data was split correctly
        assert isinstance(split, DataSplitInfo)
        assert split.X_train.shape == (4, 2)
        assert isinstance(split.X_train, pd.DataFrame)
        assert split.X_test.shape == (1, 2)
        assert isinstance(split.X_test, pd.DataFrame)
        assert split.y_train.shape == (4,)
        assert isinstance(split.y_train, pd.Series)
        assert split.y_test.shape == (1,)
        assert isinstance(split.y_test, pd.Series)
        assert split.filename == tmp_path / "datasets" / "categorical.csv"
        assert split.scaler == None
        assert split.features == ['value', 'category']
        assert split.categorical_features == ['category']
        # Check split was saved
        assert data_manager._splits == {
            "group_category_categorical": split
        }

    def test_shuffle_split_scaler(self, data_manager, tmp_path):
        """
        Test the split method using ShuffleSplit.
        """
        data_manager.scale_method = "standard"
        split = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            categorical_features=None,
            table_name=None,
            group_name="group_standard",
            filename="regression"
        )

        # Check data was split correctly
        assert isinstance(split, DataSplitInfo)
        assert split.X_train.shape == (4, 2)
        assert isinstance(split.X_train, pd.DataFrame)
        assert split.X_test.shape == (1, 2)
        assert isinstance(split.X_test, pd.DataFrame)
        assert split.y_train.shape == (4,)
        assert isinstance(split.y_train, pd.Series)
        assert split.y_test.shape == (1,)
        assert isinstance(split.y_test, pd.Series)
        assert split.filename == tmp_path / "datasets" / "regression.csv"
        assert isinstance(split.scaler, preprocessing.StandardScaler)
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        assert data_manager._splits == {
            "group_standard_regression": split
        }

    def test_split_caching(self, data_manager, tmp_path):
        """Test that splits are cached and reused correctly."""
        split1 = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="test_group",
            filename="regression",
            categorical_features=None
        )

        assert len(data_manager._splits) == 1
        assert "test_group_regression" in data_manager._splits
        
        split2 = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="test_group",
            filename="regression",
            categorical_features=None
        )
        assert split1 is split2  # Same object returned
        assert len(data_manager._splits) == 1 # Still only one split

        # Different group name should create new split
        split3 = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="other_group",
            filename="regression",
            categorical_features=None
        )
        assert len(data_manager._splits) == 2 # New split made for new group
        assert split1 is not split3
        assert split2 is not split3

    def test_split_without_group(self, data_manager, tmp_path):
        """Test that splits without group info use data_path as key."""
        # Split without group should use data_path as key
        split1 = data_manager.split(
            tmp_path / "datasets" / "test_data.db",
            categorical_features=None,
            table_name="regression",
            group_name=None,
            filename=None
        )

        assert len(data_manager._splits) == 1
        assert data_manager._splits == {
            tmp_path / "datasets" / "test_data.db": split1
        }
            
        split2 = data_manager.split(
            tmp_path / "datasets" / "test_data.db",
            categorical_features=None,
            table_name="regression",
            group_name=None,
            filename=None
        )
        assert split1 is split2
        assert len(data_manager._splits) == 1 # Still only one split
        assert data_manager._splits == {
            tmp_path / "datasets" / "test_data.db": split1
        }
            
    def test_split_validation(self, data_manager, tmp_path):
        """Test validation of group_name and filename parameters."""
        # Missing filename when group_name provided
        with pytest.raises(ValueError, match="Both group_name and filename must be provided together"):
            data_manager.split(
                tmp_path / "datasets" / "regression.csv",
                group_name="test_group",
                categorical_features=None
            )
            
        # Missing group_name when filename provided
        with pytest.raises(ValueError, match="Both group_name and filename must be provided together"):
            data_manager.split(
                tmp_path / "datasets" / "regression.csv",
                filename="regression",
                categorical_features=None
            )
        
        # No error when both group_name and filename provided
        data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="test_group",
            filename="regression",
            categorical_features=None
        )

    def test_to_markdown(self, data_manager):
        """Test the to_markdown method."""
        expected_markdown = """
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
random_state: 42
```
"""
        assert data_manager.to_markdown().strip() == expected_markdown.strip()
