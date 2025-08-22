import importlib

from numpy import int64, float64
import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit, 
    GroupKFold, StratifiedKFold, KFold, StratifiedGroupKFold
    )
from sklearn import preprocessing

from brisk.data.data_manager import DataManager
from brisk.data.data_split_info import DataSplitInfo
from brisk.data.data_splits import DataSplits
from brisk.data.preprocessing import (
    MissingDataPreprocessor, 
    ScalingPreprocessor, 
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor
)

@pytest.fixture
def data_manager(mock_brisk_project):
    data_file = mock_brisk_project / "data.py"
    spec = importlib.util.spec_from_file_location(
        "data", str(data_file)
        )
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    data_manager = data_module.BASE_DATA_MANAGER
    return data_manager


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
        assert data_manager.n_splits == 2
        assert data_manager.random_state == 42
        assert data_manager.problem_type == "classification"
        assert data_manager.algorithm_config == None
        assert data_manager.preprocessors == []
        assert data_manager._splits == {}

    def test_validate_config_invalid_split_method(self, mock_brisk_project):
        """
        Test that an invalid split method raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid split_method: invalid_method. Choose 'shuffle' or 'kfold'."):
            DataManager(split_method="invalid_method")

    def test_validate_config_invalid_group_stratified(self, mock_brisk_project):
        """
        Test that an invalid combination of group_column and stratified raises an error.
        """
        with pytest.raises(
            ValueError, match="Group stratified shuffle is not supported. Use split_method='kfold' for grouped and stratified splits."
            ):
            DataManager(
                split_method="shuffle", group_column="group", stratified=True
            )

    def test_valid_config_no_errors(self, mock_brisk_project):
        """
        Test that a valid configuration does not raise an error.
        """
        DataManager(
            split_method="shuffle", problem_type="classification"
        )
        DataManager(
            split_method="kfold", problem_type="regression"
        )
        DataManager(
            split_method="shuffle", group_column="group"
        )
        DataManager(
            split_method="kfold", group_column="group"
        )
        DataManager(
            split_method="kfold", group_column="group", stratified=True
        )

    def test_shuffle_without_stratified_or_group(self, mock_brisk_project):
        data_manager = DataManager(
            test_size=0.2, split_method="shuffle", group_column=None, 
            stratified=False
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, ShuffleSplit)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.test_size == 0.2

    def test_shuffle_with_group_column(self, mock_brisk_project):
        """Test GroupShuffleSplit is selected when group_column is True and not stratified."""
        data_manager = DataManager(
            test_size=0.2, split_method="shuffle", group_column="group", 
            stratified=False
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, GroupShuffleSplit)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.test_size == 0.2

    def test_shuffle_with_stratified(self, mock_brisk_project):
        """Test StratifiedShuffleSplit is selected when stratified is True and no group column."""
        data_manager = DataManager(
            test_size=0.2, split_method="shuffle", group_column=None, 
            stratified=True
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, StratifiedShuffleSplit)
        assert splitter_obj.n_splits == 5
        assert splitter_obj.test_size == 0.2

    def test_kfold_with_group_column(self, mock_brisk_project):
        """Test GroupKFold is selected when using kfold and group_column is True."""
        data_manager = DataManager(
            n_splits=5, split_method="kfold", group_column="group", 
            stratified=False
            )
        splitter_obj = data_manager.splitter

        assert isinstance(splitter_obj, GroupKFold)
        assert splitter_obj.n_splits == 5

    def test_kfold_with_stratified(self, mock_brisk_project):
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

    def test_kfold_without_stratified_or_group(self, mock_brisk_project):
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

    def test_kfold_with_stratified_and_group_column(self, mock_brisk_project):
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
        data_splits = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            categorical_features=None,
            group_name="group1",
            filename="regression",
            table_name=None
        )
        split = data_splits.get_split(0)

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
        assert split.scaler == None
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        for key, value in data_manager._splits.items():
            assert key == ("group1", "regression", None)
            assert isinstance(value, DataSplits)
        assert len(data_manager._splits) == 1
        assert len(data_manager._splits[("group1", "regression", None)]) == 2

    def test_kfold_split(self, data_manager, tmp_path):
        """
        Test the split method using KFold.
        """
        data_splits = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            categorical_features=None,
            group_name="group",
            filename="regression",
            table_name=None
        )
        split = data_splits.get_split(0)

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
        assert split.scaler == None
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        for key, value in data_manager._splits.items():
            assert key == ("group", "regression", None)
            assert isinstance(value, DataSplits)
        assert len(data_manager._splits) == 1
        assert len(data_manager._splits[("group", "regression", None)]) == 2

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
        data_splits = data_manager.split(
            tmp_path / "datasets" / "group.csv",
            categorical_features=None,
            table_name=None,
            group_name="group_data",
            filename="group"
        )
        split = data_splits.get_split(0)

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
        assert split.scaler == None
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        assert isinstance(split.group_index_train, dict)
        assert isinstance(split.group_index_test, dict)
        # Check split was saved
        for key, value in data_manager._splits.items():
            assert key == ("group_data", "group", None)
            assert isinstance(value, DataSplits)
        assert len(data_manager._splits) == 1
        assert len(data_manager._splits[("group_data", "group", None)]) == 5

    def test_shuffle_split_categorical(self, data_manager, tmp_path):
        """
        Test the split method using ShuffleSplit.
        """
        data_splits = data_manager.split(
            tmp_path / "datasets" / "categorical.csv",
            categorical_features=["category"],
            table_name=None,
            group_name="group_category",
            filename="categorical"
        )
        split = data_splits.get_split(0)

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
        assert split.scaler == None
        assert split.features == ['value', 'category']
        assert split.categorical_features == ['category']
        # Check split was saved
        for key, value in data_manager._splits.items():
            assert key == ("group_category", "categorical", None)
            assert isinstance(value, DataSplits)
        assert len(data_manager._splits) == 1
        assert len(data_manager._splits[("group_category", "categorical", None)]) == 2

    def test_shuffle_split_with_preprocessing(self, data_manager, tmp_path):
        """
        Test the split method using ShuffleSplit with preprocessing.
        """
        data_manager.preprocessors = [ScalingPreprocessor()]
        data_splits = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            categorical_features=None,
            table_name=None,
            group_name="group_standard",
            filename="regression"
        )
        split = data_splits.get_split(0)

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
        assert isinstance(split.scaler, preprocessing.StandardScaler)
        assert split.features == ['x', 'y']
        assert split.categorical_features == []
        # Check split was saved
        for key, value in data_manager._splits.items():
            assert key == ("group_standard", "regression", None)
            assert isinstance(value, DataSplits)
        assert len(data_manager._splits) == 1
        assert len(data_manager._splits[("group_standard", "regression", None)]) == 2

    def test_split_caching(self, data_manager, tmp_path):
        """Test that splits are cached and reused correctly."""
        data_splits_1 = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="test_group",
            filename="regression",
            categorical_features=None
        )

        assert len(data_manager._splits) == 1
        assert ("test_group", "regression", None) in data_manager._splits
        
        data_splits_2 = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="test_group",
            filename="regression",
            categorical_features=None
        )
        assert data_splits_1 is data_splits_2  # Same object returned
        assert len(data_manager._splits) == 1 # Still only one split

        # Different group name should create new split
        data_splits_3 = data_manager.split(
            tmp_path / "datasets" / "regression.csv",
            group_name="other_group",
            filename="regression",
            categorical_features=None
        )
        assert len(data_manager._splits) == 2 # New split made for new group
        assert data_splits_1 is not data_splits_3
        assert data_splits_2 is not data_splits_3

    def test_split_without_group(self, data_manager, tmp_path):
        """Test that splits without group info use data_path as key."""
        # Split without group should use data_path as key
        data_splits_1 = data_manager.split(
            tmp_path / "datasets" / "test_data.db",
            categorical_features=None,
            table_name="regression",
            group_name="group_name",
            filename="test_data"
        )

        assert len(data_manager._splits) == 1
        assert ("group_name", "test_data", "regression") in data_manager._splits

        data_splits_2 = data_manager.split(
            tmp_path / "datasets" / "test_data.db",
            categorical_features=None,
            table_name="regression",
            group_name="group_name",
            filename="test_data"
        )
        assert data_splits_1 is data_splits_2
        assert len(data_manager._splits) == 1 # Still only one split
        assert ("group_name", "test_data", "regression") in data_manager._splits

    def test_to_markdown(self, data_manager):
        """Test the to_markdown method."""
        expected_markdown = """```python
DataManager Configuration:
test_size: 0.2
n_splits: 2
split_method: shuffle
stratified: False
random_state: 42
problem_type: classification
preprocessors: []
```"""
        assert data_manager.to_markdown().strip() == expected_markdown.strip()

    def test_preprocessing_pipeline(self, data_manager, tmp_path):
        """
        Test the complete preprocessing pipeline with multiple preprocessors.
        """       
        # Create test data with missing values and categorical features
        test_data = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [0.1, 0.2, 0.3, np.nan, 0.5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Save test data
        test_file = tmp_path / "datasets" / "test_preprocessing.csv"
        test_data.to_csv(test_file, index=False)
        
        # User's choice for feature selection
        n_features_to_select = 3
        
        # Set up preprocessors in fixed order
        data_manager.preprocessors = [
            MissingDataPreprocessor(strategy="impute", impute_method="mean"),
            CategoricalEncodingPreprocessor(method="onehot"),  # Remove categorical_features from constructor
            ScalingPreprocessor(method="standard"),
            FeatureSelectionPreprocessor(method="selectkbest", n_features_to_select=n_features_to_select, problem_type="classification")
        ]
        
        # Run split with preprocessing
        split = data_manager.split(
            test_file,
            categorical_features=['category'],
            group_name="test_preprocessing",
            filename="test"
        )
        
        # Check results
        assert isinstance(split, DataSplits)
        split = split.get_split(0)  # Get the first split
        # Check if scaling was requested (should have scaler if ScalingPreprocessor was used)
        has_scaling = any(isinstance(p, ScalingPreprocessor) for p in data_manager.preprocessors)
        if has_scaling:
            assert split.scaler is not None  # Should have fitted scaler if scaling was requested
        else:
            assert split.scaler is None  # Should not have scaler if no scaling was requested
        assert split.X_train.shape[1] == n_features_to_select  
        assert split.X_test.shape[1] == n_features_to_select   
        assert split.X_train.isnull().sum().sum() == 0  
        assert split.X_test.isnull().sum().sum() == 0  

    def test_preprocessing_flexible_combinations(self, data_manager, tmp_path):
        """
        Test that DataManager can handle any combination of preprocessors in fixed order.
        """
        test_data = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [0.1, 0.2, 0.3, np.nan, 0.5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        test_file = tmp_path / "datasets" / "test_flexible.csv"
        test_data.to_csv(test_file, index=False)
        
        # Test 1: Only scaling (numeric data only)
        numeric_data = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [0.1, 0.2, 0.3, np.nan, 0.5],
            'numeric3': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        numeric_file = tmp_path / "datasets" / "test_numeric.csv"
        numeric_data.to_csv(numeric_file, index=False)
        
        data_manager.preprocessors = [
            MissingDataPreprocessor(strategy="impute", impute_method="mean"),
            ScalingPreprocessor(method="standard")
        ]
        splits = data_manager.split(
            numeric_file, categorical_features=None,
            group_name="test_only_scaling", filename="test"
        )
        split = splits.get_split(0)
        assert isinstance(split.scaler, preprocessing.StandardScaler)
        assert split.X_train.shape[1] == 3

        # Test 4: Missing data + Encoding (skips scaling, feature selection)
        data_manager.preprocessors = [
            MissingDataPreprocessor(strategy="impute", impute_method="mean"),
            CategoricalEncodingPreprocessor(method="onehot")  # Remove categorical_features from constructor
        ]
        splits = data_manager.split(test_file, categorical_features=['category'], group_name="test_missing_encoding", filename="test")
        split = splits.get_split(0)
        assert split.scaler is None
        assert split.X_train.isnull().sum().sum() == 0  # No missing values
        # Should have more columns due to one-hot encoding
        assert split.X_train.shape[1] > 3

    def test_preprocessing_fixed_order(self, data_manager, tmp_path):
        """
        Test that preprocessing is applied in the correct fixed order:
        Missing Data → Scaling → Encoding → Feature Selection
        """       
        test_data = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [0.1, 0.2, 0.3, np.nan, 0.5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        test_file = tmp_path / "datasets" / "test_fixed_order.csv"
        test_data.to_csv(test_file, index=False)
        
        # Test full pipeline in correct order
        data_manager.preprocessors = [
            MissingDataPreprocessor(strategy="impute", impute_method="mean"),
            CategoricalEncodingPreprocessor(method="onehot"),
            ScalingPreprocessor(method="standard"),
            FeatureSelectionPreprocessor(
                method="selectkbest", n_features_to_select=3,
                problem_type="classification"
            )
        ]
        
        splits = data_manager.split(
            test_file,
            categorical_features=['category'],
            group_name="test_fixed_order",
            filename="test"
        )
        split = splits.get_split(0)        
        assert isinstance(split, DataSplitInfo)
        assert split.scaler is not None  # Scaling was applied
        assert split.X_train.shape[1] == 3  # Feature selection selected 3 features
        assert split.X_train.isnull().sum().sum() == 0  # Missing data was handled
        # Should have more columns than original due to one-hot encoding
        # (but then reduced by feature selection)
        
        # Test that order doesn't matter in user input (system finds by type)
        data_manager.preprocessors = [
            FeatureSelectionPreprocessor(method="selectkbest", n_features_to_select=3, problem_type="classification"),
            MissingDataPreprocessor(strategy="impute", impute_method="mean"),
            CategoricalEncodingPreprocessor(method="onehot"),  # Remove categorical_features from constructor
            ScalingPreprocessor(method="standard")
        ]
        
        splits2 = data_manager.split(
            test_file,
            categorical_features=['category'],
            group_name="test_fixed_order_reversed",
            filename="test"
        )
        split2 = splits2.get_split(0)
        # Should get same results regardless of input order
        assert split2.scaler is not None
        assert split2.X_train.shape[1] == 3
        assert split2.X_train.isnull().sum().sum() == 0

    def test_preprocessing_edge_cases(self, data_manager, tmp_path):
        """
        Test edge cases for preprocessing pipeline.
        """
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        test_file = tmp_path / "datasets" / "test_edge_cases.csv"
        test_data.to_csv(test_file, index=False)
        
        # Test 1: Empty preprocessors list
        data_manager.preprocessors = []
        splits = data_manager.split(test_file, categorical_features=None, group_name="test_empty", filename="test")
        split = splits.get_split(0)
        assert split.scaler is None
        assert split.X_train.shape[1] == 3  # All features
        assert split.X_train.shape == (4, 3)  # 4 rows (train), 3 features

        # Test 2: Duplicate preprocessor types (should use first one found)
        # NOTE seems to be an issue with feature names after second feature selection
        # data_manager.preprocessors = [
        #     ScalingPreprocessor(method="standard"),
        #     ScalingPreprocessor(method="minmax"),  # Duplicate - should be ignored
        #     FeatureSelectionPreprocessor(method="selectkbest", n_features_to_select=2, problem_type="classification")
        # ]
        # splits = data_manager.split(test_file, categorical_features=None, group_name="test_duplicate", filename="test")
        # split = splits.get_split(0)
        # assert split.scaler is not None
        # assert split.X_train.shape[1] == 2  # Feature selection applied
        
        # Test 3: Only missing data (no other preprocessors)
        data_manager.preprocessors = [
            MissingDataPreprocessor(strategy="impute", impute_method="mean")
        ]
        splits = data_manager.split(test_file, categorical_features=None, group_name="test_only_missing", filename="test")
        split = splits.get_split(0)
        assert split.scaler is None
        assert split.X_train.shape[1] == 3  # All features
