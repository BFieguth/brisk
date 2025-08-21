import pytest
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from brisk.data.data_split_info import DataSplitInfo

@pytest.fixture
def sample_data(mock_brisk_project):
    """
    Fixture to create sample data for testing.
    """
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame({
        'feature1': rng.normal(0, 1, 100),
        'feature2': rng.normal(5, 2, 100),
        'categorical_feature': rng.choice(['A', 'B', 'C'], 100)
    })
    X_test = pd.DataFrame({
        'feature1': rng.normal(0, 1, 50),
        'feature2': rng.normal(5, 2, 50),
        'categorical_feature': rng.choice(['A', 'B', 'C'], 50)
    })
    y_train = pd.Series(rng.choice([0, 1], 100))
    y_test = pd.Series(rng.choice([0, 1], 50))

    features = ['feature1', 'feature2', 'categorical_feature']
    categorical_features = ['categorical_feature']
    continuous_features = ['feature1', 'feature2']
    group_index_train = None
    group_index_test = None
    split_key = ("group", "sample_dataset", None)
    split_index = 0

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features,
        'categorical_features': categorical_features,
        'continuous_features': continuous_features,
        'group_index_train': group_index_train,
        'group_index_test': group_index_test,
        'split_key': split_key,
        'split_index': split_index
    }


@pytest.fixture
def sample_data_scaler(mock_brisk_project):
    """
    Fixture to create sample data for testing.
    """
    rng = np.random.RandomState(50)
    X_train = pd.DataFrame({
        'feature1': rng.normal(0, 1, 100),
        'feature2': rng.normal(5, 2, 100),
        'categorical_feature': rng.choice(['A', 'B', 'C'], 100)
    })
    X_test = pd.DataFrame({
        'feature1': rng.normal(0, 1, 50),
        'feature2': rng.normal(5, 2, 50),
        'categorical_feature': rng.choice(['A', 'B', 'C'], 50)
    })
    y_train = pd.Series(rng.choice([0, 1], 100))
    y_test = pd.Series(rng.choice([0, 1], 50))

    features = ['feature1', 'feature2', 'categorical_feature']
    categorical_features = ['categorical_feature']
    continuous_features = ['feature1', 'feature2']
    scaler = StandardScaler().fit(
        X_train[[feature for feature in features 
                if feature not in categorical_features]]
        )
    group_index_train = None
    group_index_test = None
    split_key = ("group", "sample_dataset", None)
    split_index = 0

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'features': features,
        'categorical_features': categorical_features,
        'continuous_features': continuous_features,
        'group_index_train': group_index_train,
        'group_index_test': group_index_test,
        'split_key': split_key,
        'split_index': split_index
    }


@pytest.fixture
def categorical_data(mock_brisk_project):
    rng = np.random.RandomState(25)
    X_train = pd.DataFrame({
        "feature1": rng.normal(0, 1, 100),
        "feature2": rng.normal(5, 2, 100),
        "categorical_string": rng.choice(['A', 'B', 'C'], 100),
        "categorical_int": rng.choice([0, 1], 100),
        "categorical_multi_int": rng.choice([0, 1, 2, 3], 100),
        "categorical_bool": rng.choice([True, False], 100),
        "categorical_category": pd.Categorical(
            rng.choice(['Option 1', 'Option 2', 'Option 3'], 100),
            categories=['Option 1', 'Option 2', 'Option 3']
        )
    })
    X_test = pd.DataFrame({
        "feature1": rng.normal(0, 1, 50),
        "feature2": rng.normal(5, 2, 50),
        "categorical_string": rng.choice(['A', 'B', 'C'], 50),
        "categorical_int": rng.choice([0, 1], 50),
        "categorical_multi_int": rng.choice([0, 1, 2, 3], 50),
        "categorical_bool": rng.choice([True, False], 50),
        "categorical_category": pd.Categorical(
            rng.choice(['Option 1', 'Option 2', 'Option 3'], 50),
            categories=['Option 1', 'Option 2', 'Option 3']
        )
    })
    y_train = pd.Series(rng.normal(0, 1, 100))
    y_test = pd.Series(rng.normal(0, 1, 50))

    features = [
        'feature1', 'feature2', 'categorical_string', 'categorical_int',
        'categorical_bool', 'categorical_category'
    ]
    continuous_features = ["feature1", "feature2"]
    group_index_train = None
    group_index_test = None
    split_key = ("group", "categorical_dataset", None)
    split_index = 0

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features,
        'continuous_features': continuous_features,
        'group_index_train': group_index_train,
        'group_index_test': group_index_test,
        'split_key': split_key,
        'split_index': split_index
    }


class TestDataSplitInfo:
    def test_init(self, sample_data_scaler):
        """
        Test the __init__ method of DataSplitInfo.
        """
        data_info = DataSplitInfo(**sample_data_scaler)

        # Verify that attributes are correctly assigned
        assert data_info.X_train.equals(sample_data_scaler['X_train'])
        assert isinstance(data_info.X_train, pd.DataFrame)
        assert data_info.X_test.equals(sample_data_scaler['X_test'])
        assert isinstance(data_info.X_test, pd.DataFrame)
        assert data_info.y_train.equals(sample_data_scaler['y_train'])
        assert data_info.y_test.equals(sample_data_scaler['y_test'])
        assert isinstance(data_info.y_test, pd.Series)
        assert data_info.scaler == sample_data_scaler['scaler']
        assert isinstance(data_info.scaler, StandardScaler)
        assert data_info.features == sample_data_scaler['features']
        assert isinstance(data_info.features, list)
        assert data_info.categorical_features == ['categorical_feature']
        assert data_info.continuous_features == ['feature1', 'feature2']

    def test_detect_categorical_features(self, categorical_data):
        """Test categorical feature detection when names not specified."""
        data_info = DataSplitInfo(**categorical_data)
        assert data_info.categorical_features == [
            'categorical_string', 'categorical_int', 'categorical_multi_int', 
            'categorical_bool', 'categorical_category'
        ]

    def test_get_train(self, sample_data):
        """
        Test get_train without scaling.
        """
        data_info = DataSplitInfo(**sample_data)
        X_train, y_train = data_info.get_train()

        assert X_train.equals(sample_data['X_train'])
        assert y_train.equals(sample_data['y_train'])

    def test_get_train_with_scaler(self, sample_data_scaler):
        """
        Test get_train with scaling.
        """
        data_info = DataSplitInfo(**sample_data_scaler)
        X_train_scaled, y_train = data_info.get_train()

        pd.testing.assert_series_equal(y_train, sample_data_scaler['y_train'])
        assert X_train_scaled.shape == sample_data_scaler['X_train'].shape
        assert X_train_scaled.dtypes.equals(sample_data_scaler['X_train'].dtypes)

        # Verify that scaling was applied
        original_continuous = sample_data_scaler['X_train'][data_info.continuous_features]
        scaled_continuous = X_train_scaled[data_info.continuous_features]
        assert not original_continuous.equals(scaled_continuous)

        # Verify that categorical features remain unchanged
        pd.testing.assert_frame_equal(
            X_train_scaled[data_info.categorical_features],
            sample_data_scaler['X_train'][data_info.categorical_features]
        )

        # Test inverse transformation returns original data (within tolerance)
        inverse_scaled = scaled_continuous.copy()
        inverse_scaled[data_info.continuous_features] = pd.DataFrame(
            data_info.scaler.inverse_transform(scaled_continuous),
            columns=data_info.continuous_features,
            index=scaled_continuous.index
        )
        pd.testing.assert_frame_equal(
            inverse_scaled,
            original_continuous,
            check_exact=False,
            atol=1e-7
        )

    def test_get_test(self, sample_data):
        """
        Test get_test without scaling.
        """
        data_info = DataSplitInfo(**sample_data)
        X_test, y_test = data_info.get_test()

        assert X_test.equals(sample_data['X_test'])
        assert y_test.equals(sample_data['y_test'])

    def test_get_test_with_scaler(self, sample_data_scaler):
        """
        Test get_test with scaling.
        """
        data_info = DataSplitInfo(**sample_data_scaler)
        X_test_scaled, y_test = data_info.get_test()

        pd.testing.assert_series_equal(y_test, sample_data_scaler['y_test'])
        assert X_test_scaled.shape == sample_data_scaler['X_test'].shape
        assert X_test_scaled.dtypes.equals(sample_data_scaler['X_test'].dtypes)

        # Verify that scaling was applied
        original_continuous = sample_data_scaler['X_test'][data_info.continuous_features]
        scaled_continuous = X_test_scaled[data_info.continuous_features]
        assert not original_continuous.equals(scaled_continuous)

        # Verify that categorical features remain unchanged
        pd.testing.assert_frame_equal(
            X_test_scaled[data_info.categorical_features],
            sample_data_scaler['X_test'][data_info.categorical_features]
        )

        # Test inverse transformation returns original data (within tolerance)
        inverse_scaled = scaled_continuous.copy()
        inverse_scaled[data_info.continuous_features] = pd.DataFrame(
            data_info.scaler.inverse_transform(scaled_continuous),
            columns=data_info.continuous_features,
            index=scaled_continuous.index
        )
        pd.testing.assert_frame_equal(
            inverse_scaled,
            original_continuous,
            check_exact=False,
            atol=1e-7
        )

    def test_get_train_test(self, sample_data):
        """
        Test get_train_test without scaling.
        """
        data_info = DataSplitInfo(**sample_data)
        X_train, X_test, y_train, y_test = data_info.get_train_test()

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        pd.testing.assert_frame_equal(X_train, sample_data['X_train'])
        pd.testing.assert_frame_equal(X_test, sample_data['X_test'])
        pd.testing.assert_series_equal(y_train, sample_data['y_train'])
        pd.testing.assert_series_equal(y_test, sample_data['y_test'])

    def test_get_train_test_with_scaler(self, sample_data_scaler):
        """
        Test get_train_test with scaling.
        """
        data_info = DataSplitInfo(**sample_data_scaler)
        X_train_scaled, X_test_scaled, y_train, y_test = data_info.get_train_test()

        assert isinstance(X_train_scaled, pd.DataFrame)
        assert isinstance(X_test_scaled, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        assert X_train_scaled.shape == sample_data_scaler['X_train'].shape
        assert X_test_scaled.shape == sample_data_scaler['X_test'].shape

        pd.testing.assert_series_equal(y_train, sample_data_scaler['y_train'])
        pd.testing.assert_series_equal(y_test, sample_data_scaler['y_test'])

        # Verify that scaling was applied to continuous features
        original_X_train = sample_data_scaler['X_train'][data_info.continuous_features]
        scaled_X_train = X_train_scaled[data_info.continuous_features]
        assert not original_X_train.equals(scaled_X_train)

        original_X_test = sample_data_scaler['X_test'][data_info.continuous_features]
        scaled_X_test = X_test_scaled[data_info.continuous_features]
        assert not original_X_test.equals(scaled_X_test)

        # Verify that categorical features remain unchanged
        pd.testing.assert_frame_equal(
            X_train_scaled[data_info.categorical_features],
            sample_data_scaler['X_train'][data_info.categorical_features]
        )
        pd.testing.assert_frame_equal(
            X_test_scaled[data_info.categorical_features],
            sample_data_scaler['X_test'][data_info.categorical_features]
        )

    def test_get_split_metadata(self, sample_data, categorical_data):
        """Test get_split_metadata."""
        data_info = DataSplitInfo(**sample_data)
        metadata = data_info.get_split_metadata()
        assert metadata['num_features'] == 3
        assert metadata['num_samples'] == 150

        data_info = DataSplitInfo(**categorical_data)
        metadata = data_info.get_split_metadata()
        assert metadata['num_features'] == 7
        assert metadata['num_samples'] == 150
