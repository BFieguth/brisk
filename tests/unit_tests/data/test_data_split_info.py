import logging

import pytest
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from brisk.data.data_split_info import DataSplitInfo

@pytest.fixture
def sample_data():
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

    filename = 'sample_dataset.csv'
    features = ['feature1', 'feature2', 'categorical_feature']
    categorical_features = ['categorical_feature']

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'filename': filename,
        'features': features,
        'categorical_features': categorical_features,
    }


@pytest.fixture
def sample_data_scaler():
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

    filename = 'sample_dataset.csv'
    features = ['feature1', 'feature2', 'categorical_feature']
    categorical_features = ['categorical_feature']
    scaler = StandardScaler().fit(
        X_train[[feature for feature in features 
                if feature not in categorical_features]]
        )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'filename': filename,
        'scaler': scaler,
        'features': features,
        'categorical_features': categorical_features,
    }


@pytest.fixture
def categorical_data():
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

    filename = 'categorical_dataset.csv'
    features = [
        'feature1', 'feature2', 'categorical_string', 'categorical_int',
        'categorical_bool', 'categorical_category'
    ]
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'filename': filename,
        'features': features,
    }


class TestDataSplitInfo:
    def test_init(self, sample_data_scaler):
        """
        Test the __init__ method of DataSplitInfo.
        """
        data_info = DataSplitInfo(**sample_data_scaler)

        # Check that logger is correctly initialized
        assert isinstance(data_info.logger, logging.Logger)
        assert data_info.logger.name == 'brisk.data.data_split_info'
        assert data_info.logger.level == logging.INFO
        assert len(data_info.logger.handlers) == 1
        assert isinstance(data_info.logger.handlers[0], logging.StreamHandler)
        formatter = data_info.logger.handlers[0].formatter
        assert isinstance(formatter, logging.Formatter)
        expected_format = "%(asctime)s - %(levelname)s - %(message)s"
        assert formatter._fmt == expected_format

        # Verify that attributes are correctly assigned
        assert data_info.X_train.equals(sample_data_scaler['X_train'])
        assert isinstance(data_info.X_train, pd.DataFrame)
        assert data_info.X_test.equals(sample_data_scaler['X_test'])
        assert isinstance(data_info.X_test, pd.DataFrame)
        assert data_info.y_train.equals(sample_data_scaler['y_train'])
        assert data_info.y_test.equals(sample_data_scaler['y_test'])
        assert isinstance(data_info.y_test, pd.Series)
        assert data_info.filename == sample_data_scaler['filename']
        assert isinstance(data_info.filename, str)
        assert data_info.scaler == sample_data_scaler['scaler']
        assert isinstance(data_info.scaler, StandardScaler)
        assert data_info.features == sample_data_scaler['features']
        assert isinstance(data_info.features, list)
        assert data_info.categorical_features == ['categorical_feature']
        assert data_info.continuous_features == ['feature1', 'feature2']

        # Check that statistics dictionaries are populated
        assert len(data_info.continuous_stats) == 2
        assert list(data_info.continuous_stats.keys()) == ['feature1', 'feature2']
        assert len(data_info.categorical_stats) == 1
        assert list(data_info.categorical_stats.keys()) == ['categorical_feature']

    def test_detect_categorical_features(self, categorical_data):
        """Test categorical feature detection when names not specified."""
        data_info = DataSplitInfo(**categorical_data)
        assert data_info.categorical_features == [
            'categorical_string', 'categorical_int', 'categorical_multi_int', 
            'categorical_bool', 'categorical_category'
        ]

    def test_calculate_continuous_stats(self, sample_data):
        """Test continuous stats calculation."""
        data_info = DataSplitInfo(**sample_data)

        # X_train["feature1"]
        feature_series = data_info.X_train["feature1"]
        stats = data_info._calculate_continuous_stats(feature_series)
    
        assert np.isclose(stats['mean'], -0.1038465174)
        assert np.isclose(stats['median'], -0.1269562918)
        assert np.isclose(stats['std_dev'], 0.9081684280)
        assert np.isclose(stats['variance'], 0.8247698936)
        assert np.isclose(stats['min'], -2.6197451041)
        assert np.isclose(stats['max'], 1.8522781845)
        assert np.isclose(stats['range'], 4.4720232886)
        assert np.isclose(stats['25_percentile'], -0.6009056705)
        assert np.isclose(stats['75_percentile'], 0.4059520520)
        assert np.isclose(stats['skewness'], -0.1779481426)
        assert np.isclose(stats['kurtosis'], -0.1009774535)
        assert np.isclose(stats['coefficient_of_variation'], -8.7452949872)

        # X_train["feature2"]
        feature_series = data_info.X_train["feature2"]
        stats = data_info._calculate_continuous_stats(feature_series)

        assert np.isclose(stats['mean'], 5.0446091741)
        assert np.isclose(stats['median'], 5.1682143399)
        assert np.isclose(stats['std_dev'], 1.9073379324)
        assert np.isclose(stats['variance'], 3.6379379882)
        assert np.isclose(stats['min'], 1.1624575694)
        assert np.isclose(stats['max'], 10.4403383332)
        assert np.isclose(stats['range'], 9.2778807638)
        assert np.isclose(stats['25_percentile'], 3.3886789537)
        assert np.isclose(stats['75_percentile'], 6.0763408953)
        assert np.isclose(stats['skewness'], 0.3869837647)
        assert np.isclose(stats['kurtosis'], 0.0309790269)
        assert np.isclose(stats['coefficient_of_variation'], 0.3780942917)

    def test_calculate_categorical_stats(self, sample_data):
        """Test categorical stats calculation."""
        data_info = DataSplitInfo(**sample_data)

        # X_train["categorical_feature"]
        feature_series = data_info.X_train["categorical_feature"]
        stats = data_info._calculate_categorical_stats(
            feature_series, "categorical_feature"
        )
        
        assert stats['frequency'] == {
            'A': 33, 'B': 37, 'C': 30
        }
        assert stats['proportion'] == {
            'A': 0.33, 'B': 0.37, 'C': 0.30
        }
        assert stats['num_unique'] == 3
        assert np.isclose(stats['entropy'], 1.5796412064)
        assert np.isclose(stats['chi_square']["chi2_stat"], 1.9395721925)
        assert stats['chi_square']["degrees_of_freedom"] == 2
        assert np.isclose(stats['chi_square']["p_value"], 0.3791641341)
        
        # X_test["categorical_feature"]
        feature_series = data_info.X_test["categorical_feature"]
        stats = data_info._calculate_categorical_stats(
            feature_series, "categorical_feature"
        )

        assert stats['frequency'] == {
            'A': 22, 'B': 14, 'C': 14
        }
        assert stats['proportion'] == {
            'A': 0.44, 'B': 0.28, 'C': 0.28
        }
        assert stats['num_unique'] == 3
        assert np.isclose(stats['entropy'], 1.5495875212)
        assert np.isclose(stats['chi_square']["chi2_stat"], 1.9395721925)
        assert stats['chi_square']["degrees_of_freedom"] == 2
        assert np.isclose(stats['chi_square']["p_value"], 0.3791641341)

    def test_categorical_stats_missing_column(self, sample_data):
        """Test categorical stats calculation when the column is missing in X_test."""
        data_info = DataSplitInfo(**sample_data)
        data_info.X_test.drop(columns=['categorical_feature'], inplace=True)
        stats = data_info._calculate_categorical_stats(
            data_info.X_train['categorical_feature'], 'categorical_feature'
            )
        assert stats['chi_square'] is None

    def test_get_bin_number(self,sample_data):
        data_info = DataSplitInfo(**sample_data)
        feature_series_10 = pd.Series(np.random.randint(0, 1, 10))
        feature_series_25 = pd.Series(np.random.randint(0, 1, 25))
        feature_series_50 = pd.Series(np.random.randint(0, 1, 50))
        feature_series_100 = pd.Series(np.random.randint(0, 1, 100))
        feature_series_365 = pd.Series(np.random.randint(0, 1, 365))
        feature_series_1243 = pd.Series(np.random.randint(0, 1, 1243))

        assert data_info._get_bin_number(feature_series_10) == 5
        assert data_info._get_bin_number(feature_series_25) == 6
        assert data_info._get_bin_number(feature_series_50) == 7
        assert data_info._get_bin_number(feature_series_100) == 8
        assert data_info._get_bin_number(feature_series_365) == 10
        assert data_info._get_bin_number(feature_series_1243) == 12

    def test_plot_histogram_boxplot(self, sample_data, tmp_path):
        """Test that histogram and boxplot file is created."""
        data_info = DataSplitInfo(**sample_data)
        feature_name = data_info.continuous_features[0]    
        data_info._plot_histogram_boxplot(feature_name, tmp_path)
        
        expected_file = tmp_path / f"{feature_name}_hist_box.png"
        assert expected_file.exists(), f"Plot file {expected_file} was not created"
        assert expected_file.stat().st_size > 0, "Plot file is empty"

    def test_plot_histogram_boxplot_directory_creation(self, sample_data, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        data_info = DataSplitInfo(**sample_data)
        feature_name = data_info.continuous_features[0]
        
        # Use a nested directory that doesn't exist
        nested_dir = tmp_path / "nested" / "plot_dir"
        data_info._plot_histogram_boxplot(feature_name, nested_dir)
        
        # Check both directory and file were created
        assert nested_dir.exists()
        expected_file = nested_dir / f"{feature_name}_hist_box.png"
        assert expected_file.exists()

    def test_plot_correlation_matrix(self, sample_data, tmp_path):
        """Test that correlation matrix file is created."""
        data_info = DataSplitInfo(**sample_data)
        data_info._plot_correlation_matrix(tmp_path)
        
        expected_file = tmp_path / "correlation_matrix.png"
        assert expected_file.exists(), f"Plot file {expected_file} was not created"
        assert expected_file.stat().st_size > 0, "Plot file is empty"

    def test_plot_correlation_matrix_directory_creation(self, sample_data, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        data_info = DataSplitInfo(**sample_data)
        
        # Use a nested directory that doesn't exist
        nested_dir = tmp_path / "nested" / "plot_dir"
        data_info._plot_correlation_matrix(nested_dir)
        
        # Check both directory and file were created
        assert nested_dir.exists()
        expected_file = nested_dir / "correlation_matrix.png"
        assert expected_file.exists()

    def test_plot_categorical_pie(self, sample_data, tmp_path):
        """Test that categorical pie plot file is created."""
        data_info = DataSplitInfo(**sample_data)
        feature_name = data_info.categorical_features[0]    
        data_info._plot_categorical_pie(feature_name, tmp_path)

        expected_file = tmp_path / f"{feature_name}_pie_plot.png"
        assert expected_file.exists(), f"Plot file {expected_file} was not created"
        assert expected_file.stat().st_size > 0, "Plot file is empty"

    def test_plot_categorical_pie_directory_creation(self, sample_data, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        data_info = DataSplitInfo(**sample_data)
        feature_name = data_info.categorical_features[0]
        
        # Use a nested directory that doesn't exist
        nested_dir = tmp_path / "nested" / "plot_dir"
        data_info._plot_categorical_pie(feature_name, nested_dir)
        
        # Check both directory and file were created
        assert nested_dir.exists()
        expected_file = nested_dir / f"{feature_name}_pie_plot.png"
        assert expected_file.exists()

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

    def test_save_distribution(self, sample_data, tmp_path):
        """
        Test the save_distribution method.
        """
        tmp_dataset_dir = tmp_path / "dataset_dir"
        data_info = DataSplitInfo(**sample_data)
        data_info.save_distribution(tmp_dataset_dir)

        # Check if JSON files are created
        continuous_stats_path = tmp_dataset_dir / 'continuous_stats.json'
        categorical_stats_path = tmp_dataset_dir / 'categorical_stats.json'
        assert continuous_stats_path.exists()
        assert categorical_stats_path.exists()

        # Check if plot directories are created
        hist_box_plot_dir = tmp_dataset_dir / 'hist_box_plot'
        pie_plot_dir = tmp_dataset_dir / 'pie_plot'
        assert hist_box_plot_dir.exists()
        assert pie_plot_dir.exists()

        # Check if plots are created
        for feature in data_info.continuous_features:
            plot_path = hist_box_plot_dir / f"{feature}_hist_box.png"
            assert plot_path.exists()

        correlation_plot = tmp_dataset_dir / "correlation_matrix.png"
        assert correlation_plot.exists()

        for feature in data_info.categorical_features:
            plot_path = pie_plot_dir / f"{feature}_pie_plot.png"
            assert plot_path.exists()

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
