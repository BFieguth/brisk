import pytest
import pandas as pd
import numpy as np
import scipy

from sklearn.preprocessing import StandardScaler
from brisk.data.DataSplitInfo import DataSplitInfo

class TestDataSplitInfo:

    @pytest.fixture
    def sample_data(self):
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
    def sample_data_scaler(self):
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
            'features': features,
            'scaler': scaler,
            'categorical_features': categorical_features,
        }

    @pytest.fixture
    def tmp_dataset_dir(self, tmp_path):
        """
        Fixture to create a temporary directory for saving outputs.
        """
        return tmp_path / "dataset"

    def test_init(self, sample_data_scaler):
        """
        Test the __init__ method of DataSplitInfo.
        """
        data_info = DataSplitInfo(
            X_train=sample_data_scaler['X_train'],
            X_test=sample_data_scaler['X_test'],
            y_train=sample_data_scaler['y_train'],
            y_test=sample_data_scaler['y_test'],
            filename=sample_data_scaler['filename'],
            scaler=sample_data_scaler['scaler'],
            features=sample_data_scaler['features'],
            categorical_features=sample_data_scaler['categorical_features']
        )

        # Verify that attributes are correctly assigned
        assert data_info.X_train.equals(sample_data_scaler['X_train'])
        assert data_info.X_test.equals(sample_data_scaler['X_test'])
        assert data_info.y_train.equals(sample_data_scaler['y_train'])
        assert data_info.y_test.equals(sample_data_scaler['y_test'])
        assert data_info.filename == sample_data_scaler['filename']
        assert data_info.scaler == sample_data_scaler['scaler']
        assert data_info.features == sample_data_scaler['features']
        assert data_info.categorical_features == ['categorical_feature']
        assert data_info.continuous_features == ['feature1', 'feature2']

        # Check that statistics dictionaries are populated
        assert len(data_info.continuous_stats) == 2
        assert len(data_info.categorical_stats) == 1

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

    def test_save_distribution(self, sample_data, tmp_dataset_dir):
        """
        Test the save_distribution method.
        """
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

    def test_calculate_continuous_stats(self, sample_data):
        """Test continuous stats calculation."""
        data_info = DataSplitInfo(**sample_data)

        for feature in data_info.X_train[data_info.continuous_features].columns:
            feature_series = data_info.X_train[feature]
            stats = data_info._calculate_continuous_stats(feature_series)
        
            # Check each statistic
            assert np.isclose(stats['mean'], feature_series.mean())
            assert np.isclose(stats['median'], feature_series.median())
            assert np.isclose(stats['std_dev'], feature_series.std())
            assert np.isclose(stats['variance'], feature_series.var())
            assert np.isclose(stats['min'], feature_series.min())
            assert np.isclose(stats['max'], feature_series.max())
            assert np.isclose(stats['range'], feature_series.max() - feature_series.min())
            assert np.isclose(stats['25_percentile'], feature_series.quantile(0.25))
            assert np.isclose(stats['75_percentile'], feature_series.quantile(0.75))
            assert np.isclose(stats['skewness'], feature_series.skew())
            assert np.isclose(stats['kurtosis'], feature_series.kurt())
            
            # Check coefficient of variation
            assert np.isclose(
                stats['coefficient_of_variation'], 
                feature_series.std() / feature_series.mean()
                )  

    def test_calculate_categorical_stats(self, sample_data):
        """Test categorical stats calculation."""
        data_info = DataSplitInfo(**sample_data)

        for feature in data_info.X_train[data_info.categorical_features].columns:
            feature_series = data_info.X_train[feature]
            stats = data_info._calculate_categorical_stats(
                feature_series, feature
                )

            # Check frequencies and proportions
            assert stats['frequency'] == feature_series.value_counts().to_dict()
            assert stats['proportion'] == feature_series.value_counts(normalize=True).to_dict()
            assert stats['num_unique'] == feature_series.nunique()

            # Check entropy calculation
            entropy = -np.sum(np.fromiter(
                (p * np.log2(p) 
                for p in feature_series.value_counts(normalize=True) if p > 0),
                dtype=float
            ))
            assert np.isclose(stats['entropy'], entropy)

            # Check Chi-Square stats
            chi2_info = stats['chi_square']
            train_counts = data_info.X_train[feature].value_counts()
            test_counts = data_info.X_test[feature].value_counts()
            contingency_table = pd.concat([train_counts, test_counts], axis=1).fillna(0)
            chi2, p_value, dof, _ = scipy.stats.chi2_contingency(contingency_table)
            assert np.isclose(chi2_info['chi2_stat'], chi2)
            assert np.isclose(chi2_info['p_value'], p_value)
            assert chi2_info['degrees_of_freedom'] == dof

    def test_categorical_stats_missing_column(self, sample_data):
        """Test categorical stats calculation when the column is missing in X_test."""
        data_info = DataSplitInfo(**sample_data)
        data_info.X_test.drop(columns=['categorical_feature'], inplace=True)
        stats = data_info._calculate_categorical_stats(
            data_info.X_train['categorical_feature'], 'categorical_feature'
            )
        assert stats['chi_square'] is None
