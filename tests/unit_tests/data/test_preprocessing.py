"""Unit tests for the preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from brisk.data.preprocessing import (
    BasePreprocessor,
    MissingDataPreprocessor,
    ScalingPreprocessor,
    FeatureSelectionPreprocessor,
    CategoricalEncodingPreprocessor,
)


@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        'feature1': rng.normal(0, 1, 100),
        'feature2': rng.normal(5, 2, 100),
        'feature3': rng.normal(10, 3, 100),
        'feature4': rng.normal(-2, 1, 100),
        'categorical_feature': rng.choice(['A', 'B', 'C'], 100),
        'binary_feature': rng.choice([0, 1], 100),
    })
    y = pd.Series(rng.choice([0, 1], 100))
    return X, y





@pytest.fixture
def missing_data():
    """Fixture to create data with missing values for testing."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, np.nan, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'categorical_feature': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


@pytest.fixture
def categorical_data():
    """Fixture to create categorical data for testing."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        'numeric1': rng.normal(0, 1, 100),
        'numeric2': rng.normal(5, 2, 100),
        'category1': rng.choice(['A', 'B', 'C'], 100),
        'category2': rng.choice(['X', 'Y', 'Z'], 100),
        'binary_category': rng.choice([0, 1], 100),
    })
    y = pd.Series(rng.choice([0, 1], 100))
    return X, y


class TestBasePreprocessor:
    """Test the base preprocessor class."""
    
    def test_base_preprocessor_abstract(self):
        """Test that BasePreprocessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePreprocessor()


class TestMissingDataPreprocessor:
    """Test the missing data preprocessor."""
    
    def test_initialization(self):
        """Test that MissingDataPreprocessor initializes correctly."""
        preprocessor = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        assert preprocessor.strategy == "impute"
        assert preprocessor.impute_method == "mean"
        assert preprocessor.constant_value == 0
        assert preprocessor.constant_values == {}
        assert not preprocessor.is_fitted
    
    def test_drop_rows_strategy(self, missing_data):
        """Test drop rows strategy."""
        X, y = missing_data
        preprocessor = MissingDataPreprocessor(strategy="drop_rows")
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Should drop rows with any missing values
        assert X_transformed.shape[0] < X.shape[0]
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_impute_mean_strategy(self, missing_data):
        """Test impute with mean strategy."""
        X, y = missing_data
        preprocessor = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Should have same shape and no missing values
        assert X_transformed.shape == X.shape
        assert X_transformed.isnull().sum().sum() == 0
        
        # Check that missing values were filled with mean
        assert np.isclose(X_transformed.loc[2, 'feature1'], X['feature1'].mean())
        assert np.isclose(X_transformed.loc[3, 'feature2'], X['feature2'].mean())
    
    def test_impute_median_strategy(self, missing_data):
        """Test impute with median strategy."""
        X, y = missing_data
        preprocessor = MissingDataPreprocessor(strategy="impute", impute_method="median")
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed.shape == X.shape
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_impute_constant_strategy(self, missing_data):
        """Test impute with constant strategy."""
        X, y = missing_data
        preprocessor = MissingDataPreprocessor(strategy="impute", impute_method="constant", constant_value=999)
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed.shape == X.shape
        assert X_transformed.isnull().sum().sum() == 0
        assert X_transformed.loc[2, 'feature1'] == 999
        assert X_transformed.loc[3, 'feature2'] == 999
    
    def test_impute_mode_strategy(self, missing_data):
        """Test impute with mode strategy."""
        X, y = missing_data
        preprocessor = MissingDataPreprocessor(strategy="impute", impute_method="mode")
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed.shape == X.shape
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Invalid strategy: invalid. Choose from"):
            MissingDataPreprocessor(strategy="invalid")
    
    def test_invalid_impute_method(self):
        """Test that invalid impute method raises error."""
        with pytest.raises(ValueError, match="Invalid impute_method: invalid. Choose from"):
            MissingDataPreprocessor(strategy="impute", impute_method="invalid")
    
    def test_transform_without_fit(self, missing_data):
        """Test that transform without fit raises error."""
        X, y = missing_data
        preprocessor = MissingDataPreprocessor(strategy="impute")
        with pytest.raises(ValueError, match="Preprocessor must be fitted before transform"):
            preprocessor.transform(X)


class TestScalingPreprocessor:
    """Test the scaling preprocessor."""
    
    def test_initialization(self):
        """Test that ScalingPreprocessor initializes correctly."""
        preprocessor = ScalingPreprocessor(method="standard")
        assert preprocessor.method == "standard"
        assert preprocessor.scaler is None
    
    def test_standard_scaling(self, categorical_data):
        """Test standard scaling."""
        X, y = categorical_data
        preprocessor = ScalingPreprocessor(method="standard")
        preprocessor.fit(X, y, categorical_features=['category1', 'category2', 'binary_category'])
        X_transformed = preprocessor.transform(X)
        
        # Check that numeric columns are scaled
        assert np.isclose(X_transformed['numeric1'].mean(), 0, atol=1e-10)
        assert np.isclose(X_transformed['numeric2'].mean(), 0, atol=1e-10)
        assert 0.99 <= X_transformed['numeric1'].std() <= 1.01
        assert 0.99 <= X_transformed['numeric2'].std() <= 1.01    
        
        # Check that categorical columns are unchanged
        assert (X_transformed['category1'] == X['category1']).all()
        assert (X_transformed['category2'] == X['category2']).all()
        assert (X_transformed['binary_category'] == X['binary_category']).all()
    
    def test_minmax_scaling(self, categorical_data):
        """Test minmax scaling."""
        X, y = categorical_data
        preprocessor = ScalingPreprocessor(method="minmax")
        preprocessor.fit(X, y, categorical_features=['category1', 'category2', 'binary_category'])
        X_transformed = preprocessor.transform(X)
        
        # Check that numeric columns are scaled to [0, 1]
        assert np.isclose(X_transformed['numeric1'].min(), 0, atol=1e-10)
        assert np.isclose(X_transformed['numeric1'].max(), 1, atol=1e-10)
        assert np.isclose(X_transformed['numeric2'].min(), 0, atol=1e-10)
        assert np.isclose(X_transformed['numeric2'].max(), 1, atol=1e-10)
        
        # Check that categorical columns are unchanged
        assert (X_transformed['category1'] == X['category1']).all()
    
    def test_robust_scaling(self, categorical_data):
        """Test robust scaling."""
        X, y = categorical_data
        preprocessor = ScalingPreprocessor(method="robust")
        preprocessor.fit(X, y, categorical_features=['category1', 'category2', 'binary_category'])
        X_transformed = preprocessor.transform(X)
        
        # Check that categorical columns are unchanged
        assert (X_transformed['category1'] == X['category1']).all()
    
    def test_no_numeric_features(self, categorical_data):
        """Test scaling when no numeric features are present."""
        X, y = categorical_data
        preprocessor = ScalingPreprocessor(method="standard")
        preprocessor.fit(X, y, categorical_features=['numeric1', 'numeric2', 'category1', 'category2', 'binary_category'])
        X_transformed = preprocessor.transform(X)
        
        # Should return unchanged data
        assert X_transformed.equals(X)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method must be one of"):
            ScalingPreprocessor(method="invalid")


class TestFeatureSelectionPreprocessor:
    """Test the feature selection preprocessor."""
    
    def test_initialization(self):
        """Test that FeatureSelectionPreprocessor initializes correctly."""
        preprocessor = FeatureSelectionPreprocessor(
            method="selectkbest",
            n_features_to_select=3,
            problem_type="classification"
        )
        assert preprocessor.method == "selectkbest"
        assert preprocessor.n_features_to_select == 3
        assert preprocessor.problem_type == "classification"
        assert preprocessor.selector is None
        assert preprocessor.scaler is None
    
    def test_selectkbest_classification(self, sample_data):
        """Test SelectKBest feature selection for classification."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(method="onehot")
        encoder.fit(X, y, categorical_features=['categorical_feature'])
        X_encoded = encoder.transform(X)
        
        # Now feature selection can work on encoded data
        preprocessor = FeatureSelectionPreprocessor(
            method="selectkbest",
            n_features_to_select=2,
            problem_type="classification"
        )
        
        preprocessor.fit(X_encoded, y)
        X_transformed = preprocessor.transform(X_encoded)
        
        assert X_transformed.shape[1] >= 1
        assert len(preprocessor.get_feature_names(list(X_encoded.columns))) >= 1
    
    def test_selectkbest_regression(self, sample_data):
        """Test SelectKBest feature selection for regression."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(method="onehot")
        encoder.fit(X, y, categorical_features=['categorical_feature'])
        X_encoded = encoder.transform(X)
        
        # Now feature selection can work on encoded data
        preprocessor = FeatureSelectionPreprocessor(
            method="selectkbest",
            n_features_to_select=2,
            problem_type="regression"
        )
        
        preprocessor.fit(X_encoded, y)
        X_transformed = preprocessor.transform(X_encoded)
        
        assert X_transformed.shape[1] >= 1
        assert len(preprocessor.get_feature_names(list(X_encoded.columns))) >= 1
    
    def test_rfecv_with_estimator(self, sample_data):
        """Test RFECV with estimator."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(method="onehot")
        encoder.fit(X, y, categorical_features=['categorical_feature'])
        X_encoded = encoder.transform(X)
        
        estimator = LogisticRegression(random_state=42)
        preprocessor = FeatureSelectionPreprocessor(
            method="rfecv",
            estimator=estimator,
            n_features_to_select=2
        )
        
        preprocessor.fit(X_encoded, y)
        X_transformed = preprocessor.transform(X_encoded)
        assert X_transformed.shape[1] >= 2
    
    def test_sequential_with_estimator(self, sample_data):
        """Test SequentialFeatureSelector with estimator."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(method="onehot")
        encoder.fit(X, y, categorical_features=['categorical_feature'])
        X_encoded = encoder.transform(X)
        
        estimator = LogisticRegression(random_state=42)
        preprocessor = FeatureSelectionPreprocessor(
            method="sequential",
            estimator=estimator,
            n_features_to_select=2
        )
        
        preprocessor.fit(X_encoded, y)
        X_transformed = preprocessor.transform(X_encoded)
        assert X_transformed.shape[1] == 2
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method: invalid. Choose from"):
            FeatureSelectionPreprocessor(method="invalid")
    
    def test_invalid_n_features(self):
        """Test that invalid n_features_to_select raises error."""
        with pytest.raises(ValueError, match="n_features_to_select must be >= 1"):
            FeatureSelectionPreprocessor(n_features_to_select=0)
    
    def test_invalid_problem_type(self):
        """Test that invalid problem_type raises error."""
        with pytest.raises(ValueError, match="Invalid problem_type: invalid. Choose from"):
            FeatureSelectionPreprocessor(problem_type="invalid")
    
    def test_invalid_cv(self):
        """Test that invalid feature_selection_cv raises error."""
        with pytest.raises(ValueError, match="feature_selection_cv must be >= 2"):
            FeatureSelectionPreprocessor(feature_selection_cv=1)
    
    def test_rfecv_without_estimator(self, sample_data):
        """Test that RFECV without estimator raises error."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(
            method="onehot",
            categorical_features=['categorical_feature']
        )
        X_encoded = encoder.fit_transform(X, y)
        
        preprocessor = FeatureSelectionPreprocessor(method="rfecv")
        with pytest.raises(ValueError, match="algorithm_config must be provided"):
            preprocessor.fit(X_encoded, y)
    
    def test_sequential_without_estimator(self, sample_data):
        """Test that SequentialFeatureSelector without estimator raises error."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(
            method="onehot",
            categorical_features=['categorical_feature']
        )
        X_encoded = encoder.fit_transform(X, y)
        
        preprocessor = FeatureSelectionPreprocessor(method="sequential")
        with pytest.raises(ValueError, match="algorithm_config must be provided"):
            preprocessor.fit(X_encoded, y)
    
    def test_transform_without_fit(self, sample_data):
        """Test that transform without fit raises error."""
        X, y = sample_data
        
        # First encode categorical features (real flow)
        encoder = CategoricalEncodingPreprocessor(
            method="onehot",
            categorical_features=['categorical_feature']
        )
        X_encoded = encoder.fit_transform(X, y)
        
        preprocessor = FeatureSelectionPreprocessor(method="selectkbest")
        with pytest.raises(ValueError, match="Preprocessor must be fitted before transform"):
            preprocessor.transform(X_encoded)


class TestCategoricalEncodingPreprocessor:
    """Test the categorical encoding preprocessor."""
    
    def test_initialization(self):
        """Test that CategoricalEncodingPreprocessor initializes correctly."""
        preprocessor = CategoricalEncodingPreprocessor(method="label")
        assert preprocessor.method == "label"
        assert preprocessor.encoders == {}
        assert not preprocessor.is_fitted
    
    def test_label_encoding(self, categorical_data):
        """Test label encoding."""
        X, y = categorical_data
        preprocessor = CategoricalEncodingPreprocessor(method="label")
        
        preprocessor.fit(X, y, categorical_features=['category1', 'category2', 'binary_category'])
        X_transformed = preprocessor.transform(X)
        
        # Check that categorical columns are now numeric
        assert X_transformed['category1'].dtype in ['int64', 'int32']
        assert X_transformed['category2'].dtype in ['int64', 'int32']
        assert X_transformed['binary_category'].dtype in ['int64', 'int32']
        assert X_transformed.shape == X.shape
    
    def test_onehot_encoding(self, categorical_data):
        """Test one-hot encoding."""
        X, y = categorical_data
        preprocessor = CategoricalEncodingPreprocessor(method="onehot")
        
        preprocessor.fit(X, y, categorical_features=['category1', 'category2'])
        X_transformed = preprocessor.transform(X)
        
        # Check that we have more columns after one-hot encoding
        assert X_transformed.shape[1] > X.shape[1]
        # Check that original categorical columns are gone
        assert 'category1' not in X_transformed.columns
        assert 'category2' not in X_transformed.columns
        # Check that we have new one-hot columns
        assert any('category1_' in col for col in X_transformed.columns)
        assert any('category2_' in col for col in X_transformed.columns)
    
    def test_ordinal_encoding(self, categorical_data):
        """Test ordinal encoding."""
        X, y = categorical_data
        preprocessor = CategoricalEncodingPreprocessor(method="ordinal")
        
        preprocessor.fit(X, y, categorical_features=['category1', 'category2'])
        X_transformed = preprocessor.transform(X)
        
        # Check that categorical columns are now numeric
        assert X_transformed['category1'].dtype in ['float64', 'float32']
        assert X_transformed['category2'].dtype in ['float64', 'float32']
        assert X_transformed.shape == X.shape
    

    
    def test_cyclic_encoding(self, categorical_data):
        """Test cyclic encoding."""
        X, y = categorical_data
        preprocessor = CategoricalEncodingPreprocessor(method="cyclic")
        
        preprocessor.fit(X, y, categorical_features=['category1'])
        X_transformed = preprocessor.transform(X)
        
        # Check that original column is gone and sin/cos columns are created
        assert 'category1' not in X_transformed.columns
        assert 'category1_sin' in X_transformed.columns
        assert 'category1_cos' in X_transformed.columns
        assert X_transformed.shape[1] == X.shape[1] + 1  # +1 for the extra sin/cos column
    
    def test_column_specific_encoding(self, categorical_data):
        """Test column-specific encoding methods."""
        X, y = categorical_data
        preprocessor = CategoricalEncodingPreprocessor(
            method={
                'category1': 'onehot',
                'category2': 'label'
            }
        )
        
        preprocessor.fit(X, y, categorical_features=['category1', 'category2'])
        X_transformed = preprocessor.transform(X)
        
        # Check that category1 is one-hot encoded
        assert 'category1' not in X_transformed.columns
        assert any('category1_' in col for col in X_transformed.columns)
        
        # Check that category2 is label encoded
        assert 'category2' in X_transformed.columns
        assert X_transformed['category2'].dtype in ['int64', 'int32']
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method: invalid. Choose from"):
            CategoricalEncodingPreprocessor(method="invalid")
    
    def test_invalid_column_specific_method(self):
        """Test that invalid column-specific method raises error."""
        with pytest.raises(ValueError, match="Invalid method 'invalid' for column"):
            CategoricalEncodingPreprocessor(
                method={'category1': 'invalid'},
                categorical_features=['category1']
            )
    
    def test_invalid_method_type(self):
        """Test that invalid method type raises error."""
        with pytest.raises(ValueError, match="method must be a string or dict"):
            CategoricalEncodingPreprocessor(method=123)
    
    def test_transform_without_fit(self, categorical_data):
        """Test that transform without fit raises error."""
        X, y = categorical_data
        preprocessor = CategoricalEncodingPreprocessor(method="label")
        with pytest.raises(ValueError, match="Preprocessor must be fitted before transform"):
            preprocessor.transform(X)


class TestPreprocessorPipeline:
    """Test multiple preprocessors in a pipeline."""
    
    def test_encoding_then_feature_selection(self, categorical_data):
        """Test encoding followed by feature selection."""
        X, y = categorical_data
                # Create preprocessors
        encoder = CategoricalEncodingPreprocessor(method="onehot")
        selector = FeatureSelectionPreprocessor(
            method="selectkbest",
            n_features_to_select=3,
            problem_type="classification"
        )
    
                # Apply encoding first
        encoder.fit(X, y, categorical_features=['category1', 'category2'])
        X_encoded = encoder.transform(X)
        
        # Then apply feature selection
        selector.fit(X_encoded, y)
        X_selected = selector.transform(X_encoded)
        
        # Check that we have the expected number of features
        assert X_selected.shape[1] == 3
    
    def test_missing_data_then_scaling(self, missing_data):
        """Test missing data handling followed by scaling."""
        X, y = missing_data
        # Create preprocessors
        missing_processor = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        scaler = ScalingPreprocessor(method="standard")
        
        # Apply missing data handling first
        missing_processor.fit(X, y)
        X_imputed = missing_processor.transform(X)
        
        # Then apply scaling
        scaler.fit(X_imputed, y, categorical_features=['categorical_feature'])
        X_scaled = scaler.transform(X_imputed)
        
        # Check that no missing values remain
        assert X_scaled.isnull().sum().sum() == 0
        # Check that scaling was applied
        assert np.isclose(X_scaled['feature1'].mean(), 0, atol=1e-10)
    
    def test_full_pipeline(self, missing_data):
        """Test a full preprocessing pipeline."""
        X, y = missing_data
        # Add categorical features to missing data
        X['category'] = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        
                # Create preprocessors
        missing_processor = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        scaler = ScalingPreprocessor(method="standard")
        encoder = CategoricalEncodingPreprocessor(method="onehot")
        selector = FeatureSelectionPreprocessor(
            method="selectkbest",
            n_features_to_select=3,
            problem_type="classification"
        )
    
        # Apply pipeline
        missing_processor.fit(X, y)
        X_imputed = missing_processor.transform(X)
        scaler.fit(X_imputed, y, categorical_features=['categorical_feature', 'category'])
        X_scaled = scaler.transform(X_imputed)
        encoder.fit(X_scaled, y, categorical_features=['categorical_feature', 'category'])
        X_encoded = encoder.transform(X_scaled)
        selector.fit(X_encoded, y)
        X_selected = selector.transform(X_encoded)
        
        # Check final result
        assert X_selected.shape[1] == 3
        assert X_selected.isnull().sum().sum() == 0
    
 