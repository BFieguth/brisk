import datetime
from unittest import mock

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from brisk.services.metadata import MetadataService
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
from brisk.configuration.algorithm_collection import AlgorithmCollection


class MockModel:
    """Mock model with wrapper_name attribute for testing."""
    def __init__(self, wrapper_name: str):
        self.wrapper_name = wrapper_name


@pytest.fixture
def algorithm_config():
    """Create a mock algorithm configuration."""
    wrapper1 = AlgorithmWrapper(
        name="logistic_regression",
        display_name="Logistic Regression",
        algorithm_class=LogisticRegression,
        default_params={"max_iter": 1000}
    )
    wrapper2 = AlgorithmWrapper(
        name="random_forest",
        display_name="Random Forest Classifier",
        algorithm_class=RandomForestClassifier,
        default_params={"n_estimators": 100}
    )
    return AlgorithmCollection(wrapper1, wrapper2)


@pytest.fixture
def metadata_service(algorithm_config):
    metadata_service = MetadataService(
        name="metadata",
    )
    metadata_service.set_algorithm_config(algorithm_config)
    return metadata_service


class TestMetadataService:
    def test_init(self, metadata_service, algorithm_config):
        assert metadata_service.name == "metadata"
        assert metadata_service.algorithm_config == algorithm_config

    @mock.patch('datetime.datetime')
    def test_get_model_single_model(self, mock_datetime, metadata_service):
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        model = MockModel("logistic_regression")
        
        result = metadata_service.get_model(model, "test_method", is_test=False)
        
        expected = {
            "type": "model",
            "timestamp": "2025-01-01 12:00:00",
            "method": "test_method",
            "models": {
                "logistic_regression": "Logistic Regression"
            },
            "is_test": "False"
        }
        
        assert result == expected

    @mock.patch('datetime.datetime')
    def test_get_model_multiple_models(self, mock_datetime, metadata_service):
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        model1 = MockModel("logistic_regression")
        model2 = MockModel("random_forest")
        models = [model1, model2]
        
        result = metadata_service.get_model(models, "test_method", is_test=True)
        
        expected = {
            "type": "model",
            "timestamp": "2025-01-01 12:00:00",
            "method": "test_method",
            "models": {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest Classifier"
            },
            "is_test": "True"
        }
        
        assert result == expected

    @mock.patch('datetime.datetime')
    def test_get_model_test_flag(self, mock_datetime, metadata_service):
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        model = MockModel("logistic_regression")
        
        # Test with is_test=True
        result_test = metadata_service.get_model(model, "test_method", is_test=True)
        assert result_test["is_test"] == "True"
        
        # Test with is_test=False (default)
        result_train = metadata_service.get_model(model, "test_method")
        assert result_train["is_test"] == "False"

    @mock.patch('datetime.datetime')
    def test_get_dataset(self, mock_datetime, metadata_service):
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        result = metadata_service.get_dataset(
            method_name="test_method",
            dataset_name="iris",
            group_name="test_group"
        )
        
        expected = {
            "type": "dataset",
            "timestamp": "2025-01-01 12:00:00",
            "method": "test_method",
            "dataset": "iris",
            "group": "test_group"
        }
        
        assert result == expected

    def test_get_model_preserves_method_name(self, metadata_service):
        model = MockModel("logistic_regression")
        method_name = "evaluate_cross_validation"
        
        result = metadata_service.get_model(model, method_name)
        assert result["method"] == method_name

    def test_get_dataset_preserves_all_params(self, metadata_service):
        result = metadata_service.get_dataset(
            method_name="custom_method",
            dataset_name="custom_dataset",
            group_name="custom_group"
        )
        
        assert result["method"] == "custom_method"
        assert result["dataset"] == "custom_dataset"
        assert result["group"] == "custom_group"

    def test_get_model_empty_models_list(self, metadata_service):
        """Test behavior with empty models list."""
        models = []
        
        result = metadata_service.get_model(models, "test_method")
        
        assert result["models"] == {}
        assert result["type"] == "model"
        assert result["method"] == "test_method"

    def test_timestamp_format(self, metadata_service):
        """Test that timestamp follows expected format."""
        model = MockModel("logistic_regression")
        
        result = metadata_service.get_model(model, "test_method")
        
        timestamp = result["timestamp"]
        try:
            datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pytest.fail(f"Timestamp {timestamp} does not match expected format")

    def test_model_wrapper_lookup(self, metadata_service):
        """Test that algorithm wrapper is correctly looked up."""
        model = MockModel("random_forest")
        
        result = metadata_service.get_model(model, "test_method")
        
        assert result["models"]["random_forest"] == "Random Forest Classifier"

    def test_get_model_type_field(self, metadata_service):
        """Test that model metadata has correct type field."""
        model = MockModel("logistic_regression")
        result = metadata_service.get_model(model, "test_method")
        assert result["type"] == "model"

    def test_get_dataset_type_field(self, metadata_service):
        """Test that dataset metadata has correct type field."""
        result = metadata_service.get_dataset("method", "dataset", "group")
        assert result["type"] == "dataset"
