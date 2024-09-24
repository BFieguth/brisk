import pytest
from unittest.mock import MagicMock
from ml_toolkit.utility.ModelWrapper import ModelWrapper

class TestModelWrapper:
    """Test class for ModelWrapper."""

    @pytest.fixture
    def model_class_mock(self):
        """Fixture to create a mock model class."""
        return MagicMock()

    def test_instantiate_with_default_params(self, model_class_mock):
        """
        Test the instantiate method when no default parameters are provided.
        """
        wrapper = ModelWrapper(name="MockModel", model_class=model_class_mock)
        model_instance = wrapper.instantiate()

        model_class_mock.assert_called_once_with()

        assert model_instance == model_class_mock()

    def test_instantiate_with_params(self, model_class_mock):
        """
        Test the instantiate method with default parameters.
        """
        default_params = {"param1": 10, "param2": "value"}
        wrapper = ModelWrapper(name="MockModel", model_class=model_class_mock, default_params=default_params)
        model_instance = wrapper.instantiate()

        model_class_mock.assert_called_once_with(param1=10, param2="value")

        assert model_instance == model_class_mock()

    def test_get_hyperparam_grid_empty(self, model_class_mock):
        """
        Test get_hyperparam_grid method when no hyperparameter grid is provided.
        """
        wrapper = ModelWrapper(name="MockModel", model_class=model_class_mock)
        hyperparam_grid = wrapper.get_hyperparam_grid()

        assert hyperparam_grid == {}

    def test_get_hyperparam_grid_with_params(self, model_class_mock):
        """
        Test get_hyperparam_grid method with a hyperparameter grid provided.
        """
        hyperparam_grid = {"param1": [1, 2, 3], "param2": ["a", "b"]}
        wrapper = ModelWrapper(name="MockModel", model_class=model_class_mock, hyperparam_grid=hyperparam_grid)
        returned_grid = wrapper.get_hyperparam_grid()

        assert returned_grid == hyperparam_grid
