import pytest
from unittest.mock import MagicMock
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

class TestAlgorithmWrapper:
    """Test class for AlgorithmWrapper."""

    @pytest.fixture
    def algorithm_class_mock(self):
        """Fixture to create a mock model class."""
        return MagicMock()

    def test_instantiate_with_default_params(self, algorithm_class_mock):
        """
        Test the instantiate method when no default parameters are provided.
        """
        wrapper = AlgorithmWrapper(name="MockModel", algorithm_class=algorithm_class_mock)
        algorithm_instance = wrapper.instantiate()

        algorithm_class_mock.assert_called_once_with()

        assert algorithm_instance == algorithm_class_mock()

    def test_instantiate_with_params(self, algorithm_class_mock):
        """
        Test the instantiate method with default parameters.
        """
        default_params = {"param1": 10, "param2": "value"}
        wrapper = AlgorithmWrapper(name="MockModel", algorithm_class=algorithm_class_mock, default_params=default_params)
        algorithm_instance = wrapper.instantiate()

        algorithm_class_mock.assert_called_once_with(param1=10, param2="value")

        assert algorithm_instance == algorithm_class_mock()

    def test_get_hyperparam_grid_empty(self, algorithm_class_mock):
        """
        Test get_hyperparam_grid method when no hyperparameter grid is provided.
        """
        wrapper = AlgorithmWrapper(name="MockModel", algorithm_class=algorithm_class_mock)
        hyperparam_grid = wrapper.get_hyperparam_grid()

        assert hyperparam_grid == {}

    def test_get_hyperparam_grid_with_params(self, algorithm_class_mock):
        """
        Test get_hyperparam_grid method with a hyperparameter grid provided.
        """
        hyperparam_grid = {"param1": [1, 2, 3], "param2": ["a", "b"]}
        wrapper = AlgorithmWrapper(name="MockModel", algorithm_class=algorithm_class_mock, hyperparam_grid=hyperparam_grid)
        returned_grid = wrapper.get_hyperparam_grid()

        assert returned_grid == hyperparam_grid
