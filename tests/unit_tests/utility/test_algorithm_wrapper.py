import pytest
from unittest.mock import MagicMock
from brisk.configuration.algorithm_wrapper import AlgorithmWrapper

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
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock
            )
        algorithm_instance = wrapper.instantiate()

        algorithm_class_mock.assert_called_once_with()

        assert algorithm_instance == algorithm_class_mock()
        assert algorithm_instance.wrapper_name == "mock"

    def test_instantiate_with_params(self, algorithm_class_mock):
        """
        Test the instantiate method with default parameters.
        """
        default_params = {"param1": 10, "param2": "value"}
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock, default_params=default_params
            )
        algorithm_instance = wrapper.instantiate()

        algorithm_class_mock.assert_called_once_with(param1=10, param2="value")

        assert algorithm_instance == algorithm_class_mock()
        assert algorithm_instance.wrapper_name == "mock"

    def test_get_hyperparam_grid_empty(self, algorithm_class_mock):
        """
        Test get_hyperparam_grid method when no hyperparameter grid is provided.
        """
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock
            )
        hyperparam_grid = wrapper.get_hyperparam_grid()

        assert hyperparam_grid == {}

    def test_get_hyperparam_grid_with_params(self, algorithm_class_mock):
        """
        Test get_hyperparam_grid method with a hyperparameter grid provided.
        """
        hyperparam_grid = {"param1": [1, 2, 3], "param2": ["a", "b"]}
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock, hyperparam_grid=hyperparam_grid
            )
        returned_grid = wrapper.get_hyperparam_grid()

        assert returned_grid == hyperparam_grid

    def test_setitem_default_params(self, algorithm_class_mock):
        """
        Test __setitem__ method to update default_params.
        """
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock
            )
        wrapper["default_params"] = {"param1": 5, "param2": "test"}

        assert wrapper.default_params["param1"] == 5
        assert wrapper.default_params["param2"] == "test"

    def test_setitem_hyperparam_grid(self, algorithm_class_mock):
        """
        Test __setitem__ method to update hyperparam_grid.
        """
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock
            )
        wrapper["hyperparam_grid"] = {"param1": [1, 2, 3]}

        assert wrapper.hyperparam_grid["param1"] == [1, 2, 3]

    def test_setitem_invalid_key(self, algorithm_class_mock):
        """
        Test __setitem__ method raises KeyError for invalid keys.
        """
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock
            )

        with pytest.raises(KeyError, match="Invalid key: invalid_key"):
            wrapper["invalid_key"] = {"param1": 10}

    def test_instantiate_tuned_with_best_params(self, algorithm_class_mock):
        """
        Test instantiate_tuned method with specific tuned parameters.
        """
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock,
            default_params={"param1": 5, "param2": "default"}
            )
        best_params = {"param2": "tuned", "param3": "new_value"}
        algorithm_instance = wrapper.instantiate_tuned(best_params)

        algorithm_class_mock.assert_called_once_with(
            param2="tuned", param3="new_value"
            )
        assert algorithm_instance.wrapper_name == "mock"

    def test_instantiate_tuned_with_max_iter(self, algorithm_class_mock):
        """
        Test instantiate_tuned method ensures max_iter from default_params is 
        applied if present.
        """
        wrapper = AlgorithmWrapper(
            name="mock", display_name="MockModel",
            algorithm_class=algorithm_class_mock,
            default_params={"param1": 5, "max_iter": 100}
            )
        best_params = {"param2": "tuned"}
        algorithm_instance = wrapper.instantiate_tuned(best_params)

        algorithm_class_mock.assert_called_once_with(
            param2="tuned", max_iter=100
            )
        assert algorithm_instance.wrapper_name == "mock"
