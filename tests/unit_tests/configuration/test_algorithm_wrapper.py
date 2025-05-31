"""Tests for the AlgorithmWrapper class."""

import pytest
from sklearn import linear_model, neural_network

from brisk.configuration.algorithm_wrapper import AlgorithmWrapper

class TestAlgorithmWrapper:
    """Test class for AlgorithmWrapper."""
    @pytest.fixture
    def linear_wrapper(self):
        """Fixture to create a linear regression wrapper."""
        return AlgorithmWrapper(
            name="linear",
            display_name="Linear Regression",
            algorithm_class=linear_model.LinearRegression
            )

    @pytest.fixture
    def ridge_wrapper(self):
        """Fixture to create a ridge regression wrapper."""
        return AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            default_params={"alpha": 0.1},
            hyperparam_grid={"alpha": [0.01, 0.1, 1.0]}
            )

    @pytest.fixture
    def mlp_wrapper(self):
        """Fixture to create a random forest wrapper."""
        return AlgorithmWrapper(
            name="mlp",
            display_name="MLP",
            algorithm_class=neural_network.MLPClassifier,
            default_params={"hidden_layer_sizes": (100, 50), "max_iter": 1000},
            hyperparam_grid={
                "hidden_layer_sizes": [(100, 50), (200, 100), (300, 150)],
                "activation": ["relu", "tanh", "logistic"]
                }
            )

    def test_init(self, linear_wrapper, ridge_wrapper):
        """Test the initialization of AlgorithmWrapper."""
        # No hyperparameter grid or default parameters
        assert linear_wrapper.name == "linear"
        assert linear_wrapper.display_name == "Linear Regression"
        assert linear_wrapper.algorithm_class == linear_model.LinearRegression
        assert linear_wrapper.default_params == {}
        assert linear_wrapper.hyperparam_grid == {}

        # With hyperparameter grid and default parameters
        assert ridge_wrapper.name == "ridge"
        assert ridge_wrapper.display_name == "Ridge Regression"
        assert ridge_wrapper.algorithm_class == linear_model.Ridge
        assert ridge_wrapper.default_params == {"alpha": 0.1}
        assert ridge_wrapper.hyperparam_grid == {"alpha": [0.01, 0.1, 1.0]}

    def test_init_incorrect(self):
        """
        Test the initialization of AlgorithmWrapper with incorrect parameters.
        """
        class MockClass:
            pass

        with pytest.raises(TypeError, match="name must be a string"):
            AlgorithmWrapper(
                name=123,
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
            )
        with pytest.raises(TypeError, match="display_name must be a string"):
            AlgorithmWrapper(
                name="linear",
                display_name=123,
                algorithm_class=linear_model.LinearRegression,
            )
        with pytest.raises(TypeError, match="algorithm_class must be a class"):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class="linear",
            )
        with pytest.raises(
            ValueError, match="algorithm_class must be from sklearn"
            ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=MockClass,
            )
        with pytest.raises(
            TypeError, match="default_params must be a dictionary"
            ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
                default_params="fit_intercept",
            )
        with pytest.raises(
            TypeError, match="hyperparam_grid must be a dictionary"
            ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
                hyperparam_grid="alpha: [0.01, 0.1, 1.0]",
            )

    def test_setitem_default_params(self, linear_wrapper):
        """
        Test __setitem__ method to update default_params.
        """
        assert linear_wrapper.default_params == {}
        linear_wrapper["default_params"] = {"param1": 5, "param2": "test"}
        assert linear_wrapper.default_params["param1"] == 5
        assert linear_wrapper.default_params["param2"] == "test"

    def test_setitem_hyperparam_grid(self, linear_wrapper):
        """
        Test __setitem__ method to update hyperparam_grid.
        """
        assert linear_wrapper.hyperparam_grid == {}
        linear_wrapper["hyperparam_grid"] = {"param1": [1, 2, 3]}
        assert linear_wrapper.hyperparam_grid["param1"] == [1, 2, 3]

    def test_setitem_invalid_key(self, linear_wrapper):
        """
        Test __setitem__ method raises KeyError for invalid keys.
        """
        with pytest.raises(KeyError, match="Invalid key: invalid_key"):
            linear_wrapper["invalid_key"] = {"param1": 10}

    def test_setitem_invalid_value(self, linear_wrapper):
        """
        Test __setitem__ method raises TypeError for invalid value types.
        """
        with pytest.raises(TypeError, match="value must be a dict"):
            linear_wrapper["default_params"] = True
        with pytest.raises(TypeError, match="value must be a dict"):
            linear_wrapper["hyperparam_grid"] = "param"

    def test_setitem_empty_dict(self, linear_wrapper):
        """
        Test __setitem__ method works with empty dictionaries.
        """
        assert linear_wrapper.default_params == {}
        linear_wrapper["default_params"] = {"param1": 1}
        assert linear_wrapper.default_params == {"param1": 1}
        linear_wrapper["default_params"] = {}
        assert linear_wrapper.default_params == {"param1": 1}

    def test_instantiate_no_default_params(self, linear_wrapper):
        """
        Test the instantiate method when no default parameters are provided.
        """
        algorithm_instance = linear_wrapper.instantiate()
        assert isinstance(algorithm_instance, linear_wrapper.algorithm_class)
        assert algorithm_instance.wrapper_name == "linear"

    def test_instantiate_default_params(self, ridge_wrapper):
        """
        Test the instantiate method with default parameters.
        """
        algorithm_instance = ridge_wrapper.instantiate()
        assert isinstance(algorithm_instance, ridge_wrapper.algorithm_class)
        assert algorithm_instance.wrapper_name == "ridge"
        assert algorithm_instance.alpha == 0.1

    def test_instantiate_tuned_with_best_params(self, ridge_wrapper):
        """
        Test instantiate_tuned method with specific tuned parameters.
        """
        best_params = {"alpha": 1, "fit_intercept": False}
        algorithm_instance = ridge_wrapper.instantiate_tuned(best_params)
        assert algorithm_instance.wrapper_name == "ridge"
        assert algorithm_instance.alpha == 1
        assert algorithm_instance.fit_intercept is False

    def test_instantiate_tuned_with_defaults(self, mlp_wrapper):
        """
        Test instantiate_tuned method ensures default parameters are applied
        """
        best_params = {
            "activation": "logistic", 
            "hidden_layer_sizes": (200, 100)
        }
        algorithm_instance = mlp_wrapper.instantiate_tuned(best_params)
        assert algorithm_instance.wrapper_name == "mlp"
        assert algorithm_instance.max_iter == 1000
        assert algorithm_instance.activation == "logistic"
        assert algorithm_instance.hidden_layer_sizes == (200, 100)

    def test_instantiate_tuned_invalid(self, ridge_wrapper):
        """
        Test instantiate_tuned method raises TypeError for invalid parameters.
        """
        best_params = [100, True]
        with pytest.raises(TypeError, match="best_params must be a dictionary"):
            ridge_wrapper.instantiate_tuned(best_params)

    def test_get_hyperparam_grid_empty(self, linear_wrapper):
        """
        Test get_hyperparam_grid method when no hyperparameter grid is provided.
        """
        hyperparam_grid = linear_wrapper.get_hyperparam_grid()
        assert hyperparam_grid == {}

    def test_get_hyperparam_grid_with_params(self, ridge_wrapper):
        """
        Test get_hyperparam_grid method with a hyperparameter grid provided.
        """
        hyperparam_grid = {"alpha": [0.01, 0.1, 1.0]}
        returned_grid = ridge_wrapper.get_hyperparam_grid()
        assert returned_grid == hyperparam_grid

    def test_to_markdown(self, ridge_wrapper, mlp_wrapper):
        """Test the markdown representation of algorithm configurations."""
        expected_ridge_md = (
            "### Ridge Regression (`ridge`)\n"
            "\n"
            "- **Algorithm Class**: `Ridge`\n"
            "\n"
            "**Default Parameters:**\n"
            "```python\n"
            "'alpha': 0.1,\n"
            "```\n"
            "\n"
            "**Hyperparameter Grid:**\n"
            "```python\n"
            "'alpha': [0.01, 0.1, 1.0],\n"
            "```"
        )
        assert ridge_wrapper.to_markdown() == expected_ridge_md

        expected_mlp_md = (
            "### MLP (`mlp`)\n"
            "\n"
            "- **Algorithm Class**: `MLPClassifier`\n"
            "\n"
            "**Default Parameters:**\n"
            "```python\n"
            "'hidden_layer_sizes': (100, 50),\n"
            "'max_iter': 1000,\n"
            "```\n"
            "\n"
            "**Hyperparameter Grid:**\n"
            "```python\n"
            "'hidden_layer_sizes': [(100, 50), (200, 100), (300, 150)],\n"
            "'activation': ['relu', 'tanh', 'logistic'],\n"
            "```"
        )
        assert mlp_wrapper.to_markdown() == expected_mlp_md
