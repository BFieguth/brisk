"""Unit tests for the AlgorithmCollection class."""

import numpy as np
import pytest
from sklearn import linear_model

from brisk import AlgorithmCollection, AlgorithmWrapper

class TestAlgorithmCollection:
    """Unit tests for the AlgorithmCollection class."""
    @pytest.fixture
    def algorithm_config(self):
        return AlgorithmCollection(
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression
            ),
            AlgorithmWrapper(
                name="ridge",
                display_name="Ridge Regression",
                algorithm_class=linear_model.Ridge,
                default_params={"max_iter": 10000},
                hyperparam_grid={"alpha": [0.1, 0.001]}
            ),
            AlgorithmWrapper(
                name="lasso",
                display_name="LASSO Regression",
                algorithm_class=linear_model.Lasso,
                default_params={"alpha": 0.1, "max_iter": 10000},
                hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
            ),
            AlgorithmWrapper(
                name="bridge",
                display_name="Bayesian Ridge Regression",
                algorithm_class=linear_model.BayesianRidge,
                default_params={"max_iter": 10000},
                hyperparam_grid={
                    "alpha_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    "alpha_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    "lambda_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    "lambda_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                }
            ),
            AlgorithmWrapper(
                name="elasticnet",
                display_name="Elastic Net Regression",
                algorithm_class=linear_model.ElasticNet,
                default_params={"alpha": 0.1, "max_iter": 10000},
                hyperparam_grid={
                    "alpha": [0.9, 0.5, 0.3, 0.1, 0.01, 0.001],
                    "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                }
            )
        )

    def test_length(self, algorithm_config):
        assert len(algorithm_config) == 5

    def test_str_index(self, algorithm_config):
        wrapper = algorithm_config["linear"]
        assert wrapper.name == "linear"
        assert wrapper.display_name == "Linear Regression"
        assert wrapper.algorithm_class == linear_model.LinearRegression

        wrapper2 = algorithm_config["bridge"]
        assert wrapper2.name == "bridge"
        assert wrapper2.display_name == "Bayesian Ridge Regression"
        assert wrapper2.algorithm_class == linear_model.BayesianRidge
        assert wrapper2.default_params == {"max_iter": 10000}
        assert wrapper2.hyperparam_grid == {
                "alpha_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                "alpha_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                "lambda_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                "lambda_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            }

    def test_missing_str_index(self, algorithm_config):
        with pytest.raises(
            KeyError, match="No algorithm found with name: "
        ):
            _ = algorithm_config["not_a_wrapper"]

    def test_int_index(self, algorithm_config):
        wrapper = algorithm_config[1]
        assert wrapper.name == "ridge"
        assert wrapper.display_name == "Ridge Regression"
        assert wrapper.algorithm_class == linear_model.Ridge
        assert wrapper.default_params == {"max_iter": 10000}
        assert wrapper.hyperparam_grid == {"alpha": [0.1, 0.001]}

        wrapper2 = algorithm_config[4]
        hyperparam_grid = {
            "alpha": [0.9, 0.5, 0.3, 0.1, 0.01, 0.001],
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        assert wrapper2.name == "elasticnet"
        assert wrapper2.display_name == "Elastic Net Regression"
        assert wrapper2.algorithm_class == linear_model.ElasticNet
        assert wrapper2.default_params == {"alpha": 0.1, "max_iter": 10000}
        for key, value in wrapper2.hyperparam_grid.items():
            assert value == hyperparam_grid[key]

    def test_missing_int_index(self, algorithm_config):
        with pytest.raises(
            IndexError, match="list index out of range"
        ):
            _ = algorithm_config[5]

    def test_invalid_index_type(self, algorithm_config):
        with pytest.raises(
            TypeError, match="Index must be an integer or string, got"
        ):
            _ =algorithm_config[1.2]

        with pytest.raises(
            TypeError, match="Index must be an integer or string, got"
        ):
            _ = algorithm_config[["algo_name"]]

    def test_append_invalid_type(self, algorithm_config):
        with pytest.raises(
            TypeError,
            match="AlgorithmCollection only accepts AlgorithmWrapper instances"
        ):
            algorithm_config.append(5)
            algorithm_config.append("linear")
            algorithm_config.append(linear_model.LinearRegression)

    def test_duplicate_name_error(self, algorithm_config):
        with pytest.raises(
            ValueError,
            match="Duplicate algorithm name: linear"
        ):
            algorithm_config.append(
                AlgorithmWrapper(
                    name="linear",
                    display_name="Linear Regression",
                    algorithm_class=linear_model.LinearRegression
                )
            )
