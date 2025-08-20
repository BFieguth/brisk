"""Default configuration for regression algorithms.

This module provides configuration settings for different regression algorithms.
Each algorithm is wrapped in a `AlgorithmWrapper` which includes the
algorithms"s display_name, its class, default parameters, and hyperparameter
space for optimization.
"""

from typing import List

import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree

from brisk.configuration import algorithm_wrapper

REGRESSION_ALGORITHMS: List[algorithm_wrapper.AlgorithmWrapper] = [
    algorithm_wrapper.AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=linear_model.LinearRegression
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=linear_model.Ridge,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="lasso",
        display_name="LASSO Regression",
        algorithm_class=linear_model.Lasso,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    algorithm_wrapper.AlgorithmWrapper(
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
    algorithm_wrapper.AlgorithmWrapper(
        name="elasticnet",
        display_name="Elastic Net Regression",
        algorithm_class=linear_model.ElasticNet,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={
            "alpha": np.logspace(-3, 0, 100),
            "l1_ratio": list(np.arange(0.1, 1, 0.1))
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="dtr",
        display_name="Decision Tree Regression",
        algorithm_class=tree.DecisionTreeRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "criterion": ["friedman_mse", "absolute_error",
                          "poisson", "squared_error"],
            "max_depth": list(range(5, 25, 5)) + [None]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="rf",
        display_name="Random Forest",
        algorithm_class=ensemble.RandomForestRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "n_estimators": list(range(20, 160, 20)),
            "criterion": ["friedman_mse", "absolute_error",
                          "poisson", "squared_error"],
            "max_depth": list(range(5, 25, 5)) + [None]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="svr",
        display_name="Support Vector Regression",
        algorithm_class=svm.SVR,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "kernel": ["linear", "rbf", "sigmoid"],
            "C": list(np.arange(1, 30, 0.5)),
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="mlp",
        display_name="Multi-Layer Perceptron Regression",
        algorithm_class=neural_network.MLPRegressor,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            "hidden_layer_sizes": [
                (100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)
                ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="knn",
        display_name="K-Nearest Neighbour Regression",
        algorithm_class=neighbors.KNeighborsRegressor,
        hyperparam_grid={
            "n_neighbors": list(range(1,5,2)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(5, 50, 5))
        }
    )
]
