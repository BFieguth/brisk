"""Default configuration for regression algorithms.

This module provides configuration settings for different regression algorithms. 
Each algorithm is wrapped in a `AlgorithmWrapper` which includes the 
algorithms"s display_name, its class, default parameters, and hyperparameter 
space for optimization.
"""

from typing import List

import numpy as np
from sklearn import ensemble
from sklearn import kernel_ridge
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree

from brisk.utility import algorithm_wrapper

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
        name="gbr",
        display_name="Gradient Boosting Regression",
        algorithm_class=ensemble.GradientBoostingRegressor,
        hyperparam_grid={
            "loss": ["squared_error", "absolute_error", "huber"],
            "learning_rate": list(np.arange(0.01, 1, 0.1)),
            "n_estimators": list(range(50, 200, 10)),
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="adaboost",
        display_name="AdaBoost Regression",
        algorithm_class=ensemble.AdaBoostRegressor,
        hyperparam_grid={
            "n_estimators": list(range(50, 200, 10)),  
            "learning_rate": list(np.arange(0.01, 3, 0.1)), 
            "loss": ["linear", "square", "exponential"] 
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
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="lars",
        display_name="Least Angle Regression",
        algorithm_class=linear_model.Lars
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="omp",
        display_name="Orthogonal Matching Pursuit",
        algorithm_class=linear_model.OrthogonalMatchingPursuit
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="ard",
        display_name="Bayesian ARD Regression",
        algorithm_class=linear_model.ARDRegression,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "alpha_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "alpha_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   
            "lambda_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "lambda_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="passagg",
        display_name="Passive Aggressive Regressor",
        algorithm_class=linear_model.PassiveAggressiveRegressor,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "C": list(np.arange(1, 100, 1))
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="kridge",
        display_name="Kernel Ridge",
        algorithm_class=kernel_ridge.KernelRidge,
        hyperparam_grid={
            "alpha": np.logspace(-3, 0, 100)
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="nusvr",
        display_name="Nu Support Vector Regression",
        algorithm_class=svm.NuSVR,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            "kernel": ["linear", "rbf", "sigmoid"],
            "C": list(np.arange(1, 30, 0.5)),
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="rnn",
        display_name="Radius Nearest Neighbour",
        algorithm_class=neighbors.RadiusNeighborsRegressor,
        hyperparam_grid={
            "radius": [i * 0.5 for i in range(1, 7)],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(10, 60, 10))
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="xtree",
        display_name="Extra Tree Regressor",
        algorithm_class=ensemble.ExtraTreesRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "n_estimators": list(range(20, 160, 20)),
            "criterion": ["friedman_mse", "absolute_error", 
                          "poisson", "squared_error"],
            "max_depth": list(range(5, 25, 5)) + [None]
        }
    )
]
