"""Default configuration for regression algorithms.

This module provides configuration settings for different regression algorithms. 
Each algorithm is wrapped in a `AlgorithmWrapper` which includes the algorithms's 
display_name, its class, default parameters, and hyperparameter space for optimization.

"""

from typing import List

import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.linear_model as linear
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.neural_network as neural
import sklearn.kernel_ridge as kernel_ridge

from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

REGRESSION_ALGORITHMS: List[AlgorithmWrapper] = [
    AlgorithmWrapper(
        name="linear",
        display_name="Linear Regression",
        algorithm_class=linear.LinearRegression
    ),
    AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=linear.Ridge,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    AlgorithmWrapper(
        name="lasso",
        display_name="LASSO Regression",
        algorithm_class=linear.Lasso,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    AlgorithmWrapper(
        name="bridge",
        display_name="Bayesian Ridge Regression",
        algorithm_class=linear.BayesianRidge,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],    #TODO Change these?
            'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   
            'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  
            'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  
        }
    ),
    AlgorithmWrapper(
        name="elasticnet",
        display_name="Elastic Net Regression",
        algorithm_class=linear.ElasticNet,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={
            "alpha": np.logspace(-3, 0, 100),
            "l1_ratio": list(np.arange(0.1, 1, 0.1))
        }
    ),
    AlgorithmWrapper(
        name="dtr",
        display_name="Decision Tree Regression",
        algorithm_class=tree.DecisionTreeRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }
    ),
    AlgorithmWrapper(
        name="rf",
        display_name="Random Forest",
        algorithm_class=ensemble.RandomForestRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'n_estimators': list(range(20, 160, 20)),   # TODO add min_samples_split?
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }
    ),
    AlgorithmWrapper(
        name="gbr",
        display_name="Gradient Boosting Regression",
        algorithm_class=ensemble.GradientBoostingRegressor,
        hyperparam_grid={
            'loss': ['squared_error', 'absolute_error', 'huber'],
            'learning_rate': list(np.arange(0.01, 1, 0.1)),
            'n_estimators': list(range(50, 200, 10)),   
            # 'alpha': list(np.arange(0.1, 1, 0.1)) # Range [0, 1], only use if 'huber' is selected
        } 
    ),
    AlgorithmWrapper(
        name="adaboost",
        display_name="AdaBoost Regression",
        algorithm_class=ensemble.AdaBoostRegressor,
        hyperparam_grid={
            'n_estimators': list(range(50, 200, 10)),  
            'learning_rate': list(np.arange(0.01, 3, 0.1)), 
            'loss': ['linear', 'square', 'exponential'] 
        } 
    ),
    AlgorithmWrapper(
        name="svr",
        display_name="Support Vector Regression",
        algorithm_class=svm.SVR,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'C': list(np.arange(1, 30, 0.5)), 
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    ),
    AlgorithmWrapper(
        name="mlp",
        display_name="Multi-Layer Perceptron Regression",
        algorithm_class=neural.MLPRegressor,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            'hidden_layer_sizes': [
                (100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)
                ], 
            'activation': ['identity', 'logistic', 'tanh', 'relu'],    
            'alpha': [0.0001, 0.001, 0.01],     # TODO surely this could be better                    
            'learning_rate': ['constant', 'invscaling', 'adaptive']   
        }
    ),
    AlgorithmWrapper(
        name="knn",
        display_name="K-Nearest Neighbour Regression",
        algorithm_class=neighbors.KNeighborsRegressor,
        hyperparam_grid={
            'n_neighbors': list(range(1,5,2)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': list(range(5, 50, 5))
        } 
    ),
    AlgorithmWrapper(
        name="lars",
        display_name="Least Angle Regression",
        algorithm_class=linear.Lars
    ),
    AlgorithmWrapper(
        name="omp",
        display_name="Orthogonal Matching Pursuit",
        algorithm_class=linear.OrthogonalMatchingPursuit
    ),
    AlgorithmWrapper(
        name="ard",
        display_name="Bayesian ARD Regression",
        algorithm_class=linear.ARDRegression,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],    # TODO same as bayesian regression
            'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   
            'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        }
    ),
    AlgorithmWrapper(
        name="passagg",
        display_name="Passive Aggressive Regressor",
        algorithm_class=linear.PassiveAggressiveRegressor,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'C': list(np.arange(1, 100, 1)) # TODO fine tune this?
        }
    ),
    AlgorithmWrapper(
        name="kridge",
        display_name="Kernel Ridge",
        algorithm_class=kernel_ridge.KernelRidge,
        hyperparam_grid={
            'alpha': np.logspace(-3, 0, 100)
        }
    ),
    AlgorithmWrapper(
        name="nusvr",
        display_name="Nu Support Vector Regression",
        algorithm_class=svm.NuSVR,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'C': list(np.arange(1, 30, 0.5)),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    ),
    AlgorithmWrapper(
        name="rnn",
        display_name="Radius Nearest Neighbour",
        algorithm_class=neighbors.RadiusNeighborsRegressor,
        hyperparam_grid={
            'radius': [i * 0.5 for i in range(1, 7)],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': list(range(10, 60, 10))
        }
    ),
    AlgorithmWrapper(
        name="xtree",
        display_name="Extra Tree Regressor",
        algorithm_class=ensemble.ExtraTreesRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'n_estimators': list(range(20, 160, 20)),
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }   # TODO add min_sample_split?
    )
]
