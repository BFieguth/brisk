"""Default configuration for classification algorithms.

This module provides configuration settings for different classification 
algorithms. Each algorithm is wrapped in a `AlgorithmWrapper` which includes the
algorithms's display_name, its class, default parameters, and hyperparameter 
space for optimization.
"""
from typing import List

import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network

from brisk.configuration import algorithm_wrapper

CLASSIFICATION_ALGORITHMS: List[algorithm_wrapper.AlgorithmWrapper] = [
    algorithm_wrapper.AlgorithmWrapper(
        name="logistic",
        display_name="Logistic Regression",
        algorithm_class=linear_model.LogisticRegression,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "penalty": [None, "l2", "l1", "elasticnet"],
            "l1_ratio": list(np.arange(0.1, 1, 0.1)),
            "C": list(np.arange(1, 30, 0.5)),
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="svc",
        display_name="Support Vector Classification",
        algorithm_class=svm.SVC,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "kernel": ["linear", "rbf", "sigmoid"],
            "C": list(np.arange(1, 30, 0.5)), 
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="knn_classifier",
        display_name="k-Nearest Neighbours Classifier",
        algorithm_class=neighbors.KNeighborsClassifier,
        hyperparam_grid={
            "n_neighbors": list(range(1,5,2)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(5, 50, 5)),
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="dtc",
        display_name="Decision Tree Classifier",
        algorithm_class=tree.DecisionTreeClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": list(range(5, 25, 5)) + [None],           
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="rf_classifier",
        display_name="Random Forest Classifier",
        algorithm_class=ensemble.RandomForestClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "n_estimators": list(range(20, 160, 20)),
            "criterion": ["friedman_mse", "absolute_error", 
                          "poisson", "squared_error"],
            "max_depth": list(range(5, 25, 5)) + [None],
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        algorithm_class=naive_bayes.GaussianNB,
        hyperparam_grid={
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="ridge_classifier",
        display_name="Ridge Classifier",
        algorithm_class=linear_model.RidgeClassifier,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="bagging_classifier",
        display_name="Bagging Classifier",
        algorithm_class=ensemble.BaggingClassifier,
        hyperparam_grid={
            "n_estimators": list(range(10, 160, 20)),
        }
    ),
    algorithm_wrapper.AlgorithmWrapper(
        name="voting_classifier",
        display_name="Voting Classifier",
        algorithm_class=ensemble.VotingClassifier,
        hyperparam_grid={
            "voting": ["hard", "soft"],
        }
    )
]
