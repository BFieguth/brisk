"""Default configuration for classification algorithms.

This module provides configuration settings for different classification algorithms. 
Each algorithm is wrapped in a `AlgorithmWrapper` which includes the algorithms's 
display_name, its class, default parameters, and hyperparameter space for optimization.

"""
from typing import List

import numpy as np
import sklearn.linear_model as linear
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.naive_bayes as nb
import sklearn.neighbors as neighbors
import sklearn.neural_network as neural

from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

CLASSIFICATION_ALGORITHMS: List[AlgorithmWrapper] = [
    AlgorithmWrapper(
        name="logistic",
        display_name="Logistic Regression",
        algorithm_class=linear.LogisticRegression,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "penalty": [None, "l2", "l1", "elasticnet"],
            "l1_ratio": list(np.arange(0.1, 1, 0.1)),
            'C': list(np.arange(1, 30, 0.5)),
            }
    ),
    AlgorithmWrapper(
        name="svc",
        display_name="Support Vector Classification",
        algorithm_class=svm.SVC,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'C': list(np.arange(1, 30, 0.5)), 
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        }
    ),
    AlgorithmWrapper(
        name="linear_svc",
        display_name="Linear Support Vector Classification",
        algorithm_class=svm.LinearSVC,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'C': list(np.arange(1, 30, 0.5)), 
            "penalty": ["l1", "l2"],
        }
    ),
    AlgorithmWrapper(
        name="knn_classifier",
        display_name="k-Nearest Neighbours Classifier",
        algorithm_class=neighbors.KNeighborsClassifier,
        hyperparam_grid={
            'n_neighbors': list(range(1,5,2)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': list(range(5, 50, 5)),
        }
    ),   
    AlgorithmWrapper(
        name="dtc",
        display_name="Decision Tree Classifier",
        algorithm_class=tree.DecisionTreeClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": list(range(5, 25, 5)) + [None],           
        }
    ),
        AlgorithmWrapper(
        name="rf_classifier",
        display_name="Random Forest Classifier",
        algorithm_class=ensemble.RandomForestClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'n_estimators': list(range(20, 160, 20)),
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None],
        }
    ),
    AlgorithmWrapper(
        name="gbm_classifier",
        display_name="Gradient Boosting Machine Classifier",
        algorithm_class=ensemble.GradientBoostingClassifier,
        hyperparam_grid={
            'loss': ['squared_error', 'absolute_error', 'huber'],
            'learning_rate': list(np.arange(0.01, 1, 0.1)),
            'n_estimators': list(range(50, 200, 10)),   
        } 
    ),
    AlgorithmWrapper(
        name="adaboost_classifier",
        display_name="Adaboost Classifier",
        algorithm_class=ensemble.AdaBoostClassifier,
        hyperparam_grid={
            'n_estimators': list(range(50, 200, 10)),  
            'learning_rate': list(np.arange(0.01, 3, 0.1)), 
        }
    ),
    AlgorithmWrapper(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        algorithm_class=nb.GaussianNB,
        hyperparam_grid={
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        }
    ),
    AlgorithmWrapper(
        name="mlp_classifier",
        display_name="Multi-Layer Perceptron Classification",
        algorithm_class=neural.MLPClassifier,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            'hidden_layer_sizes': [
                (100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)
                ], 
            'activation': ['identity', 'logistic', 'tanh', 'relu'],    
            'alpha': [0.0001, 0.001, 0.01],                   
            'learning_rate': ['constant', 'invscaling', 'adaptive']   
        }
    ),
    AlgorithmWrapper(
        name="ridge_classifier",
        display_name="Ridge Classifier",
        algorithm_class=linear.RidgeClassifier,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    AlgorithmWrapper(
        name="bagging_classifier",
        display_name="Bagging Classifier",
        algorithm_class=ensemble.BaggingClassifier,
        hyperparam_grid={
            'n_estimators': list(range(10, 160, 20)),
        }
    ),
    AlgorithmWrapper(
        name="xtree_classifier",
        display_name="Extra Tree Classifier",
        algorithm_class=ensemble.ExtraTreesClassifier,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'n_estimators': list(range(20, 160, 20)),
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }
    ),
    AlgorithmWrapper(
        name=   "voting_classifier",
        display_name="Voting Classifier",
        algorithm_class=ensemble.VotingClassifier,
        hyperparam_grid={
            "voting": ["hard", "soft"],
        }
    )
]
