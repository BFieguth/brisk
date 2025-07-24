.. _default_regression_algorithms:

Default Regression Algorithms
==============================

Brisk provides a set of predefined regression algorithms wrapped as ``AlgorithmWrapper`` instances.
These algorithms are sourced from scikit-learn.

.. note::
   The default algorithms serve as a convenience for getting started with regression 
   tasks but are unlikely to be optimal for most projects. You should consider these 
   as a starting point you can use to get familiar with the framework. There are many
   other algorithms available in scikit-learn and many more hyperparameters you may 
   want to consider.

Regression Algorithms
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Algorithm
     - Internal Name
     - Hyperparameter Grid
   * - Linear Regression
     - linear
     - 
   * - Ridge Regression
     - ridge
     - | **alpha**: logarithmic space from 10^-3 to 10^0 (100 values)
   * - LASSO Regression
     - lasso
     - | **alpha**: logarithmic space from 10^-3 to 10^0 (100 values)
   * - Bayesian Ridge Regression
     - bridge
     - | **alpha_1**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
       | **alpha_2**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
       | **lambda_1**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
       | **lambda_2**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
   * - Elastic Net Regression
     - elasticnet
     - | **alpha**: logarithmic space from 10^-3 to 10^0 (100 values)
       | **l1_ratio**: 0.1 to 0.9 (step 0.1)
   * - Decision Tree Regression
     - dtr
     - | **criterion**: ["friedman_mse", "absolute_error", "poisson", "squared_error"]
       | **max_depth**: [5, 10, 15, 20, None]
   * - Random Forest
     - rf
     - | **n_estimators**: 20 to 140 (step 20)
       | **criterion**: ["friedman_mse", "absolute_error", "poisson", "squared_error"]
       | **max_depth**: [5, 10, 15, 20, None]
   * - Support Vector Regression
     - svr
     - | **kernel**: ["linear", "rbf", "sigmoid"]
       | **C**: 1 to 29.5 (step 0.5)
       | **gamma**: ["scale", "auto", 0.001, 0.01, 0.1]
   * - Multi-Layer Perceptron Regression
     - mlp
     - | **hidden_layer_sizes**: [(100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)]
       | **activation**: ["identity", "logistic", "tanh", "relu"]
       | **alpha**: [0.0001, 0.001, 0.01]
       | **learning_rate**: ["constant", "invscaling", "adaptive"]
   * - K-Nearest Neighbour Regression
     - knn
     - | **n_neighbors**: [1, 3]
       | **weights**: ["uniform", "distance"]
       | **algorithm**: ["auto", "ball_tree", "kd_tree", "brute"]
       | **leaf_size**: 5 to 45 (step 5)

Usage
-----

To use these algorithms in your Brisk project, you can import them directly:

.. code-block:: python

   from brisk import REGRESSION_ALGORITHMS
   
   # In your algorithms.py file
   ALGORITHM_CONFIG = brisk.AlgorithmCollection(
       *REGRESSION_ALGORITHMS
   )
   
   # Or select specific algorithms
   ALGORITHM_CONFIG = brisk.AlgorithmCollection(
       REGRESSION_ALGORITHMS[0],  # linear regression
       REGRESSION_ALGORITHMS[1],  # ridge regression
   )
