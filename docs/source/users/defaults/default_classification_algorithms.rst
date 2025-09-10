.. _default_classification_algorithms:

Default Classification Algorithms
==================================

Brisk provides a set of predefined classification algorithms wrapped as ``AlgorithmWrapper`` instances.
These algorithms are sourced from scikit-learn.

.. note::
   The default algorithms serve as a convenience for getting started with classification 
   tasks but are unlikely to be optimal for most projects. You should consider these 
   as a starting point you can use to get familiar with the framework. There are many
   other algorithms available in scikit-learn and many more hyperparameters you may 
   want to consider.

Classification Algorithms
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Algorithm
     - Internal Name
     - Hyperparameter Grid
   * - Logistic Regression
     - logistic
     - | **penalty**: [None, "l2", "l1", "elasticnet"]
       | **l1_ratio**: 0.1 to 0.9 (step 0.1)
       | **C**: 1 to 29.5 (step 0.5)
   * - Support Vector Classification
     - svc
     - | **kernel**: ["linear", "rbf", "sigmoid"]
       | **C**: 1 to 29.5 (step 0.5)
       | **gamma**: ["scale", "auto", 0.001, 0.01, 0.1]
   * - k-Nearest Neighbours Classifier
     - knn_classifier
     - | **n_neighbors**: [1, 3]
       | **weights**: ["uniform", "distance"]
       | **algorithm**: ["auto", "ball_tree", "kd_tree", "brute"]
       | **leaf_size**: 5 to 45 (step 5)
   * - Decision Tree Classifier
     - dtc
     - | **criterion**: ["gini", "entropy", "log_loss"]
       | **max_depth**: [5, 10, 15, 20, None]
   * - Random Forest Classifier
     - rf_classifier
     - | **n_estimators**: 20 to 140 (step 20)
       | **criterion**: ["friedman_mse", "absolute_error", "poisson", "squared_error"]
       | **max_depth**: [5, 10, 15, 20, None]
   * - Gaussian Naive Bayes
     - gaussian_nb
     - | **var_smoothing**: [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
   * - Ridge Classifier
     - ridge_classifier
     - | **alpha**: logarithmic space from 10^-3 to 10^0 (100 values)
   * - Bagging Classifier
     - bagging_classifier
     - | **n_estimators**: 10 to 150 (step 20)
   * - Voting Classifier
     - voting_classifier
     - | **voting**: ["hard", "soft"]

Usage
-----

To use these algorithms in your Brisk project, you can import them directly:

.. code-block:: python

   from brisk import CLASSIFICATION_ALGORITHMS
   
   # In your algorithms.py file
   ALGORITHM_CONFIG = brisk.AlgorithmCollection(
       *CLASSIFICATION_ALGORITHMS
   )
   
   # Or select specific algorithms
   ALGORITHM_CONFIG = brisk.AlgorithmCollection(
       CLASSIFICATION_ALGORITHMS[0],  # logistic regression
       CLASSIFICATION_ALGORITHMS[3],  # decision tree
   )
