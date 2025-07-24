.. _default_regression_metrics:

Default Regression Metrics
===========================

Brisk provides a set of predefined regression metrics wrapped as ``MetricWrapper`` instances. 
These metrics are sourced from scikit-learn and are ready to use in your projects without 
additional configuration.

Regression Metrics
------------------

Once imported you can select these metrics using internal name or abbreviation.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Metric
     - Internal Name
     - Abbreviation
   * - Explained Variance Score
     - explained_variance_score
     - 
   * - Max Error
     - max_error
     - 
   * - Mean Absolute Error
     - mean_absolute_error
     - MAE
   * - Mean Absolute Percentage Error
     - mean_absolute_percentage_error
     - MAPE
   * - Mean Pinball Loss
     - mean_pinball_loss
     - 
   * - Mean Squared Error
     - mean_squared_error
     - MSE
   * - Mean Squared Log Error
     - mean_squared_log_error
     - 
   * - Median Absolute Error
     - median_absolute_error
     - 
   * - R2 Score
     - r2_score
     - R2
   * - Root Mean Squared Error
     - root_mean_squared_error
     - RMSE
   * - Root Mean Squared Log Error
     - root_mean_squared_log_error
     - 
   * - Concordance Correlation Coefficient
     - concordance_correlation_coefficient
     - CCC
   * - Negative Mean Absolute Error
     - neg_mean_absolute_error
     - NegMAE
   * - Adjusted R2 Score
     - adjusted_r2_score
     - AdjR2

Usage
-----

To use these metrics in your Brisk project, you can import them directly:

.. code-block:: python

   from brisk import REGRESSION_METRICS
   
   # In your metrics.py file
   METRIC_CONFIG = brisk.MetricManager(
       *REGRESSION_METRICS
   )
   
   # Or select specific metrics
   METRIC_CONFIG = brisk.MetricManager(
       REGRESSION_METRICS[0],  # explained_variance_score
       REGRESSION_METRICS[2],  # mean_absolute_error
   )
