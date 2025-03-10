Default Classification Metrics
==============================

Brisk provides a set of predefined classification metrics wrapped as ``MetricWrapper`` instances. 
These metrics are sourced from scikit-learn and are ready to use in your projects without 
additional configuration.

Classification Metrics
----------------------

Once imported you can select these metrics using internal name or abbreviation.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Metric
     - Internal Name
     - Abbreviation
   * - Accuracy
     - accuracy
     - 
   * - Precision
     - precision
     - 
   * - Recall
     - recall
     - 
   * - F1 Score
     - f1_score
     - f1
   * - Balanced Accuracy
     - balanced_accuracy
     - bal_acc
   * - Top-k Accuracy Score
     - top_k_accuracy
     - top_k
   * - Log Loss
     - log_loss
     - 
   * - Area Under the ROC Curve
     - roc_auc
     - 
   * - Brier Score Loss
     - brier
     - 
   * - Receiver Operating Characteristic
     - roc
     - 

Usage
-----

To use these metrics in your Brisk project, you can import them directly:

.. code-block:: python

   from brisk import CLASSIFICATION_METRICS
   
   # In your metrics.py file
   METRIC_CONFIG = brisk.MetricManager(
       *CLASSIFICATION_METRICS
   )
   
   # Or select specific metrics
   METRIC_CONFIG = brisk.MetricManager(
       CLASSIFICATION_METRICS[0],  # accuracy
       CLASSIFICATION_METRICS[2],  # recall
   )