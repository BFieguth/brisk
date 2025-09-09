.. _custom_metrics:

Creating Custom Metrics
=======================

If there is an evaluation metric you want to use that is not provided by Brisk, 
you can add it to your project by adding a ``MetricWrapper`` in ``metrics.py``.

.. note::
   For the ``MetricWrapper`` documentation, see the :doc:`MetricWrapper </api/evaluation/metric_wrapper>` API reference.


Defining a Metric Function
--------------------------
The first step is to define a function that calculates the metric. This functions
must take two arguments, ``y_true`` and ``y_pred`` (in this order). These are array-like
objects of the same length with the actual and predicted values, respectively. Your metric
function should return a float. You can define these functions in the ``metrics.py`` file
before you create the ``MetricManager``. You may also import functions that implement an
evaluation metric from scikit-learn or other libraries, given it follows the expected interface.

Here is an example of a metric function that calculates the concordance correlation coefficient:

.. code-block:: python
    
    import numpy as np
    import scipy

    def concordance_correlation_coefficient(y_true, y_pred):
        corr, _ = scipy.stats.pearsonr(y_true, y_pred)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)
        numerator = 2 * corr * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        return numerator / denominator

There may be additional arguments that are required by your metric function and these
can be defined after the ``y_true`` and ``y_pred`` arguments. 


Split Metadata
--------------
You may encounter metric functions that require additional information about the data. 
For example, the adjusted R\ :sup:`2` score requires the number of features in the dataset. 
Brisk provides a ``split_metadata`` argument that is passed to your metric function.
This dictionary contains a couple of values for the data split being used.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Key
     - Value
   * - num_features
     - The number of features in the dataset
   * - num_samples
     - The number of samples in the dataset

To use this metadata in your metric function, you can access the ``split_metadata``
dictionary in your function.

.. code-block:: python

    from sklearn.metrics import _regression

    def adjusted_r2_score(y_true, y_pred, split_metadata):
        r2 = _regression.r2_score(y_true, y_pred)
        adjusted_r2 = (1 - (1 - r2) * (len(y_true) - 1) /
                    (len(y_true) - split_metadata["num_features"] - 1))
        return adjusted_r2

If you do not need to use the split metadata, you can ignore it in your function definition.


Multiclass Metrics
------------------
Some metrics are defined for the binary classification problem and only consider the positive label.
When extending a binary metric to a multiclass problem, the problem is treated as a series of
binary classification problems for each class. These values need to be averaged to get a single
value for the metric for all classes. For sklearn metrics, this is done by passing the ``average``
parameter as an argument to the ``MetricWrapper``.

More details can be found in the `scikit-learn documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#average>`_.


Create a MetricWrapper
----------------------
Once the metric function is defined we simply need to create a ``MetricWrapper`` 
and add it to the ``MetricManager`` in ``metrics.py``. Using the concordance correlation coefficient metric
function from the previous section, we can create the following ``MetricWrapper``:

.. code-block:: python

    METRIC_CONFIG = brisk.MetricManager(
        brisk.MetricWrapper(
            name="concordance_correlation_coefficient",
            func=concordance_correlation_coefficient,
            display_name="Concordance Correlation Coefficient",
            abbr="CCC"
        )
    )

The ``name`` and ``abbr`` atttributes must be unique as they are used to identify the metric.
You can name them whatever makes sense for your project. The ``display_name`` attribute is
used in plots and tables to identify the metric.

By adding the ``MetricWrapper`` to the ``MetricManager``, the metric will be available for use
in your workflows.

