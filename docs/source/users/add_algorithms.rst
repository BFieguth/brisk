.. _add_algorithms:

Add Algorithms
=================

To use a supervised learning algorithm in Brisk you need to add it to the ``AlgorithmCollection`` in 
``algorithms.py``. You can do this by adding an :ref:`AlgorithmWrapper <algorithm_wrapper>` to the 
``AlgorithmCollection``.

.. note::
    Brisk expects the ``AlgorithmCollection`` to be named ``ALGORITHM_CONFIG``.

In this example we will add wrappers for a Linear Regression and Ridge Regression algorithm:

.. code-block:: python

    import brisk
    from sklearn import linear_model

    ALGORITHM_CONFIG = brisk.AlgorithmCollection(
        brisk.AlgorithmWrapper(
            name="linear",
            display_name="Linear Regression",
            algorithm_class=linear_model.LinearRegression
        ),
        brisk.AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            default_params={"max_iter": 10000},
            hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
        )
    )

The first wrapper is for the Linear Regression algorithm and shows the minimum
required arguments:

- ``name`` is the value you use to access this algorithm in ``settings.py``.
- ``display_name`` will be used in plots, reports and other outputs.
- ``algorithm_class`` is the scikit-learn algorithm class implementing the algorithm

The second wrapper for Ridge Regression includes two additional arguments:

- ``default_params`` these values will be used to instantiate the algorithm.
- ``hyperparam_grid`` these values will be used for hyperparameter tuning.


Custom Algorithm Implementations
--------------------------------
Brisk is designed to work with scikit-learn ``BaseEstimator`` subclasses as many of
Brisk's builtin methods rely on the scikit-learn Estimator API. If you want to implement your
own algorithm you can follow `scikit-learn's documentation <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_
to see how to implement a class compatible with the scikit-learn API.

Any method that passes the ``check_estimator`` function from scikit-learn should
work with Brisk.
