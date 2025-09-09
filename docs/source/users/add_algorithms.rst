.. _add_algorithms:

Add Algorithms
=================

Just like the MetricManager, we need to leave the AlgorithmCollection class as is.  
You define the algorithms you want to use by adding AlgorithmWrappers to the 
AlgorithmCollection. We are going to add Linear Regression, Lasso Regression, and 
Ridge Regression:

.. code-block:: python

    import brisk
    import numpy as np
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
            hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
        ),
        brisk.AlgorithmWrapper(
            name="lasso",
            display_name="LASSO Regression",
            algorithm_class=linear_model.Lasso,
            hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
        ),
    )

Hopefully this structure is familiar from defining the metrics. You may have noticed
that we added a ``hyperparam_grid`` argument to the Ridge and Lasso Regression 
wrappers. This is used to define the hyperparameter space for the algorithm. 
Brisk will use this to perform hyperparameter tuning.
