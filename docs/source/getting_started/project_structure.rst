.. _project_structure:  

Project Structure
=================

Brisk organizes projects into multiple files to promote modularity, maintainability, 
and separation of concerns. This approach might seem more complex initially, but 
it provides many benefits as your machine learning projects grow.

Why use multiple files?
-----------------------

When working on machine learning projects, it's easy to end up with large, unwieldy 
scripts that mix together data loading, preprocessing, model configuration, training logic, 
and evaluation code. This approach quickly becomes difficult to maintain and understand.

Brisk takes a different approach by separating your project into logical components:

* Configuration files handle setup of data processing, algorithms, and metrics

* Workflow files define a training and evaluation process

* The ``settings.py`` file specifies what experiments to run

This separation offers several key advantages:

* Better organization: Each file has a clear, focused purpose
* Improved reusability: Components can be reused across projects
* Reduced complexity: You can focus on one component at a time

Core Project Files
------------------

A Brisk project contains the following files:

.briskconfig
~~~~~~~~~~~~~

A simple configuration file that stores basic project information. It helps Brisk 
locate your project root directory.


datasets/
~~~~~~~~~

Directory where all your datasets should be stored. Brisk expects to find your data here.


algorithms.py
~~~~~~~~~~~~~

Defines the machine learning algorithms you want to use in your project. This configuration 
file uses ``AlgorithmWrapper`` objects to define each algorithm along with its hyperparameters.

.. code-block:: python

    import brisk
    from sklearn import linear_model

    ALGORITHM_CONFIG = brisk.AlgorithmCollection(
        brisk.AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            hyperparam_grid={"alpha": [0.1, 1.0, 10.0]}
        )
    )


metrics.py
~~~~~~~~~~

Specifies the evaluation metrics you want to use to measure the performance of your models. 
Like algorithms, metrics are defined using ``MetricWrapper`` objects that provide a 
consistent interface.

.. code-block:: python

    import brisk
    from sklearn import metrics

    METRIC_CONFIG = brisk.MetricManager(
        brisk.MetricWrapper(
            name="mean_absolute_error",
            func=metrics.mean_absolute_error,
            display_name="Mean Absolute Error",
            abbr="MAE"
        )
    )

Brisk provides a set of predefined wrappers that include many common metrics:

.. code-block:: python

    import brisk

    METRIC_CONFIG = brisk.MetricManager(
        *brisk.REGRESSION_METRICS,  # For regression tasks
        *brisk.CLASSIFICATION_METRICS  # For classification tasks
    )

data.py
~~~~~~~

Configures how your data will be processed and split. This file defines a 
``DataManager`` that handles train-test splitting and preprocessing pipelines. 
This is the default processing used for all datasets. You can override this 
default for individual experiment groups in ``settings.py``.

.. code-block:: python

    from brisk.data.data_manager import DataManager

    BASE_DATA_MANAGER = DataManager(
        test_size=0.2,
        split_method="shuffle"
    )

settings.py
~~~~~~~~~~~

This is where you define what experiments you want to run. Unlike the previous 
configuration files, you'll frequently modify this file to try different 
combinations of datasets and algorithms.

For example if you wanted to compare the performance of different scaling methods,
you could define two experiment groups:

.. code-block:: python

    from brisk.configuration.configuration import Configuration
    from brisk.configuration.configuration_manager import ConfigurationManager
    from brisk.data.preprocessing import ScalingPreprocessor

    def create_configuration() -> ConfigurationManager:
        config = Configuration(
            default_workflow="my_workflow",
            default_algorithms=["linear", "ridge"],
            categorical_features={"data.csv": ["category_col", "region"]}  
        )

        config.add_experiment_group(
            name="minmax_scaled",
            description="Training models with minmax scaling",
            datasets=["diabetes.csv"],
            data_config={
                "test_size": 0.25, 
                "preprocessors": [ScalingPreprocessor(method="minmax")]
            }
        )

        config.add_experiment_group(
            name="standard_scaled",
            description="Training models with standard scaling",
            datasets=["diabetes.csv"],
            data_config={
                "test_size": 0.25,
                "preprocessors": [ScalingPreprocessor(method="standard")]
            }
        )

        return config.build()

The ``categorical_features`` parameter maps dataset filenames to lists of categorical column names. 
Brisk automatically detects categorical features, but you can explicitly specify them for better 
control over preprocessing. Categorical features are handled differently during preprocessing - 
they're excluded from scaling operations and processed by categorical encoders.

.. note::
   Consider specifying categorical features explicitly if auto-detection doesn't work 
   well for your data. This is especially useful for numeric categorical features 
   (like zip codes or customer IDs) that might be misclassified as continuous.


workflows/
~~~~~~~~~~~

Contains Python files that define the actual training and evaluation process using 
Brisk's ``Workflow`` class. Each workflow file should contain exactly one workflow subclass:

.. code-block:: python

    # workflows/my_workflow.py
    from brisk.training.workflow import Workflow

    class MyWorkflow(Workflow):
        def workflow(self):
            # Fit the model
            self.model.fit(self.X_train, self.y_train)

            # Evaluate the model
            self.evaluate_model(
                self.model, self.X_test, self.y_test,
                ["mean_absolute_error"], "model_score"
            )

            # Generate visualizations
            self.plot_learning_curve(
                self.model, self.X_train, self.y_train
            )

As mentioned earlier, you can create multiple workflows. You can use one workflow for your entire project, or assign different workflows to different experiment groups.
For details on using default workflows or assigning specific workflows to experiment groups, see 
:doc:`Using Experiment Groups </users/using_experiment_groups>`.


By embracing this modular structure, we hope you'll find it easier to try many
different experiments with your machine learning project while maintaining clean, 
organized code.