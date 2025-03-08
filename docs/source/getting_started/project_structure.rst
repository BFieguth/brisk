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

* ``training.py`` brings everything together to train the models

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

Brisk provides a set of predefined wrappers that include most common metrics:

.. code-block:: python

    import brisk

    METRIC_CONFIG = brisk.MetricManager(
        *brisk.REGRESSION_METRICS,  # For regression tasks
        *brisk.CLASSIFICATION_METRICS  # For classification tasks
    )

data.py
~~~~~~~

Configures how your data will be processed and split. This file defines a 
``DataManager`` that handles train-test splitting and scaling. This is the default
processing used for all datasets. You can override this default for individual 
experiment groups in ``settings.py``.

.. code-block:: python

    from brisk.data.data_manager import DataManager

    BASE_DATA_MANAGER = DataManager(
        test_size=0.2,
        scale_method="minmax",
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

    from brisk.configuration.configuration import Configuration, ConfigurationManager

    def create_configuration() -> ConfigurationManager:
        config = Configuration(
            default_algorithms=["linear", "ridge", "lasso"],
        )

        config.add_experiment_group(
            name="minmax_scaled",
            description="Training models with minmax scaling",
            datasets=["diabetes.csv"],
            data_config={"test_size": 0.25, "scale_method": "minmax"}
        )

        config.add_experiment_group(
            name="standard_scaled",
            description="Training models with standard scaling",
            datasets=["diabetes.csv"],
            data_config={"test_size": 0.25, "scale_method": "standard"}
        )

        return config.build()


training.py
~~~~~~~~~~~

Responsible for running the experiments you've defined. This file typically 
doesn't need modification - it loads your configurations and creates a 
``TrainingManager`` that handles the training process.

.. code-block:: python

    from brisk.training.training_manager import TrainingManager
    from metrics import METRIC_CONFIG
    from settings import create_configuration

    config = create_configuration()

    # Define the TrainingManager for experiments
    manager = TrainingManager(
        metric_config=METRIC_CONFIG,
        config_manager=config
    )

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


By embracing this modular structure, we hope you'll find it easier to try many
different experiments with your machine learning project while maintaining clean, 
organized code.
