====================
Start a New Project
====================

First make sure you have activated the virtual environment you created in the :ref:`install` section.

If you are using conda, you can activate your environment by running:

.. code-block:: bash

   conda activate myenv

If you are using venv, run:

.. code-block:: bash

   source venv/bin/activate


Create a Project Directory
==========================

Brisk comes with a :ref:`command line interface (CLI) <brisk_cli>` that is used for several tasks, 
including creating projects. To create a new project, run:

.. code-block:: bash

   brisk create -n tutorial

This will create a new project directory called ``tutorial`` in the current 
working directory. It will also generate several files with some boilerplate 
code to get started. Brisk has you split your configuration into several files, 
to keep your code organized and modular. For more details on these files, see 
:ref:`project_structure`.

The directory structure should look like this:

.. code-block::
   :caption: Project Directory Structure

   tutorial/
   ├── datasets/
   ├── workflows/
   │   └── workflow.py
   ├── .briskconfig
   ├── algorithms.py
   ├── data.py
   ├── metrics.py
   └── settings.py


Configure the Project
=====================

Before we can start training models, we need to configure the project. This involves
modifying several files in the ``tutorial`` directory.

Load a Dataset
--------------

First we need to have some data to model. For this tutorial we will use the 
diabetes dataset from the *sklearn* library. We can load the dataset using the 
``load-data`` command. Run the following commands:

.. code-block:: bash
   
   cd datasets/
   brisk load-data --dataset diabetes
   cd ..

You should now see a file called ``diabetes.csv`` in the ``datasets/`` directory.
Note that Brisk expects any dataset you use to be in this directory. It also expects
the dataset directory to be named ``datasets`` and remain in the root project directory.


Define Metrics
--------------

``metrics.py`` is where you define the metrics you want to use for evaluating the 
models. Brisk uses the MetricWrapper class to wrap the metric function along with
other useful information. When you open ``metrics.py`` you will see there is some 
boilerplate code that should look like this:

.. code-block:: python

    import brisk

    METRIC_CONFIG = brisk.MetricManager(
        brisk.MetricWrapper()
    )

You will want to leave the MetricManager class as is. MetricManager is used internally
by Brisk to manage the metrics. For this tutorial, we will use **mean absolute error**.
To do this we need to define a MetricWrapper for mean absolute error.

You can fill in the arguments for MetricWrapper as follows:

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

Make sure to import the ``metrics`` module from ``sklearn``. This is the function 
we want to use to calculate the mean absolute error. We also define a name and abbreviation (abbr)
for the metric. These will be used later to select the metric we want to use.
The display name is used whenever the metric name is used in plots or tables.

You can add more metrics by defining more MetricWrappers. Brisk also provides a
set of default metrics for :ref:`regression <default_regression_metrics>` and 
:ref:`classification <default_classification_metrics>` that should be sufficient 
for most projects.


Define Algorithms
------------------

``algorithms.py`` plays a similar role to metrics.py, but instead of defining the 
metrics, it defines the algorithms you want to use for training your models. You 
should see code that looks like this:

.. code-block:: python

    import brisk

    ALGORITHM_CONFIG = brisk.AlgorithmCollection(
        brisk.AlgorithmWrapper()
    )

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

Data Splitting
--------------

``data.py`` is where we set how we want to process and split our data by default. 
For this tutorial we can leave the test_size of 0.2. This will use 20% of the dataset 
for testing and 80% for training.

We won’t be processing the data in this tutorial, so we don’t need to change anything else. 
See the DataManager for more details on how the DataManager can be used to preprocess and split your data.

Define Workflows
----------------

Before we configure our experiments, we need to define how we want to train and 
evaluate our models. This is where the ``Workflow`` class comes in. In Brisk, a 
Workflow defines the steps we want to take for each experiment.

In ``workflows/workflow.py`` you will see a class called ``MyWorkflow`` that inherits 
the ``Workflow`` class. You will also see a ``workflow`` method that takes no 
arguments. This is where you define the steps you want to take for each experiment.

We will create a simple workflow as follows:

.. code-block:: python

    class MyWorkflow(Workflow):
        def workflow(self):
            # Fit the model to the training data
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate the model on the testing data
            self.evaluate_model(
                self.model, self.X_test, self.y_test,
                ["mean_absolute_error"], "pre_tuning_score"
            )
            
            # Tune the model hyperparameters
            tuned_model = self.hyperparameter_tuning(
                self.model, "random", self.X_train, self.y_train,
                "MAE", kf=5, num_rep=2, n_jobs=-1
            )
            
            # Evaluate the tuned model on the testing data
            self.evaluate_model(
                tuned_model, self.X_test, self.y_test,
                ["MAE"], "post_tuning_score"
            )
            
            # Plot the learning curve
            self.plot_learning_curve(
                tuned_model, self.X_train, self.y_train, metric="MAE"
            )
            
            # Plot feature importance using SHAP values
            self.plot_shapley_values(
                tuned_model, self.X_test, self.y_test, 
                filename="feature_importance"
            )

            # Save the tuned model
            self.save_model(tuned_model, "tuned_model")

We can access our mean absolute error metric from ``metrics.py`` by using the name 
or the abbreviation. Remember that this workflow will be executed once for each 
algorithm in the experiment groups that use it. Since the same workflow code runs 
for different algorithms (linear, ridge, lasso), it is best not to hardcode algorithm 
names in variables or filenames. See the :ref:`EvaluationManager` for more information on the evaluation methods available.

As a final note you’ll notice that workflows are given their own ``workflows`` directory. 
This allows you to have multiple workflows in the same project. Each .py file can only contain one Workflow subclass. This is to avoid using the wrong workflow at runtime.

Training Settings
-----------------

``settings.py`` is where we configure our experiments by bringing together all the 
components we've defined. In Brisk, an experiment refers to running specific algorithms on datasets with particular 
configurations. We use ExperimentGroups to organize related experiments together.

When the CLI creates this file it defines a ``create_configuration`` function that
returns a ``ConfigurationManager`` instance. ``Configuration`` class provides an 
interface for defining the experiments and checks all the inputs are valid. It is
important that this function returns ``config.build()``

You should see code that looks like this:

.. code-block:: python

    from brisk.configuration.configuration import Configuration, ConfigurationManager

    def create_configuration() -> ConfigurationManager:
        config = Configuration(
            default_workflow="workflow",
            default_algorithms=["linear"]
        )

        config.add_experiment_group(
            name="group_name",
        )
                    
        return config.build()

First we specify the default workflow and algorithms to use. The ``default_workflow="workflow"`` 
tells Brisk to use the ``MyWorkflow`` class from ``workflows/workflow.py`` for any experiment 
groups that don't specify their own workflow. The ``default_algorithms`` will be used by any 
ExperimentGroup that doesn't specify its own algorithms. We want to use all three algorithms 
we defined in ``algorithms.py``:

.. code-block:: python

    config = Configuration(
        default_workflow="workflow",
        default_algorithms=["linear", "ridge", "lasso"]
    )

We use the ``name`` property of the AlgorithmWrappers to select the algorithms 
we want to use.

Next we will define an ExperimentGroup:

.. code-block:: python

    config.add_experiment_group(
        name="tutorial",
        description="Training linear models for the Brisk tutorial.",
        datasets=["diabetes.csv"]
    )

The results will be organized by experiment group and dataset. Providing a meaningful
name and an optional description is useful for organizing your results and remembering
how the models were trained. We also need to specify a list of datasets we want 
to use. In this case we only have one dataset, but we could add more if we wanted.
Notice the path to the dataset is relative to the ``datasets/`` directory for convenience.

You can add more ExperimentGroups by calling ``add_experiment_group`` again. Most
of your time will be spent here defining the experiments you want to run. This guide
only covers the basics, but you can learn more about ExperimentGroups in the 
:ref:`using_experiment_groups` section.

Next, let's look at how we can run the experiments!


