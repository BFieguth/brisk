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
   ├── evaluators.py
   ├── metrics.py
   └── settings.py


Configure the Project
=====================

Before we can start training models, we need to provide training data and configure the
experiments we want to run. This involves modifying several files in the ``tutorial`` directory.

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
models. Brisk uses the :ref:`MetricWrapper <metric_wrapper>` class to wrap the metric function along with
other useful information. When you open ``metrics.py`` you will see there is some 
boilerplate code that should look like this:

.. code-block:: python

    import brisk

    METRIC_CONFIG = brisk.MetricManager(
        *brisk.REGRESSION_METRICS,
        *brisk.CLASSIFICATION_METRICS
    )

Brisk comes with a set of predefined metrics for :ref:`regression <default_regression_metrics>` and 
:ref:`classification <default_classification_metrics>`. These wrappers are imported
and unpacked into the :ref:`MetricManager <metric_manager>` class, making them
available for use by various components of Brisk. In most cases these provided metrics
should be sufficient, but you can always define your own metrics by following this
:ref:`guide <custom_metrics>`.

.. important::
    You must use the name ``METRIC_CONFIG`` as this is what Brisk will
    look for to load this data at runtime.

Define Algorithms
------------------

``algorithms.py`` plays a similar role to metrics.py, but instead of defining the 
metrics, it defines the algorithms you want to use for training your models. You 
should see code that looks like this:

.. code-block:: python

    import brisk

    ALGORITHM_CONFIG = brisk.AlgorithmCollection(
        *brisk.REGRESSION_ALGORITHMS,
        *brisk.CLASSIFICATION_ALGORITHMS
    )

As with ``metrics.py`` Brisk provides a set of predefined algorithms for :ref:`regression <default_regression_algorithms>` and 
:ref:`classification <default_classification_algorithms>`. These wrappers are imported
and unpacked into the :ref:`AlgorithmCollection <algorithm_collection>` class.

These algorithms are meant to be a convenience for getting started with Brisk. They
are unlikely to be optimal for most projects. See the :ref:`adding algorithms<add_algorithms>` guide
for more information on how to define your own algorithms.

Data Splitting
--------------

``data.py`` is where we set how we want to process and split our data by default. 
For this tutorial we can leave the test_size of 0.2. This will use 20% of the dataset 
for testing and the remaining 80% for training.

.. code-block:: python

    from brisk.data.data_manager import DataManager

    BASE_DATA_MANAGER = DataManager(
        test_size = 0.2
    )


We won’t be processing the data in this tutorial, so we don’t need to change anything else. 
See :ref:`DataManager <api_data_manager>` for more details on how the DataManager
can be used to split your data or the :ref:`applying preprocessing<applying_preprocessing>` guide for more information
on how to use the built-in data preprocessing capabilities.

.. note::
    By default DataManager will create 5 training and testing splits.
    You can reduce this number by changing the ``n_splits`` argument if you want
    the tutorial to run faster.
    
    .. code-block:: python

        BASE_DATA_MANAGER = DataManager(
            test_size = 0.2,
            n_splits = 1
        )

Define Workflows
----------------

Before we configure our experiments, we need to define how we want to train and 
evaluate our models. This is where the ``Workflow`` class comes in. In Brisk, a 
Workflow defines the steps we want to take for each experiment.

In ``workflows/workflow.py`` you will see a class called ``MyWorkflow`` that inherits 
the ``Workflow`` class and an empty ``workflow`` method. This is where you define
the steps you want to take to train and evaluate models for each experiment.

Brisk comes with a simple workflow setup for a regression problem. You can see it below:

.. code-block:: python

    from brisk.training.workflow import Workflow

    class MyWorkflow(Workflow):
        def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
            self.model.fit(self.X_train, self.y_train)
            self.evaluate_model_cv(
                self.model, self.X_train, self.y_train, ["MAE"], "pre_tune_score"
            )
            tuned_model = self.hyperparameter_tuning(
                self.model, "grid", self.X_train, self.y_train, "MAE",
                kf=5, num_rep=3, n_jobs=-1
            )
            self.evaluate_model(
                tuned_model, self.X_test, self.y_test, ["MAE"], "post_tune_score"
            )
            self.plot_learning_curve(tuned_model, self.X_train, self.y_train)
            self.save_model(tuned_model, "tuned_model")

.. note::
    If you want to use this workflow to try a classification problem, you can change the
    ``MAE`` value to ``accuracy`` or any other classification metric. This is not always
    the case as some methods are specific to classification or regression type problems.

We access our mean absolute error metric from ``metrics.py`` by using the name,
or in this case the abbreviation. This workflow will be run once for each 
algorithm in the experiment setup. Since the same workflow code runs 
for different algorithms it is best not to hardcode algorithm names in variables
or filenames as this may lead to confusion when looking at the results.

As a final note you’ll notice that the ``workflow.py`` file is given its own ``workflows`` directory. 
This allows you to have multiple workflows in the same project. Each .py file can
only contain one Workflow subclass. This is to avoid using the wrong workflow at runtime.
You can specify the workflow to use in the next step by using the file name without the ``.py`` extension.

Training Settings
-----------------

``settings.py`` is where we configure our experiments by bringing together all the 
components we've defined. In Brisk, an experiment refers to running a specific workflow
on a dataset. We use ExperimentGroups to organize related experiments together and
override default values allowing you to try different setups quickly and easily.

When the CLI creates this file it defines a ``create_configuration`` function that
returns a ``ConfigurationManager`` instance. The ``Configuration`` class provides an 
interface for defining the experiments and checks all the inputs are valid. It is
important that this function returns ``config.build()``

You should see code that looks like this:

.. code-block:: python

    from brisk.configuration.configuration import Configuration
    from brisk.configuration.configuration_manager import ConfigurationManager

    def create_configuration() -> ConfigurationManager:
        config = Configuration(
            default_workflow = "workflow",
            default_algorithms = ["linear"],
        )

        config.add_experiment_group(
            name="group_name",
            datasets=[],
            workflow="workflow"  
        )

        return config.build()

First we specify the default workflow and algorithms to use. The ``default_workflow="workflow"`` 
tells Brisk to use the ``MyWorkflow`` class from ``workflows/workflow.py`` for any experiment 
groups that don't specify their own workflow. The same applies to all the default values.
We select the algorithm by using the ``name`` property of the AlgorithmWrappers to
select the algorithms we want to use. For this tutorial we will just train a
linear regression model.

Next we will add an ExperimentGroup:

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

You can add as many experiment groups as you want by calling ``add_experiment_group`` again.
Most of your time will be spent here defining the experiments you want to run. This guide
only covers the basics, but you can learn more about ExperimentGroups in the 
:ref:`using_experiment_groups` section.

Next, let's look at how we can run the experiments!
