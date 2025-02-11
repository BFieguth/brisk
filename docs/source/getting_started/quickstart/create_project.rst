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

Brisk comes with a command line interface (CLI) that is used for several tasks, 
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
   ├── settings.py
   └── training.py


Configure the Project
=====================

Before we can start training models, we need to configure the project. This involves
modifying several files in the ``tutorial`` directory.

Load a Dataset
--------------

Before we can start training models, we need to have some data to model. For this
tutorial we will use the diabetes dataset from the *sklearn* library. We can load 
the dataset using the ``load-data`` command. Run the following commands:

.. code-block:: bash
   
   cd datasets/
   brisk load-data -dataset diabetes
   cd ..

You should now see a file called ``diabetes.csv`` in the ``datasets/`` directory.


metrics.py
----------

``metrics.py`` is where you define the metrics you want to use for evaluating your 
models. Brisk uses the MetricWrapper class to wrap the metric function along with
other useful information. For this tutorial, we will use the default metrics 
provided by Brisk. 

You can delete the MetricWrapper class and replace it with the following code:

.. code-block:: python

   import brisk

   METRIC_CONFIG = brisk.MetricManager(
       *brisk.REGRESSION_METRICS
   )

This will unpack a list of the :ref:`default_regression_metrics`.


algorithms.py
-------------

``algorithms.py`` plays a similar role to metrics.py, but instead of defining the 
metrics, it defines the algorithms you want to use for training your models. 
For this tutorial, we will use the default algorithm provided by Brisk. See 
:ref:`default_regression_algorithms` for more details. 

These defaults are useful for getting started quickly, however it is recommended 
that you define AlgorithmWrappers and adjust the hyperparameter space based on 
your dataset.

You can delete the AlgorithmWrapper class and replace it with the following code:

.. code-block:: python

    import brisk
                    
    ALGORITHM_CONFIG = [
        *brisk.REGRESSION_ALGORITHMS
    ]        


data.py
-------

``data.py`` is where we set how we want to process and split our data by default.
For this tutorial we can leave the test_size of 0.2. This will use 20% of 
the dataset for testing and 80% for training. 

We won't be processing the data in this tutorial, so we don't need to change 
anything else. See the :ref:`using_data_manager` for more details on how the 
DataManager can be used to preprocess and split your data.


settings.py
-----------

``settings.py`` is where we define the experiments we want to run. In Brisk an 
experiment refers to the combinations of data splits and algorithms we want to 
run. For this tutorial we will only run a single experiment.

When the CLI creates this file it defines a ``create_configuration`` function that
returns a ``ConfigurationManager`` object. The Configuration class provides an 
interface for defining the experiments and checks all the inputs are valid.

The default_algorithms will be used for all experiment groups unless other algorithms 
are specified. We will add a Ridge Regression and an ElasticNet Regression to the 
default algorithms.

.. code-block:: python

    config = Configuration(
        default_algorithms = ["linear", "ridge", "elasticnet"],
    )

Here we use the name property of the AlgorithmWrappers to select the algorithms 
we want to use.

There is already one experiment group, created with the ``add_experiment_group`` 
method. We can change the name and description of this experiment group. We also
need to select the datasets we want to use. We will use the diabetes dataset we 
loaded earlier.

.. code-block:: python

    config.add_experiment_group(
        name="tutorial",
        description="A tutorial to learn the basics of Brisk",
        datasets=["diabetes.csv"],
    )


training.py
-----------

``training.py`` loads in the objects we defined in the other files and passes them 
to the TrainingManager. For the most part, you don't need to change anything in this 
file. You can set the *verbose* parameter to True to get more information about the 
experiments as they are run.

.. code-block:: python

    manager = TrainingManager(
        metric_config=METRIC_CONFIG,
        config_manager=config,
        verbose=True
    )


Create a Workflow
=================

So far we have defined the default values to use in our project and have setup 
some experiments to run. Before we can run the experiments, we need to define the 
steps to train and evaluate the model in each experiment. To do this we define a 
workflow method in the ``workflows/workflow.py`` file.

When you open the file you will see a class called ``MyWorkflow`` that inherits 
from the ``Workflow`` class. In the ``workflow`` method we have access to our model 
as well as the data splits and several instance methods. These are all accessed 
through ``self``. Brisk takes care of ensuring that model and data splits for 
the current experiment are passed to the workflow.

We can start by fitting the model to the training data. Then we can save the model 
to a file. Next we can plot the learning curve and evaluate the model on the test 
data. Finally we can plot the predicted vs observed values. All of these methods 
will save the results in the ``results/`` and organized by the experiment group 
and dataset. They will use the filenames you define here.

Remember the workflow will be run on each of your experiments which will use different
algorithms. It is a good idea to avoid using the algorithm name in the filenames.

You should have a workflow file that looks like this:

.. code-block:: python

    from brisk.training.workflow import Workflow

    class MyWorkflow(Workflow):
    def workflow(self):
        metrics_list = ["MAE", "MSE", "CCC"]

        self.model.fit(self.X_train, self.y_train)
        self.save_model(self.model, "fitted_model")

        self.plot_learning_curve(
            self.model, self.X_train, self.y_train
        )

        self.evaluate_model(
            self.model, self.X_test, self.y_test, metrics_list, "test_metrics"
        )

        self.plot_pred_vs_obs(
            self.model, self.X_test, self.y_test, "pred_vs_obs_test"
        )


Next
====

With a workflow defined, we are ready to train the models. Continue to the 
:ref:`run_experiments` section for instructions on how to run the experiments.
