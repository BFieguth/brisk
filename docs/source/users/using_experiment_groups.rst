.. _using_experiment_groups:

Using ExperimentGroups
======================

The ``ExperimentGroup`` object is used to define a group of experiments. This can be used
to help organize your experiments. The ``ExperimentGroup`` class also allows you to override
the default settings for a set of experiments. This allows you to test out different values
without having to change the configuration files.


Modify DataManager
------------------
By default Brisk will use the ``DataManager`` defined in the ``data.py`` configuration file.
The ``data_config`` argument is used to modify the ``DataManager`` for the group.
To do this you define a dictionary of arguments that will be passed to the ``DataManager``
constructor.

As an example imagine we define the following ``DataManager`` in the ``data.py`` file:

.. code-block:: python

    from brisk.data.data_manager import DataManager
    from brisk.data.preprocessing import ScalingPreprocessor

    data_manager = DataManager(
        test_size=0.2,
        split_method="shuffle",
        preprocessors=[ScalingPreprocessor(method="standard")]
    )

This will use a shuffle split method to create a train/test split of 80/20. It will also
use standard scaling to scale the input features.

What if we want to compare different preprocessing strategies? We can create 
experiment groups with different preprocessing configurations:

.. code-block:: python

    from brisk.data.preprocessing import (
        ScalingPreprocessor, 
        FeatureSelectionPreprocessor
    )

    config.add_experiment_group(
        name="standard_scaling",
        datasets=["data.csv"],
        description="Use standard scaling (from data.py configuration)"
    )

    config.add_experiment_group(
        name="minmax_scaling",
        datasets=["data.csv"],
        data_config={"preprocessors": [ScalingPreprocessor(method="minmax")]},
        description="Use minmax scaling instead"
    )

    config.add_experiment_group(
        name="with_feature_selection",
        datasets=["data.csv"],
        data_config={"preprocessors": [
            ScalingPreprocessor(method="robust"),
            FeatureSelectionPreprocessor(method="selectkbest", n_features_to_select=10)
        ]},
        description="Robust scaling with top 10 features"
    )

Any argument passed to the ``DataManager`` constructor can be modified in this way.

.. note::
    If you are using multiple datasets in the ``ExperimentGroup`` then the same 
    ``DataManager`` will be used for **all** of the datasets. If you want to use different
    ``DataManager`` for each dataset then you will need to define a separate 
    ``ExperimentGroup`` for each dataset.

.. note::
    For database files, you can specify datasets as tuples: ``datasets=[("database.db", "table_name")]``


Modify Algorithms
-----------------
The ``algorithm_config`` argument is used to modify the hyperparameter grid used
for the algorithm in this group. You provide a nested dictionary where the first key
is the algorithm name and the second key is the hyperparameter to modify.

For example if we want to modify the hyperparameter grid for the ``ridge`` algorithm
to only use the values [0.01, 0.05, 0.1, 0.5, 1.0] for the ``alpha`` hyperparameter
we can do the following:

.. code-block:: python

    config.add_experiment_group(
        name="group1",
        datasets=["data.csv"],
        algorithms=["ridge"],
        algorithm_config={"ridge": 
            {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}
        }
    )

This allows you to test out different hyperparameter values without having to change the
``algorithms.py`` file.

.. important::
    ``algorithm_config`` will only modify the specified hyperparameter in the 
    hyperparameter grid. This means the other hyperparameters will still use the 
    values defined in the ``AlgorithmWrapper``. It is not possible to remove a 
    hyperparameter from the grid using ``algorithm_config``.


Assign Different Workflows
---------------------------
Each experiment group can use a different workflow by specifying the ``workflow`` 
parameter. This allows you to test different modeling approaches across experiment groups.

Setting Default Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, all experiment groups use the ``default_workflow`` specified in the 
``Configuration`` object:

.. code-block:: python

    # settings.py
    config = Configuration(
        default_algorithms=["linear", "ridge"],
        default_workflow="basic_workflow"  # All groups use this unless overridden
    )
    
    config.add_experiment_group(
        name="baseline", 
        datasets=["data.csv"]
        # Uses default_workflow automatically
    )

Assign Different Workflows for Specific Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can override the default workflow for specific experiment groups:

.. code-block:: python

    # settings.py
    config = Configuration(
        default_algorithms=["linear", "ridge"],
        default_workflow="basic_workflow"  # Default for all groups
    )
    
    config.add_experiment_group(
        name="baseline_models",
        datasets=["data.csv"],
        algorithms=["linear", "ridge"]
        # Uses default_workflow (basic_workflow)
    )

    config.add_experiment_group(
        name="tuned_models",
        datasets=["data.csv"],
        algorithms=["rf", "svm"],
        workflow="advanced_workflow",  # Overrides default_workflow
        description="Hyperparameter tuning with customized evaluation"
    )

.. important::
   The workflow filename (without ``.py``) must match the workflow name you specify 
   in ``add_experiment_group()``. For example, ``workflow="my_analysis"`` expects 
   a file named ``workflows/my_analysis.py``.

.. note::
   For detailed examples of creating these workflow files, see the 
   :doc:`Project Structure </getting_started/project_structure>` guide.

Pass Workflow Arguments
-----------------------

If you want to use ``workflow_args`` it is best practice to first define the default values
in the ``Configuration`` object. This will ensure that the ``Workflow`` class has the
correct attributes for every run. If a value is not set for a workflow argument in an experiment group,
the workflow may fail when trying to use this argument. 

As an example say we want to use a different value of ``kfold`` for each
experiment group we can do the following. First we define the default value in the 
``Configuration`` object:

.. code-block:: python

    config = Configuration(
        default_workflow="basic_workflow",
        default_algorithms=["linear", "ridge"],
        categorical_features={"housing_data.csv": ["neighborhood", "property_type"]},
        default_workflow_args={"kfold": 5}
    )

Then we can create two experiment groups with different values of ``kfold``.

.. code-block:: python

    config.add_experiment_group(
        name="group1",
        datasets=["data.csv"]
        # Uses default_workflow_args
    )

    config.add_experiment_group(
        name="group2",
        datasets=["data.csv"],
        workflow_args={"kfold": 10} # Overrides default_workflow_args
    )

Now when we are creating a workflow we can access the ``kfold`` value as ``self.kfold``:

.. code-block:: python

    def workflow(self):
        self.evaluate_model_cv(
            self.model, self.X_train, self.y_train, 
            ["MAE", "MSE"], "evaluate_cv", cv=self.kfold
        )

You can include as many arguments as you want in the ``workflow_args`` dictionary.
This is a good way to avoid hardcoding values in the ``Workflow`` class and helps
ensure you use the same value for a particular argument across all method calls in
the workflow.
