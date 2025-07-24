.. _using_experiment_groups:

Using ExperimentGroups
======================

The ``ExperimentGroup`` object is used to define a group of experiments. This can be used
to help orgainze your experiments. The ``ExperimentGroup`` class also allows you to override
the default settings for a set of experiments. This allows you to test out different values
without having to change the configuration file.


Modify DataManager
------------------
By default Brisk will use the ``DataManager`` defined in the ``data.py`` configuration file.
The ``data_config`` argument is used to modify the ``DataManager`` for the group.
To do this you define a dictionary of arguments that will be passed to the ``DataManager``
constructor.

As an example imagine we define the following ``DataManager`` in the ``data.py`` file:

.. code-block:: python

    data_manager = DataManager(
        test_size=0.2,
        split_method="shuffle",
        scale_method="standard"
    )

This will use a shuffle split method to create a train/test split of 80/20. It will also
use standard scaling to scale the input features.

What if we want to compare minmax scaling to standard scaling? We can create an 
``ExperimentGroup`` that uses minmax scaling instead of standard scaling.

.. code-block:: python

    config.add_experiment_group(
        name="standard_scaling",
        datasets=["data.csv"],
        description="Use standard scaling on the data."
    )

    config.add_experiment_group(
        name="minmax_scaling",
        datasets=["data.csv"],
        data_config={"scale_method": "minmax"},
        description="Use minmax scaling on the data."
    )

Any argument passed to the ``DataManager`` constructor can be modified in this way.

.. note::
    If you are using multiple datasets in the ``ExperimentGroup`` then the same 
    ``DataManager`` will be used for **all** of the datasets. If you want to use different
    ``DataManager`` for each dataset then you will need to define a seperate 
    ``ExperimentGroup`` for each dataset.


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
configuration file.

.. important::
    ``algorithm_config`` will only modify the specified hyperparameter in the 
    hyperparameter grid. This means the other hyperparameters will still use the 
    values defined in the ``AlgorithmWrapper``. It is not possible to remove a 
    hyperparameter from the grid using ``algorithm_config``.


Pass Workflow Arguments
-----------------------
It is also possible to pass values to the ``Workflow`` class. This can be done using the
``workflow_args`` argument. This is a dictionary of arguments that will be passed to the
``Workflow`` constructor. The keys of the dictionary will be the name of the instance variable
in the ``Workflow`` class.

When doing this there are a few things to keep in mind:

- Since these values are accessed as instance variables in the workflow they must be defined for all ExperimentGroups. Otherwise the workflow will not have access to the values and will throw a ValueError.

- You should avoid using the names of attributes that are already defined in the ``Workflow`` class. See :ref:`api_workflow` for more information.

If you want to use ``workflow_args`` it is best practice to first define the default values
in the ``Configuration`` object. This will ensure that the ``Workflow`` class has the
correct attributes. For example if we want to use a different value of ``kfold`` for each
experiment group we can do the following. First we define the default value in the 
``Configuration`` object:

.. code-block:: python

    config = Configuration(
        default_algorithms=["linear", "ridge"],
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
