.. _brisk_cli:

Brisk CLI
=========

How to use the Brisk CLI
------------------------

Brisk provides a command-line interface (CLI) to help you create and run 
machine learning projects. The CLI is installed automatically when you install 
Brisk into a virtual environment, making it available through the ``brisk`` 
command in your terminal.

To see all available commands and options, run:

.. code-block:: bash

    brisk --help

Common Commands
---------------

create
^^^^^^

The ``create`` command initializes a new Brisk project with all the necessary 
files and directory structure:

.. code-block:: bash

    brisk create -n <project_name>

**Arguments:**

* ``-n, --project_name`` (required): Name of the project directory to create

**Example:**

.. code-block:: bash

    brisk create -n my_regression_project

This creates a new directory structure with the following files:

* ``.briskconfig``: Project configuration file
* ``settings.py``: Configuration settings for experiments
* ``algorithms.py``: Algorithm definitions
* ``metrics.py``: Metric definitions
* ``data.py``: Data management setup
* ``training.py``: Training manager setup
* ``workflows/``: Directory for workflow files
* ``datasets/``: Directory for data storage

run
^^^

The ``run`` command runs the experiments defined in the ``settings.py`` file using 
the specified workflow. It is best to call this command from the root of your 
project directory to ensure all the files are found.

.. code-block:: bash

    brisk run -w <workflow> -n <results_name> --disable_report

**Arguments:**

* ``-w, --workflow`` (required): Name of the workflow file (without .py extension) in the workflows/ directory
* ``-n, --results_name`` (optional): Custom name for the results directory
* ``--disable_report`` (optional): Flag to disable HTML report generation

**Example:**

.. code-block:: bash

    brisk run -w my_workflow -n experiment_1_results

This runs the workflow defined in ``workflows/my_workflow.py`` and saves the results in a directory named "experiment_1_results" within the results directory.

load_data
^^^^^^^^^

The ``load_data`` command wraps the ``load_sklearn_dataset`` function from scikit-learn 
and saves the dataset as a CSV file in the project's datasets directory.

.. code-block:: bash

    brisk load_data --dataset <dataset_name> --dataset_name <custom_name>

**Arguments:**

* ``--dataset`` (required): Name of the sklearn dataset to load (options: iris, wine, breast_cancer, diabetes, linnerud)
* ``--dataset_name`` (optional): Custom name to save the dataset as

**Example:**

.. code-block:: bash

    brisk load_data --dataset diabetes --dataset_name diabetes_data

This downloads the diabetes dataset from scikit-learn and saves it as "diabetes_data.csv" in your project's datasets directory.

create_data
^^^^^^^^^^^

The ``create_data`` command generates synthetic datasets for testing:

.. code-block:: bash

    brisk create_data --data_type <type> [options]

**Arguments:**

* ``--data_type`` (required): Type of dataset to generate (classification or regression)
* ``--n_samples`` (optional): Number of samples to generate (default: 100)
* ``--n_features`` (optional): Number of features to generate (default: 20)
* ``--n_classes`` (optional): Number of classes for classification (default: 2)
* ``--random_state`` (optional): Random seed for reproducibility (default: 42)
* ``--dataset_name`` (optional): Name for the dataset file (default: synthetic_dataset)

**Example:**

.. code-block:: bash

    brisk create_data --data_type regression --n_samples 500 --n_features 10 --dataset_name synthetic_regression

This creates a synthetic regression dataset with 500 samples and 10 features, saving it as "synthetic_regression.csv" in your project's datasets directory.

Working with the CLI
--------------------

The Brisk CLI is designed to be used from the root of your project directory. 
When running commands, Brisk will look for the `.briskconfig` file to 
identify the project root.
