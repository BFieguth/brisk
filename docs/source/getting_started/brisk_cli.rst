.. _brisk_cli:

Brisk CLI
=========

How to use the Brisk CLI
------------------------

Brisk provides a comprehensive command-line interface (CLI) to help you create and run 
machine learning projects, manage datasets, and ensure reproducible experiments. The CLI is 
installed automatically when you install Brisk into a virtual environment, making it available 
through the ``brisk`` command in your terminal.

To see all available commands and options, run:

.. code-block:: bash

    brisk --help

Available Commands
------------------

Brisk provides six main commands for managing your machine learning workflow:

* ``create`` - Initialize a new project with template files
* ``run`` - Execute experiments based on your configuration
* ``load_data`` - Load scikit-learn datasets into your project
* ``create_data`` - Generate synthetic datasets for testing
* ``export-env`` - Export environment requirements from previous runs
* ``check-env`` - Check environment compatibility with previous runs

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
* ``evaluators.py``: Template for custom evaluators
* ``workflows/``: Directory for workflow files
* ``datasets/``: Directory for data storage

run
^^^

The ``run`` command runs the experiments defined in the ``settings.py`` file. You can 
either use one workflow for all experiment groups (by setting a ``default_workflow``) 
or assign different workflows to specific experiment groups. All workflow assignments 
are configured in ``settings.py``, so you just run the command from your project root.

.. code-block:: bash

    brisk run [OPTIONS]

**Arguments:**

* ``-n, --results_name`` (optional): Custom name for the results directory. If not provided, uses timestamp format DD_MM_YYYY_HH_MM_SS
* ``-f, --config_file`` (optional): Name of the results folder to run from saved configuration. If provided, reruns experiments using the saved configuration
* ``--disable_report`` (optional): Flag to disable HTML report generation after experiments complete
* ``--verbose`` (optional): Flag to enable verbose logging output

**Examples:**

.. code-block:: bash

    # Run experiments with custom results name
    brisk run -n experiment_1_results

    # Run experiments with verbose output and no report
    brisk run --verbose --disable_report

    # Rerun experiments from a previous configuration
    brisk run -f previous_run


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

export-env
^^^^^^^^^^

The ``export-env`` command creates a requirements.txt file from the environment captured during a previous experiment run. This helps with reproducibility by allowing you to recreate the exact environment used for specific experiments.

.. code-block:: bash

    brisk export-env <run_id> [OPTIONS]

**Arguments:**

* ``run_id`` (required): The run ID to export environment from (e.g., '2024_01_15_14_30_00')
* ``-o, --output`` (optional): Output path for requirements file. If not provided, saves as 'requirements_{run_id}.txt' in the project root
* ``--include-all`` (optional): Flag to include all packages from the original environment, not just critical ones (numpy, pandas, scikit-learn, scipy, joblib)

**Examples:**

.. code-block:: bash

    # Export critical packages only
    brisk export-env my_run_20240101_120000

    # Export all packages to custom file
    brisk export-env my_run_20240101_120000 --output my_requirements.txt --include-all

    # Export to specific location
    brisk export-env my_run_20240101_120000 -o /path/to/requirements.txt

This generates a requirements.txt file with proper package version pinning for reproducibility, including header comments with generation timestamp and Python version information.

check-env
^^^^^^^^^

The ``check-env`` command compares the current Python environment with the environment used in a previous experiment run. It identifies version differences and potential compatibility issues that could affect reproducibility.

.. code-block:: bash

    brisk check-env <run_id> [OPTIONS]

**Arguments:**

* ``run_id`` (required): The run ID to check environment against (e.g., '2024_01_15_14_30_00')
* ``-v, --verbose`` (optional): Flag to show detailed compatibility report with all package differences. If not provided, shows only summary information

**Examples:**

.. code-block:: bash

    # Quick compatibility check
    brisk check-env my_run_20240101_120000

    # Detailed compatibility report
    brisk check-env my_run_20240101_120000 --verbose

The compatibility check examines Python version compatibility, critical package versions (numpy, pandas, scikit-learn, scipy, joblib), and identifies missing or extra packages. Critical packages must have matching major.minor versions for compatibility.

Working with the CLI
--------------------

The Brisk CLI is designed to be used from the root of your project directory. 
When running commands, Brisk will look for the `.briskconfig` file to 
identify the project root.
