"""Command-line interface for the Brisk machine learning framework.

This module provides a comprehensive CLI for managing machine learning
experiments with Brisk. It includes commands for creating new projects, running
experiments, loading datasets, generating synthetic data, and managing
environment reproducibility.

The CLI is built using Click and provides the following main functionality:
- Project initialization with template files
- Experiment execution with configurable workflows
- Dataset loading from scikit-learn and synthetic data generation
- Environment management for reproducible experiments
- Results export and environment compatibility checking

Commands
--------
create
    Initialize a new project directory with configuration files
run
    Execute experiments based on a specified workflow
load_data
    Load datasets from scikit-learn into the project
create_data
    Generate synthetic datasets for testing
export-env
    Export environment requirements from a previous run
check-env
    Check environment compatibility with a previous run

Examples
--------
Create a new project:
    $ brisk create -n my_project

Run an experiment:
    $ brisk run

Load a dataset:
    $ brisk load_data --dataset iris --dataset_name my_iris

Check environment compatibility:
    $ brisk check-env my_run_20240101_120000 --verbose

Export environment requirements:
    $ brisk export-env my_run_20240101_120000 --output requirements.txt
"""
import os
import sys
from typing import Optional
from datetime import datetime
import json
import pathlib

import click
import pandas as pd
from sklearn import datasets

from brisk.configuration import project
from brisk.cli.cli_helpers import (
    _run_from_project, _run_from_config, load_sklearn_dataset,
)
from brisk.cli.environment import EnvironmentManager, VersionMatch

@click.group()
def cli() -> None:
    """Main entry point for Brisk's command line interface."""
    pass


@cli.command()
@click.option(
    "-n",
    "--project_name",
    required=True,
    help="Name of the project directory."
)
def create(project_name: str) -> None:
    """Create a new project directory with template files.

    Initializes a new Brisk project with all necessary configuration files
    and directory structure. Creates template files for algorithms, metrics,
    data management, workflows, and evaluators.

    Parameters
    ----------
    project_name : str
        Name of the project directory to create

    Notes
    -----
    Creates the following structure:
    - .briskconfig : Project configuration file
    - settings.py : Configuration settings with default experiment groups
    - algorithms.py : Algorithm definitions using Brisk's built-in algorithms
    - metrics.py : Metric definitions using Brisk's built-in metrics
    - data.py : Data management setup with default parameters
    - evaluators.py : Template for custom evaluators
    - workflows/ : Directory for workflow definitions
        - workflow.py : Template workflow class
    - datasets/ : Directory for data storage

    The created files contain working examples and can be customized
    for specific project needs.
    """
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)

    with open(
        os.path.join(project_dir, ".briskconfig"), "w", encoding="utf-8"
    ) as f:
        f.write(f'project_name={project_name}\n')

    with open(
        os.path.join(project_dir, "settings.py"), "w", encoding="utf-8"
    ) as f:
        f.write("""# settings.py
from brisk.configuration.configuration import Configuration, ConfigurationManager

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
""")

    with open(
        os.path.join(project_dir, "algorithms.py"), "w", encoding="utf-8"
    ) as f:
        f.write("""# algorithms.py
import brisk

ALGORITHM_CONFIG = brisk.AlgorithmCollection(
    *brisk.REGRESSION_ALGORITHMS,
    *brisk.CLASSIFICATION_ALGORITHMS
)
""")

    with open(
        os.path.join(project_dir, "metrics.py"), "w", encoding="utf-8"
    ) as f:
        f.write("""# metrics.py
import brisk

METRIC_CONFIG = brisk.MetricManager(
    *brisk.REGRESSION_METRICS,
    *brisk.CLASSIFICATION_METRICS
)
""")

    with open(
        os.path.join(project_dir, "data.py"), "w", encoding="utf-8"
    ) as f:
        f.write("""# data.py
from brisk.data.data_manager import DataManager

BASE_DATA_MANAGER = DataManager(
    test_size = 0.2
)
""")

    datasets_dir = os.path.join(project_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    workflows_dir = os.path.join(project_dir, "workflows")
    os.makedirs(workflows_dir, exist_ok=True)

    with open(
        os.path.join(workflows_dir, "workflow.py"), "w", encoding="utf-8"
    ) as f:
        f.write("""# workflow.py
# Define the workflow for training and evaluating models

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
""")
    with open(
        os.path.join(project_dir, "evaluators.py"), "w", encoding="utf-8"
    ) as f:
        f.write("""# evaluators.py
# Define custom evaluation methods here to integrate with Brisk's builtin tools
from brisk.evaluation.evaluators.registry import EvaluatorRegistry
from brisk import PlotEvaluator, MeasureEvaluator

def register_custom_evaluators(registry: EvaluatorRegistry, plot_settings) -> None:
    # registry.register(
    # Initalize an evaluator instance here to register
    # )
    pass

""")

    print(f'A new project was created in: {project_dir}')


@cli.command()
@click.option(
    "-n",
    "--results_name",
    default=None,
    help="The name of the results directory."
)
@click.option(
    "-f",
    "--config_file",
    default=None,
    help="Name of the results folder to run from config file."
)
@click.option(
    "--disable_report",
    is_flag=True,
    default=False,
    help="Disable the creation of an HTML report."
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Change the verbosity of the logger."
)
def run(
    results_name: Optional[str],
    config_file: Optional[str],
    disable_report: bool,
    verbose: bool
) -> None:
    """Run experiments using experiment groups in settings.py.

    Executes machine learning experiments based on configuration defined
    in the project's settings.py file. Can run from scratch or rerun
    from a saved configuration.

    Parameters
    ----------
    results_name : str, optional
        Custom name for results directory. If not provided, uses timestamp
        format: DD_MM_YYYY_HH_MM_SS
    config_file : str, optional
        Name of the results folder to run from saved configuration.
        If provided, reruns experiments using the saved configuration.
    disable_report : bool, default=False
        Whether to disable HTML report generation after experiments complete
    verbose : bool, default=False
        Whether to enable verbose logging output

    Notes
    -----
    The function automatically:
    1. Finds the project root directory
    2. Creates a results directory with timestamp or custom name
    3. Loads algorithms, metrics, and configuration from project files
    4. Executes experiments according to the workflow
    5. Generates an HTML report (unless disabled)

    Raises
    ------
    FileNotFoundError
        If project root not found or required files are missing
    FileExistsError
        If results directory already exists
    ValueError
        If experiment groups are missing workflow mappings or configuration
        errors
    """
    create_report = not disable_report
    project_root = project.find_project_root()

    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))

    if not results_name:
        results_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    results_dir = os.path.join("results", results_name)
    if os.path.exists(results_dir):
        raise FileExistsError(
            f"Results directory '{results_dir}' already exists."
        )
    os.makedirs(results_dir, exist_ok=False)

    if config_file:
        _run_from_config(
            project_root, verbose, create_report, results_dir, config_file
        )
    else:
        _run_from_project(
            project_root, verbose, create_report, results_dir
        )


@cli.command()
@click.option(
    "--dataset",
    type=click.Choice(
        ["iris", "wine", "breast_cancer", "diabetes", "linnerud"]
        ),
    required=True,
    help=(
        "Name of the sklearn dataset to load. Options are iris, wine, "
        "breast_cancer, diabetes, or linnerud."
    )
)
@click.option(
    "--dataset_name",
    type=str,
    default=None,
    help="Name to save the dataset as."
)
def load_data(dataset: str, dataset_name: Optional[str] = None) -> None:
    """Load a scikit-learn dataset into the project.

    Downloads and saves a scikit-learn dataset as a CSV file in the
    project's datasets directory. Automatically handles feature names
    and target variable formatting.

    Parameters
    ----------
    dataset : {'iris', 'wine', 'breast_cancer', 'diabetes', 'linnerud'}
        Name of the scikit-learn dataset to load
    dataset_name : str, optional
        Custom name for the saved dataset file. If not provided,
        uses the original dataset name

    Notes
    -----
    Saves the dataset as a CSV file in the project's datasets directory.
    The CSV includes:
    - Feature columns with proper names (or feature_0, feature_1, etc.)
    - Target column named 'target'
    - No index column

    Available datasets:
    - iris: 150 samples, 4 features, 3 classes
    - wine: 178 samples, 13 features, 3 classes  
    - breast_cancer: 569 samples, 30 features, 2 classes
    - diabetes: 442 samples, 10 features, regression target
    - linnerud: 20 samples, 3 features, 3 targets

    Raises
    ------
    FileNotFoundError
        If project root directory is not found
    """
    try:
        project_root = project.find_project_root()
        datasets_dir = os.path.join(project_root, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        data = load_sklearn_dataset(dataset)
        if data is None:
            print(
                f"Dataset \'{dataset}\' not found in sklearn. Options are "
                "iris, wine, breast_cancer, diabetes or linnerud."
            )
            return
        X = data.data # pylint: disable=C0103
        y = data.target

        feature_names = (
            data.feature_names
            if hasattr(data, "feature_names")
            else [f"feature_{i}" for i in range(X.shape[1])]
            )
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        dataset_filename = dataset_name if dataset_name else dataset
        csv_path = os.path.join(datasets_dir, f"{dataset_filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f'Dataset saved to {csv_path}')

    except FileNotFoundError as e:
        print(f'Error: {e}')


@cli.command()
@click.option(
    "--data_type",
    type=click.Choice(["classification", "regression"]),
    required=True,
    help="Type of the synthetic dataset."
)
@click.option(
    "--n_samples",
    type=int,
    default=100,
    help="Number of samples for synthetic data."
)
@click.option(
    "--n_features",
    type=int,
    default=20,
    help="Number of features for synthetic data."
)
@click.option(
    "--n_classes",
    type=int,
    default=2,
    help="Number of classes for classification data."
)
@click.option(
    "--random_state",
    type=int,
    default=42,
    help="Random state for reproducibility."
)
@click.option(
    "--dataset_name",
    type=str,
    default="synthetic_dataset",
    help="Name of the dataset file to be saved."
)
def create_data(
    data_type: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    random_state: int,
    dataset_name: str
    ) -> None:
    """Create synthetic data and add it to the project.

    Generates synthetic datasets for testing and experimentation using
    scikit-learn's data generation functions. Supports both classification
    and regression datasets with configurable parameters.

    Parameters
    ----------
    data_type : {'classification', 'regression'}
        Type of dataset to generate
    n_samples : int, default=100
        Number of samples to generate
    n_features : int, default=20
        Number of features to generate
    n_classes : int, default=2
        Number of classes for classification datasets
    random_state : int, default=42
        Random seed for reproducibility
    dataset_name : str, default='synthetic_dataset'
        Name for the output CSV file (without extension)

    Notes
    -----
    For classification datasets:
        - 80% informative features
        - 20% redundant features  
        - No repeated features
        - Balanced class distribution

    For regression datasets:
        - 80% informative features
        - 0.1 noise level
        - Linear relationship between features and target

    The generated dataset is saved as a CSV file in the project's
    datasets directory with feature columns and a 'target' column.

    Raises
    ------
    FileNotFoundError
        If project root directory is not found
    ValueError
        If data_type is not 'classification' or 'regression'
    """
    try:
        project_root = project.find_project_root()
        datasets_dir = os.path.join(project_root, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        if data_type == "classification":
            X, y = datasets.make_classification( # pylint: disable=C0103
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.8),
                n_redundant=int(n_features * 0.2),
                n_repeated=0,
                n_classes=n_classes,
                random_state=random_state
            )
        elif data_type == "regression":
            X, y, _ = datasets.make_regression( # pylint: disable=C0103
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.8),
                noise=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        df = pd.DataFrame(X)
        df["target"] = y
        csv_path = os.path.join(datasets_dir, f"{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Synthetic dataset saved to {csv_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")


@cli.command("export-env")
@click.argument("run_id")
@click.option("--output", "-o", help="Output path for requirements file")
@click.option(
    "--include-all",
    is_flag=True,
    help="Include all packages, not just critical ones"
)
def export_env(run_id: str, output: Optional[str], include_all: bool) -> None:
    """Export environment requirements from a previous run.
    
    Creates a requirements.txt file from the environment captured during
    a previous experiment run. By default, only includes critical packages
    that affect computation results.

    Parameters
    ----------
    run_id : str
        The run ID to export environment from (e.g., '2024_01_15_14_30_00')
    output : str, optional
        Output path for requirements file. If not provided, saves as
        'requirements_{run_id}.txt' in the project root
    include_all : bool, default=False
        Include all packages from the original environment, not just
        critical ones (numpy, pandas, scikit-learn, scipy, joblib)

    Notes
    -----
    The generated requirements.txt file includes:
    - Header comments with generation timestamp
    - Python version information
    - Critical packages section (always included)
    - Other packages section (if include_all=True)
    - Proper package version pinning for reproducibility

    Examples
    --------
    Export critical packages only:
        brisk export-env my_run_20240101_120000

    Export all packages to custom file:
        brisk export-env my_run_20240101_120000 --output my_requirements.txt
        --include-all

    Raises
    ------
    FileNotFoundError
        If run configuration file is not found
    """
    project_root = project.find_project_root()
    config_path = project_root / "results" / run_id / "run_config.json"

    if not config_path.exists():
        print(f"Error: Run configuration not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    env_manager = EnvironmentManager(project_root)

    if output:
        output_path = pathlib.Path(output)
    else:
        output_path = project_root / f"requirements_{run_id}.txt"

    saved_env = config.get("env", {})
    if not saved_env:
        print("Error: No environment information found in run configuration")
        return

    req_path = env_manager.export_requirements(
        saved_env,
        output_path,
        include_all=include_all
    )

    print(f"Requirements exported to: {req_path}")
    print("\nTo recreate this environment:")
    print("  python -m venv brisk_env")
    print(
        "  source brisk_env/bin/activate  "
        "# On Windows: brisk_env\\Scripts\\activate"
    )
    print(f"  pip install -r {req_path.name}")


@cli.command("check-env")
@click.argument("run_id")
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed compatibility report"
)
def check_env(run_id: str, verbose: bool) -> None:
    """Check environment compatibility with a previous run.
    
    Compares the current Python environment with the environment used
    in a previous experiment run. Identifies version differences and
    potential compatibility issues that could affect reproducibility.

    Parameters
    ----------
    run_id : str
        The run ID to check environment against (e.g., '2024_01_15_14_30_00')
    verbose : bool, default=False
        Show detailed compatibility report with all package differences.
        If False, shows only summary information

    Notes
    -----
    The compatibility check examines:
    - Python version compatibility (major.minor version must match)
    - Critical package versions (numpy, pandas, scikit-learn, scipy, joblib)
    - Non-critical package differences
    - Missing or extra packages

    Compatibility rules:
    - Critical packages: major.minor version must match exactly
    - Non-critical packages: major version must match
    - Missing critical packages: breaks compatibility
    - Python version: major.minor must match

    Examples
    --------
    Quick compatibility check:
        brisk check-env my_run_20240101_120000

    Detailed compatibility report:
        brisk check-env my_run_20240101_120000 --verbose

    Raises
    ------
    FileNotFoundError
        If run configuration file is not found
    """
    project_root = pathlib.Path.cwd()
    config_path = project_root / "results" / run_id / "run_config.json"

    if not config_path.exists():
        print(f"Error: Run configuration not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    saved_env = config.get("env", {})
    if not saved_env:
        print("Error: No environment information found in run configuration")
        return

    env_manager = EnvironmentManager(project_root)

    if verbose:
        report = env_manager.generate_environment_report(saved_env)
        print(report)
    else:
        differences, is_compatible = env_manager.compare_environments(saved_env)

        if is_compatible:
            print("Environment is compatible")
        else:
            critical_diffs = [
                d for d in differences
                if d.status in [VersionMatch.MISSING, VersionMatch.INCOMPATIBLE]
            ]

            print(f"Environment has {len(critical_diffs)} critical differences")
            print("\nRun with --verbose for full report, or use:")
            print(f"  brisk export-env {run_id} --output requirements.txt")
            print(
                "to export requirements for recreating the original environment"
            )


if __name__ == "__main__":
    cli()
