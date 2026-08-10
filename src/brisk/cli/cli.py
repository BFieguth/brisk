"""Command-line interface for the Brisk machine learning framework.

This module provides a comprehensive CLI for managing machine learning
experiments with Brisk. It includes commands for creating new projects, running
experiments, loading datasets, generating synthetic data, and managing
environment reproducibility.

The CLI is built using Click and provides the following main functionality:
- Project initialization with template files
- Experiment execution with configurable workflows
- Dataset loading from scikit-learn and synthetic data generation
- Preprocessing a dataset without running training
- Environment management for reproducible experiments
- Results export and environment compatibility checking

Commands
--------
create
    Initialize a new project directory with configuration files
run
    Execute experiments based on a specified workflow
ui
    Launch the brisk web UI for dashboard or project creation
migrate
    Migrate a project from legacy .briskconfig to .brisk/ format
load_data
    Load datasets from scikit-learn into the project
create_data
    Generate synthetic datasets for testing
preprocess
    Apply configured preprocessing to a dataset without training
export-env
    Export environment requirements from a previous run
check-env
    Check environment compatibility with a previous run

Examples
--------
Create a new project:
    $ brisk create -n my_project -t regression

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
import sqlite3
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
    _preprocess_dataset,
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
@click.option(
    "-t",
    "--type",
    "problem_type",
    required=True,
    type=click.Choice(["classification", "regression"]),
    help="Problem type: classification or regression."
)
def create(project_name: str, problem_type: str) -> None:
    """Create a new project directory with template files.

    Initializes a new Brisk project with all necessary configuration files
    and directory structure. Templates are tailored to the specified problem
    type with appropriate default algorithms, metrics, and workflow.

    Parameters
    ----------
    project_name : str
        Name of the project directory to create
    problem_type : str
        The machine learning problem type (classification or regression)

    Notes
    -----
    Creates the following structure:

    - .brisk/ : Project metadata directory
        - brisk.sqlite : Project database (empty)
        - project.json : Project metadata for the UI
    - settings.py : Configuration settings with default experiment groups
    - algorithms.py : Algorithm definitions for the chosen problem type
    - metrics.py : Metric definitions for the chosen problem type
    - data.py : Data management setup with default parameters
    - evaluators.py : Template for custom evaluators
    - workflows/ : Directory for workflow definitions
        - workflow.py : Template workflow class
    - datasets/ : Directory for data storage
    """
    project_dir = pathlib.Path.cwd() / project_name
    project_dir.mkdir(exist_ok=True)

    _create_brisk_dir(project_dir, project_name, problem_type)
    _create_settings(project_dir, problem_type)
    _create_algorithms(project_dir, problem_type)
    _create_metrics(project_dir, problem_type)
    _create_data(project_dir)
    _create_evaluators(project_dir)

    (project_dir / "datasets").mkdir(exist_ok=True)

    workflows_dir = project_dir / "workflows"
    workflows_dir.mkdir(exist_ok=True)
    _create_workflow(workflows_dir, problem_type)

    print(f"A new {problem_type} project was created in: {project_dir}")
    print("\nNext steps:")
    print(f"  1. Add a dataset CSV to {project_dir / 'datasets/'}")
    print("  2. Update settings.py with your dataset filename")
    print("  3. Run `brisk run` to train your first model")


def _create_brisk_dir(
    project_dir: pathlib.Path,
    project_name: str,
    problem_type: str
) -> None:
    """Create the .brisk/ metadata directory with sqlite db and project.json."""
    brisk_dir = project_dir / ".brisk"
    brisk_dir.mkdir(exist_ok=True)

    db_path = brisk_dir / "brisk.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.close()

    project_json = {
        "project_name": project_name,
        "project_path": str(project_dir.resolve()),
        "project_description": "",
        "project_type": problem_type,
        "datasets": [],
    }
    with open(
        brisk_dir / "project.json", "w", encoding="utf-8"
    ) as f:
        json.dump(project_json, f, indent=2)


def _create_settings(
    project_dir: pathlib.Path, problem_type: str
) -> None:
    """Write settings.py with problem-type-appropriate defaults."""
    if problem_type == "regression":
        default_algorithm = "linear"
    else:
        default_algorithm = "logistic"

    with open(
        project_dir / "settings.py", "w", encoding="utf-8"
    ) as f:
        f.write(f'''# settings.py
from brisk.configuration.configuration import Configuration
from brisk.configuration.configuration_manager import ConfigurationManager

def create_configuration() -> ConfigurationManager:
    config = Configuration(
        default_workflow = "workflow",
        default_algorithms = ["{default_algorithm}"],
    )

    config.add_experiment_group(
        name="group_name",
        # Add your dataset filenames from datasets/ here,
        # e.g. datasets=["my_data.csv"]
        datasets=[],
        workflow="workflow"
    )

    return config.build()
''')


def _create_algorithms(
    project_dir: pathlib.Path, problem_type: str
) -> None:
    """Write algorithms.py with only the relevant algorithm set."""
    if problem_type == "regression":
        constant = "REGRESSION_ALGORITHMS"
    else:
        constant = "CLASSIFICATION_ALGORITHMS"

    with open(
        project_dir / "algorithms.py", "w", encoding="utf-8"
    ) as f:
        f.write(f"""# algorithms.py
import brisk

ALGORITHM_CONFIG = brisk.AlgorithmCollection(
    *brisk.{constant}
)
""")


def _create_metrics(
    project_dir: pathlib.Path, problem_type: str
) -> None:
    """Write metrics.py with only the relevant metric set."""
    if problem_type == "regression":
        constant = "REGRESSION_METRICS"
    else:
        constant = "CLASSIFICATION_METRICS"

    with open(
        project_dir / "metrics.py", "w", encoding="utf-8"
    ) as f:
        f.write(f"""# metrics.py
import brisk

METRIC_CONFIG = brisk.MetricManager(
    *brisk.{constant}
)
""")


def _create_data(project_dir: pathlib.Path) -> None:
    """Write data.py with default DataManager settings."""
    with open(
        project_dir / "data.py", "w", encoding="utf-8"
    ) as f:
        f.write("""# data.py
from brisk.data.data_manager import DataManager

BASE_DATA_MANAGER = DataManager(
    test_size = 0.2
)
""")


def _create_evaluators(project_dir: pathlib.Path) -> None:
    """Write evaluators.py with the custom evaluator registration stub."""
    with open(
        project_dir / "evaluators.py", "w", encoding="utf-8"
    ) as f:
        f.write("""# evaluators.py
from brisk.evaluation.evaluators.registry import EvaluatorRegistry
from brisk import PlotEvaluator, MeasureEvaluator

def register_custom_evaluators(registry: EvaluatorRegistry, plot_settings) -> None:
    # registry.register(
    # Initialize an evaluator instance here to register
    # )
    pass
""")


def _create_workflow(
    workflows_dir: pathlib.Path, problem_type: str
) -> None:
    """Write workflow.py with problem-type-appropriate metric references."""
    if problem_type == "regression":
        metric = "MAE"
    else:
        metric = "accuracy"

    with open(
        workflows_dir / "workflow.py", "w", encoding="utf-8"
    ) as f:
        f.write(f'''# workflow.py
from brisk.training.workflow import Workflow

class MyWorkflow(Workflow):
    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        self.model.fit(self.X_train, self.y_train)
        self.evaluate_model_cv(
            self.model, self.X_train, self.y_train,
            ["{metric}"], "pre_tune_score"
        )
        tuned_model = self.hyperparameter_tuning(
            self.model, "grid", self.X_train, self.y_train, "{metric}",
            kf=5, num_rep=3, n_jobs=-1
        )
        self.evaluate_model(
            tuned_model, self.X_test, self.y_test,
            ["{metric}"], "post_tune_score"
        )
        self.plot_learning_curve(tuned_model, self.X_train, self.y_train)
        self.save_model(tuned_model, "tuned_model")
''')


@cli.command()
@click.option("--port", type=int, default=8050, help="Port for the UI server.")
@click.option(
    "--create", "create_mode", is_flag=True,
    help="Launch in create mode for new project setup."
)
@click.option(
    "--no-browser", is_flag=True,
    help="Don't open the browser automatically."
)
def ui(port: int, create_mode: bool, no_browser: bool) -> None:
    """Launch the brisk web UI for the current project."""
    try:
        import brisk_ui  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("Error: brisk-ui is not installed.")
        print("Install it with:")
        print("  pip install briskui")
        print("  # or: pip install brisk-ml[ui]")
        return

    if create_mode:
        project_path = pathlib.Path.cwd()
        project_path.mkdir(parents=True, exist_ok=True)
        print(f"Create mode: projects will be created in {project_path}")
    else:
        try:
            project_path = project.find_project_root()
        except FileNotFoundError:
            print("Error: Not inside a brisk project directory.")
            print("Run this command from a brisk project, or use --create:")
            print("  brisk ui --create")
            return

        db_path = project_path / ".brisk" / "brisk.sqlite"
        if not db_path.exists():
            print(f"Error: No brisk.sqlite found at {project_path}/.brisk/")
            print("Use --create to start a new project:")
            print("  brisk ui --create")
            return

    open_browser = not no_browser
    print(f"Starting brisk UI on port {port}...")
    print(f"Project: {project_path}")
    brisk_ui.start_server(
        project_path=project_path,
        port=port,
        create_mode=create_mode,
        open_browser=open_browser,
    )


@cli.command()
def migrate() -> None:
    """Migrate a project from .briskconfig to the new .brisk/ format.

    Converts a legacy Brisk project that uses .briskconfig to the new
    .brisk/ directory structure containing brisk.sqlite and project.json.
    The old .briskconfig file is removed after migration.
    """
    current = pathlib.Path.cwd()
    brisk_dir = current / ".brisk"
    config_path = current / ".briskconfig"

    if brisk_dir.is_dir() and not config_path.exists():
        print("This project already uses the new .brisk/ format.")
        return

    if not config_path.exists():
        print("Error: No .briskconfig found in the current directory.")
        print("Run this command from the root of a brisk project.")
        return

    if brisk_dir.is_dir() and config_path.exists():
        response = input(
            "Both .brisk/ and .briskconfig exist. Overwrite .brisk/ "
            "contents? [y/N]: "
        )
        if response.strip().lower() != "y":
            print("Migration cancelled.")
            return

    project.migrate_project(current)
    print(f"Project migrated successfully in: {current}")
    print("  Created .brisk/brisk.sqlite")
    print("  Created .brisk/project.json")
    print("  Removed .briskconfig")


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


@cli.command()
@click.option(
    "-d",
    "--dataset",
    required=True,
    help="Dataset filename in the project's datasets directory."
)
@click.option(
    "-o",
    "--output",
    default=None,
    help=(
        "Directory for preprocessed CSVs. Defaults to "
        "'preprocessed' under the project root."
    )
)
@click.option(
    "--table",
    "table_name",
    default=None,
    help="Table name when the dataset is a SQLite database."
)
@click.option(
    "--categorical-features",
    default=None,
    help="Comma-separated categorical column names."
)
@click.option(
    "--split-index",
    type=int,
    default=None,
    help="Write only this split index. Default writes every split."
)
def preprocess(
    dataset: str,
    output: Optional[str],
    table_name: Optional[str],
    categorical_features: Optional[str],
    split_index: Optional[int]
) -> None:
    """Apply preprocessing to a dataset without running training.

    Loads BASE_DATA_MANAGER from the project's data.py and applies the
    same split-then-preprocess pipeline used during experiments. Writes
    train and test CSVs so you can inspect preprocessed data without
    training models.

    Parameters
    ----------
    dataset : str
        Dataset filename in the project's datasets directory
    output : str, optional
        Directory for preprocessed CSVs. Defaults to 'preprocessed'
        under the project root
    table_name : str, optional
        Table name when the dataset is a SQLite database
    categorical_features : str, optional
        Comma-separated categorical column names
    split_index : int, optional
        Write only this split index. If omitted, every split is written

    Raises
    ------
    FileNotFoundError
        If the project root or dataset file is not found
    ValueError
        If preprocessing configuration is invalid
    """
    try:
        project_root = project.find_project_root()
        if project_root not in sys.path:
            sys.path.insert(0, str(project_root))
        _preprocess_dataset(
            project_root, dataset, output, table_name,
            categorical_features, split_index
        )
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
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
