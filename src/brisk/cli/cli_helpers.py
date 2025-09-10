"""Define helper functions used by CLI commands."""
from typing import Union, Dict, Any, List
import json
import os
import pathlib

from sklearn import datasets

from brisk.services import initialize_services, get_services, io
from brisk import version
from brisk.training import training_manager
from brisk.configuration import configuration
from brisk.cli import environment

def _run_from_project(
    project_root: pathlib.Path,
    verbose: bool,
    create_report: bool,
    results_dir: str
) -> None:
    """Run experiments from a project directory.

    Loads algorithms, metrics, and configuration from a project directory
    and executes the training workflow. This function is used when running
    experiments from project files.

    Parameters
    ----------
    project_root : pathlib.Path
        Path to the project root directory containing algorithms.py,
        metrics.py, and settings.py files.
    verbose : bool
        Whether to enable verbose logging output.
    create_report : bool
        Whether to generate an HTML report after experiments complete.
    results_dir : str
        Directory path where experiment results will be saved.

    Notes
    -----
    This function expects the following files to exist in the project root:
    - algorithms.py: Contains algorithm definitions
    - metrics.py: Contains metric configurations
    - settings.py: Contains a create_configuration function

    Raises
    ------
    FileNotFoundError
        If required project files are missing.
    ImportError
        If there are issues importing project modules.
    AttributeError
        If required functions or objects are not found in modules.
    ValueError
        If there are configuration or data validation errors.
    """
    try:
        print(
            "Beginning experiment creation. "
            f"The results will be saved to {results_dir}"
        )

        initialize_services(
            results_dir, verbose=verbose, mode="capture", rerun_config=None
        )
        services = get_services()

        services.io.load_algorithms(project_root / "algorithms.py")
        metric_config = services.io.load_metric_config(
            project_root / "metrics.py"
        )

        create_configuration = services.io.load_module_object(
            project_root, "settings.py", "create_configuration"
        )

        config_manager = create_configuration()

        manager = training_manager.TrainingManager(
            metric_config=metric_config,
            config_manager=config_manager
        )

        manager.run_experiments(
            create_report=create_report
        )

    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
        print(f'Error: {str(e)}')


def _run_from_config(
    project_root: pathlib.Path,
    verbose: bool,
    create_report: bool,
    results_dir: str,
    config_file: str
) -> None:
    """Run experiments from a saved configuration file.

    Loads a previously saved experiment configuration and reruns the
    experiments with the same settings. Validates environment compatibility
    and dataset integrity before execution.

    Parameters
    ----------
    project_root : Path
        Path to the project root directory.
    verbose : bool
        Whether to enable verbose logging output.
    create_report : bool
        Whether to generate an HTML report after experiments complete.
    results_dir : str
        Directory path where experiment results will be saved.
    config_file : str
        Name of the results directory (without path) to load the configuration
        from.

    Notes
    -----
    The configuration file should be located at:
    {project_root}/results/{config_file}/run_config.json

    This function performs several validation checks:
    - Package version compatibility
    - Dataset file existence and integrity
    - Environment package compatibility

    Raises
    ------
    RuntimeError
        If the configuration was created with a different Brisk version.
    FileNotFoundError
        If the configuration file or dataset files are missing.
    ValueError
        If dataset validation fails or there are configuration errors.
    """
    try:
        config_path = os.path.join(
            project_root, "results", config_file, "run_config.json"
        )

        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error in config_file handling: {e}")
        return

    if configs["package_version"] != version.__version__:
        raise RuntimeError(
            "Configuration file was created using Brisk version "
            f"{configs["package_version"]} but Brisk version "
            f"{version.__version__} was detected."
        )

    for dataset_file, metadata in configs["datasets"].items():
        dataset_path = os.path.join(
            project_root, "datasets", dataset_file
        )
        _validate_dataset(dataset_path, metadata)

    saved_env = configs.get("env")
    if saved_env:
        env_manager = environment.EnvironmentManager(pathlib.Path(project_root))
        differences, is_compatible = env_manager.compare_environments(saved_env)

        if not is_compatible:
            _handle_incompatible(config_file, differences, env_manager)

            response = input("\nContinue anyway? (y/N): ").strip().lower()
            if response == "N":
                print("Rerun cancelled.")
                return

    initialize_services(
        results_dir, verbose=verbose, mode="coordinate", rerun_config=configs
    )
    services = get_services()

    services.io.load_algorithms(project_root / "algorithms.py")
    metric_config = services.io.load_metric_config(project_root / "metrics.py")
    configuration_args = services.rerun.get_configuration_args()
    config = configuration.Configuration(**configuration_args)
    experiment_groups = services.rerun.get_experiment_groups()
    for group in experiment_groups:
        config.add_experiment_group(**group)

    config_manager = config.build()
    manager = training_manager.TrainingManager(metric_config, config_manager)

    try:
        manager.run_experiments(create_report)
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
        print(f'Error: {str(e)}')
        return


def load_sklearn_dataset(name: str) -> Union[dict, None]:
    """Load a dataset from scikit-learn.

    Parameters
    ----------
    name : {'iris', 'wine', 'breast_cancer', 'diabetes', 'linnerud'}
        Name of the dataset to load

    Returns
    -------
    dict or None
        Loaded dataset object or None if not found
    """
    datasets_map = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
        "linnerud": datasets.load_linnerud
    }
    if name in datasets_map:
        return datasets_map[name]()
    return None


def _validate_dataset(dataset_path: str, metadata: Dict[str, Any]) -> None:
    """Validate dataset file against expected metadata.

    Checks that the dataset file exists and matches the expected structure
    including number of samples, features, and feature names.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file to validate.
    metadata : dict
        Dictionary containing expected dataset metadata with keys:
        - table_name: Name of the table in the dataset file
        - num_samples: Expected number of rows/samples
        - num_features: Expected number of columns/features
        - feature_names: List of expected feature column names

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the dataset dimensions or feature names don't match expectations.

    Notes
    -----
    This function loads the dataset using the IOService and performs
    validation to ensure data integrity for experiment reruns.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = io.IOService.load_data(dataset_path, metadata["table_name"])

    rows, cols = df.shape
    if rows != metadata["num_samples"]:
        raise ValueError(
            f"Number of rows for {dataset_path} do not match expected rows "
            f"{metadata["num_samples"]}"
        )
    if cols != metadata["num_features"]:
        raise ValueError(
            f"Number of columns for {dataset_path} do not match expected "
            f"columns {metadata["num_features"]}"
        )

    current_features = list(df.columns)
    if current_features.sort() != metadata["feature_names"].sort():
        raise ValueError(
            f"Feature name for {dataset_path} do not match expected features "
            f"{metadata["feature_names"]}"
        )


def _handle_incompatible(
    config_file: str,
    differences: List[environment.EnvironmentDiff],
    env_manager: environment.EnvironmentManager
) -> None:
    """Handle environment compatibility warnings and provide guidance.

    Displays warnings about environment differences and provides instructions
    for recreating the original environment or continuing with potential
    compatibility issues.

    Parameters
    ----------
    config_file : str
        Name of the configuration file being rerun.
    differences : list
        List of environment differences detected by EnvironmentManager.
    env_manager : EnvironmentManager
        Environment manager instance used for comparison.

    Notes
    -----
    This function identifies critical package differences and provides
    detailed instructions for:
    - Recreating the original environment
    - Exporting environment requirements
    - Checking detailed environment differences

    The function categorizes differences as critical if they involve:
    - Missing packages
    - Incompatible versions of critical packages
    - Python version changes
    """
    print("\n" + "="*60)
    print("ENVIRONMENT COMPATIBILITY WARNING")
    print("="*60)

    critical_status = [
        environment.VersionMatch.MISSING, environment.VersionMatch.INCOMPATIBLE
    ]
    critical_diffs = [
        d for d in differences
        if (
            d.status in critical_status
            and (
                d.package in env_manager.CRITICAL_PACKAGES or
                d.package == "python"
            ))]

    if critical_diffs:
        print("\nCritical package differences detected:")
        print(
            "   (Note: Critical packages now require major.minor version "
            "compatibility)"
        )
        for diff in critical_diffs:
            print(f"   {str(diff)}")

    print("\nResults may differ significantly from the original run!")
    print("\n  To recreate the original environment:")
    print(f"   brisk export-env {config_file} --output requirements.txt")
    print("   python -m venv brisk_env && source brisk_env/bin/activate")
    print("   pip install -r requirements.txt")
    print(f"   brisk rerun {config_file}")

    print("\nFor detailed comparison:")
    print(f"   brisk check-env {config_file} --verbose")

    print("\n" + "="*60)
