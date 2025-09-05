"""Define helper functions used by CLI commands."""
from typing import Union
import json
import os
from pathlib import Path

from sklearn import datasets

from brisk.services import initialize_services, get_services, io
from brisk import version
from brisk.training.training_manager import TrainingManager
from brisk.configuration import configuration
from brisk.cli.environment import EnvironmentManager, VersionMatch

def _run_from_project(project_root, verbose, create_report, results_dir):
    try:
        print(
            "Begining experiment creation. "
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

        manager = TrainingManager(
            metric_config=metric_config,
            config_manager=config_manager
        )

        manager.run_experiments(
            create_report=create_report
        )

    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
        print(f'Error: {str(e)}')
        return


def _run_from_config(project_root, verbose, create_report, results_dir, config_file):
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
        env_manager = EnvironmentManager(Path(project_root))
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
    manager = TrainingManager(metric_config, config_manager)

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
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'diabetes': datasets.load_diabetes,
        'linnerud': datasets.load_linnerud
    }
    if name in datasets_map:
        return datasets_map[name]()
    else:
        return None


def _validate_dataset(dataset_path, metadata):
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
            f"Number of columns for {dataset_path} do not match expected columns "
            f"{metadata["num_features"]}"
        )

    current_features = list(df.columns)
    if current_features.sort() != metadata["feature_names"].sort():
        raise ValueError(
            f"Feature name for {dataset_path} do not match expected features "
            f"{metadata["feature_names"]}"
        )


def _handle_incompatible(config_file, differences, env_manager):
    print("\n" + "="*60)
    print("ENVIRONMENT COMPATIBILITY WARNING")
    print("="*60)
    
    critical_status = [VersionMatch.MISSING, VersionMatch.INCOMPATIBLE]
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
        print("   (Note: Critical packages now require major.minor version compatibility)")
        for diff in critical_diffs:
            print(f"   {str(diff)}")
    
    print("\nResults may differ significantly from the original run!")
    print("\n🔧 To recreate the original environment:")
    print(f"   brisk export-env {config_file} --output requirements.txt")
    print(f"   python -m venv brisk_env && source brisk_env/bin/activate")
    print(f"   pip install -r requirements.txt")
    print(f"   brisk rerun {config_file}")
    
    print("\nFor detailed comparison:")
    print(f"   brisk check-env {config_file} --verbose")
    
    print("\n" + "="*60)
