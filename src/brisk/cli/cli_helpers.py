"""Define helper functions used by CLI commands."""
from typing import Union
import json
import os

from sklearn import datasets

from brisk.services import initialize_services, get_services
from brisk import version
from brisk.training.training_manager import TrainingManager
from brisk.configuration import configuration

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
        exit() # NOTE: temp while developing

    if configs["package_version"] != version.__version__:
        raise RuntimeError(
            "Configuration file was created using Brisk version "
            f"{configs["package_version"]} but Brisk version "
            f"{version.__version__} was detected."
        )

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

    # TODO 10. Verify data files
    # TODO 11. Create environment for "env" and use this to run
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
