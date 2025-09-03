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

    except FileNotFoundError as e:
        print(f'Error: {e}')

    except (ImportError, AttributeError, ValueError) as e:
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

    # TODO: 0. Verify brisk version
    if configs["package_version"] != version.__version__:
        raise RuntimeError(
            "Configuration file was created using Brisk version "
            f"{configs["package_version"]} but Brisk version "
            f"{version.__version__} was detected."
        )

    # TODO 1. initalize services
    initialize_services(
        results_dir, verbose=verbose, mode="coordinate", rerun_config=configs
    )
    services = get_services()

    # TODO 2. load algorithm config
    algo_config = services.io.load_algorithms(project_root / "algorithms.py")

    # TODO 3. load metric config
    metric_config = services.io.load_metric_config(project_root / "metrics.py")

    # TODO 4. Setup Configuration instance
    configuration_args = services.rerun.get_configuration_args()
    config = configuration.Configuration(**configuration_args)
    experiment_groups = services.rerun.get_experiment_groups()
    for group in experiment_groups:
        config.add_experiment_group(**group)

    # TODO 5. Create ConfigurationManger instance
    # TODO 6. setup base data manager
    # TODO 7. Setup workflow files
    config_manager = config.build()

# TODO 8. Initalize TrainingManager and call run_experiemnts
# TODO 9. Setup evaluator file
# TODO 10. Verify data files
# TODO 11. Create environment for "env" and use this to run

    print(f"AlgorithmCollection: type={type(algo_config)}")
    print(f"MetricManager: type={type(metric_config)}")
    print(f"Configuration: type={type(config)}")
    print(f"Plotnine Theme: type={type(config.plot_settings.theme)}")
    print(f"ConfigurationManager: type={type(config_manager)}")
    print("\nFinished running from config file!")


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
