"""Define helper functions used by CLI commands."""
import importlib
from typing import Union
import json
import os
import sys

from sklearn import datasets

from brisk.services import initialize_services
from brisk import version


def _run_from_project(project_root, verbose, create_report, results_dir):
    try:
        print(
            "Begining experiment creation. "
            f"The results will be saved to {results_dir}"
        )

        algorithm_config = load_module_object(
            project_root, 'algorithms.py', 'ALGORITHM_CONFIG'
        )
        metric_config = load_module_object(
            project_root, "metrics.py", "METRIC_CONFIG"
        )
        initialize_services(
            algorithm_config, metric_config, results_dir, verbose=verbose
        )

        manager = load_module_object(project_root, 'training.py', 'manager')

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
            config = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error in config_file handling: {e}")
        exit() # NOTE: temp while developing

# TODO: 00. Verify brisk version
    if config["package_version"] != version.__version__:
        raise RuntimeError(
            "Configuration file was created using Brisk version "
            f"{config["package_version"]} but Brisk version "
            f"{version.__version__} was detected."
        )

# TODO : 0. Verify data files
        

# TODO: 01. Setup workflow files
# TODO: 02. Setup evaluator file
# TODO: 1. load algorithm config
# TODO 2. load metric config
# TODO 3. initalize services
# TODO 3.5. setup base data manager
# TODO 4. Setup Configuration instance
# TODO 5. Create ConfigurationManger instance
# TODO 6. Initalize TrainingManager and call run_experiemnts
# TODO 7. Create environment for "env" and use this to run


        # except Exception as e:
        #     print(f"Error in config_file handling: {e}")
        #     exit() # NOTE: temp while developing


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


def load_module_object(
    project_root: str,
    module_filename: str,
    object_name: str,
    required: bool = True
) -> Union[object, None]:
    """
    Dynamically loads an object from a specified module file.

    Parameters
    ----------
    project_root : str
        Path to project root directory
    module_filename : str
        Name of the module file
    object_name : str
        Name of object to load
    required : bool, default=True
        Whether to raise error if object not found

    Returns
    -------
    object or None
        Loaded object or None if not found and not required

    Raises
    ------
    FileNotFoundError
        If module file not found
    AttributeError
        If required object not found in module
    """
    module_path = os.path.join(project_root, module_filename)

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f'{module_filename} not found in {project_root}'
        )

    module_name = os.path.splitext(module_filename)[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    if hasattr(module, object_name):
        return getattr(module, object_name)
    elif required:
        raise AttributeError(
            f'The object \'{object_name}\' is not defined in {module_filename}'
        )
    else:
        return None
