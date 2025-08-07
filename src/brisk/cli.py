"""Command-line interface for the Brisk framework.

This module provides a CLI for managing machine learning experiments with Brisk.
It includes commands for creating new projects, running experiments, and loading
data.

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

Examples
--------
Create a new project:
    $ brisk create -n my_project

Run an experiment:
    $ brisk run -w my_workflow

Load a dataset:
    $ brisk load_data --dataset iris --dataset_name my_iris
"""
import importlib
import inspect
import os
import sys
from typing import Optional, Union
from datetime import datetime

import click
import pandas as pd
from sklearn import datasets

from brisk.training.workflow import Workflow
from brisk.configuration import project
from brisk.services import initialize_services, shutdown_services

@click.group()
def cli():
    """Main entry point for Brisk's command line interface."""
    pass


@cli.command()
@click.option(
    '-n',
    '--project_name',
    required=True,
    help='Name of the project directory.'
)
def create(project_name: str) -> None:
    """Create a new project directory with template files.

    Parameters
    ----------
    project_name : str
        Name of the project directory to create

    Notes
    -----
    Creates the following structure:
    - .briskconfig : Project configuration file
    - settings.py : Configuration settings
    - algorithms.py : Algorithm definitions
    - metrics.py : Metric definitions
    - data.py : Data management setup
    - training.py : Training manager setup
    - workflows/ : Directory for workflow definitions
    - datasets/ : Directory for data storage
    """
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)

    with open(
        os.path.join(project_dir, '.briskconfig'), 'w', encoding='utf-8') as f:
        f.write(f"project_name={project_name}\n")

    with open(
        os.path.join(project_dir, 'settings.py'), 'w', encoding='utf-8') as f:
        f.write("""# settings.py
from brisk.configuration.configuration import Configuration, ConfigurationManager

def create_configuration() -> ConfigurationManager:
    config = Configuration(
        default_algorithms = ["linear"],
    )

    config.add_experiment_group(
        name="group_name",
    )
                
    return config.build()
""")

    with open(
        os.path.join(project_dir, 'algorithms.py'), 'w', encoding='utf-8') as f:
        f.write("""# algorithms.py
import brisk
                
ALGORITHM_CONFIG = brisk.AlgorithmCollection(
    brisk.AlgorithmWrapper(),
)        
""")

    with open(
        os.path.join(project_dir, 'metrics.py'), 'w', encoding='utf-8') as f:
        f.write("""# metrics.py
import brisk
                
METRIC_CONFIG = brisk.MetricManager(
    brisk.MetricWrapper()
)                   
""")

    with open(
        os.path.join(project_dir, 'data.py'), 'w', encoding='utf-8') as f:
        f.write("""# data.py
from brisk.data.data_manager import DataManager                

BASE_DATA_MANAGER = DataManager(
    test_size = 0.2
)              
""")

    with open(
        os.path.join(project_dir, 'training.py'), 'w', encoding='utf-8') as f:
        f.write("""# training.py
from brisk.training.training_manager import TrainingManager
from metrics import METRIC_CONFIG
from settings import create_configuration
                                
config = create_configuration()

# Define the TrainingManager for experiments
manager = TrainingManager(
    metric_config=METRIC_CONFIG,
    config_manager=config
)
""")

    datasets_dir = os.path.join(project_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    workflows_dir = os.path.join(project_dir, 'workflows')
    os.makedirs(workflows_dir, exist_ok=True)

    with open(
        os.path.join(workflows_dir, 'workflow.py'), 'w', encoding='utf-8') as f:
        f.write("""# workflow.py
# Define the workflow for training and evaluating models

from brisk.training.workflow import Workflow

class MyWorkflow(Workflow):
    def workflow(self):
        pass           
""")

    print(f"A new project was created in: {project_dir}")


@cli.command()
@click.option(
    '-w',
    '--workflow',
    required=True,
    help='Specify the workflow file (without .py) in workflows/'
)
@click.option(
    '-n',
    '--results_name',
    default=None,
    help='The name of the results directory.'
)
@click.option(
    '--disable_report',
    is_flag=True,
    default=False,
    help='Disable the creation of an HTML report.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Change the verbosity of the logger.'
)
def run(
    workflow: str,
    results_name: Optional[str],
    disable_report: bool,
    verbose: bool
) -> None:
    """Run experiments using the specified workflow.

    Parameters
    ----------
    workflow : str
        Name of the workflow file (without .py extension)
    results_name : str, optional
        Custom name for results directory
    disable_report : bool, default=False
        Whether to disable HTML report generation
    verbose : bool, default=False
        Whether to enable verbose logging

    Raises
    ------
    FileNotFoundError
        If project root or workflow file not found
    AttributeError
        If workflow class not found or multiple workflows defined
    """
    create_report = not disable_report
    try:
        project_root = project.find_project_root()

        if project_root not in sys.path:
            sys.path.insert(0, str(project_root))

        if not results_name:
            results_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            results_dir = os.path.join("results", results_name)
        else:
            results_dir = os.path.join("results", results_name)
        if os.path.exists(results_dir):
            raise FileExistsError(
                f"Results directory '{results_dir}' already exists."
            )
        os.makedirs(results_dir, exist_ok=False)

        algorithm_config = load_module_object(
            project_root, 'algorithms.py', 'ALGORITHM_CONFIG'
        )
        initialize_services(algorithm_config, results_dir, verbose=verbose)

        manager = load_module_object(project_root, 'training.py', 'manager')

        workflow_module = importlib.import_module(f'workflows.{workflow}')
        workflow_classes = [
            obj for name, obj in inspect.getmembers(workflow_module)
            if inspect.isclass(obj)
            and issubclass(obj, Workflow)
            and obj is not Workflow
        ]

        if len(workflow_classes) == 0:
            raise AttributeError(f'No Workflow subclass found in {workflow}.py')
        elif len(workflow_classes) > 1:
            raise AttributeError(
                f'Multiple Workflow subclasses found in {workflow}.py. '
                'There can only be one Workflow per file.'
                )

        workflow_class = workflow_classes[0]

        manager.run_experiments(
            workflow=workflow_class,
            # results_name=results_name,
            create_report=create_report
        )

        shutdown_services()

    except FileNotFoundError as e:
        print(f"Error: {e}")

    except (ImportError, AttributeError) as e:
        print(f"Error loading workflow: {workflow}. Error: {str(e)}")
        return


@cli.command()
@click.option(
    '--dataset', 
    type=click.Choice(
        ['iris', 'wine', 'breast_cancer', 'diabetes', 'linnerud']
        ),
    required=True,
    help=(
        'Name of the sklearn dataset to load. Options are iris, wine, '
        'breast_cancer, diabetes, or linnerud.'
    )
)
@click.option(
    '--dataset_name',
    type=str,
    default=None,
    help='Name to save the dataset as.'
)
def load_data(dataset: str, dataset_name: Optional[str] = None) -> None:
    """Load a scikit-learn dataset into the project.

    Parameters
    ----------
    dataset : {'iris', 'wine', 'breast_cancer', 'diabetes', 'linnerud'}
        Name of the dataset to load
    dataset_name : str, optional
        Custom name for the saved dataset file

    Notes
    -----
    Saves the dataset as a CSV file in the project's datasets directory.
    """
    try:
        project_root = project.find_project_root()
        datasets_dir = os.path.join(project_root, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)

        data = load_sklearn_dataset(dataset)
        if data is None:
            print(
                f"Dataset '{dataset}' not found in sklearn. Options are iris, "
                'wine, breast_cancer, diabetes or linnerud.'
                )
            return
        X = data.data # pylint: disable=C0103
        y = data.target

        feature_names = (
            data.feature_names
            if hasattr(data, 'feature_names')
            else [f'feature_{i}' for i in range(X.shape[1])]
            )
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        dataset_filename = dataset_name if dataset_name else dataset
        csv_path = os.path.join(datasets_dir, f"{dataset_filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")


@cli.command()
@click.option(
    '--data_type',
    type=click.Choice(['classification', 'regression']),
    required=True,
    help='Type of the synthetic dataset.'
)
@click.option(
    '--n_samples',
    type=int,
    default=100,
    help='Number of samples for synthetic data.'
)
@click.option(
    '--n_features',
    type=int,
    default=20,
    help='Number of features for synthetic data.'
)
@click.option(
    '--n_classes',
    type=int,
    default=2,
    help='Number of classes for classification data.'
)
@click.option(
    '--random_state',
    type=int,
    default=42,
    help='Random state for reproducibility.'
)
@click.option(
    '--dataset_name',
    type=str,
    default='synthetic_dataset',
    help='Name of the dataset file to be saved.'
)
def create_data(
    data_type: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    random_state: int,
    dataset_name: str
    ):
    """Create synthetic data and add it to the project.

    Parameters
    ----------
    data_type : {'classification', 'regression'}
        Type of dataset to generate
    n_samples : int, default=100
        Number of samples to generate
    n_features : int, default=20
        Number of features to generate
    n_classes : int, default=2
        Number of classes (classification only)
    random_state : int, default=42
        Random seed for reproducibility
    dataset_name : str, default='synthetic_dataset'
        Name for the output file

    Notes
    -----
    For classification:
        - 80% informative features
        - 20% redundant features
        - No repeated features

    For regression:
        - 80% informative features
        - 0.1 noise level
    """
    try:
        project_root = project.find_project_root()
        datasets_dir = os.path.join(project_root, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)

        if data_type == 'classification':
            X, y = datasets.make_classification( # pylint: disable=C0103
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.8),
                n_redundant=int(n_features * 0.2),
                n_repeated=0,
                n_classes=n_classes,
                random_state=random_state
            )
        elif data_type == 'regression':
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
        df['target'] = y
        csv_path = os.path.join(datasets_dir, f"{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Synthetic dataset saved to {csv_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")


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
            f"{module_filename} not found in {project_root}"
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
            f"The object '{object_name}' is not defined in {module_filename}"
            )
    else:
        return None


if __name__ == '__main__':
    cli()
