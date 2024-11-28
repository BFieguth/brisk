"""cli.py

This module provides the command-line interface (CLI) for the Brisk framework, 
which is designed to facilitate the training of machine learning models using 
scikit-learn. The CLI allows users to create new projects, run experiments, 
load datasets, and generate synthetic data.

Commands:
- `create`: Initializes a new project directory with configuration files.
- `run`: Executes experiments based on the specified workflow.
- `load_data`: Loads a dataset from scikit-learn into the project.
- `create_data`: Generates synthetic datasets for classification or regression 
                 tasks.

Usage:
To use the CLI, run the script directly or through a command line interface, 
specifying the desired command and its options.

Example:
    python cli.py create -n my_project
    python cli.py run -w my_workflow
    python cli.py load_data --dataset iris --dataset_name my_iris_data
"""
import importlib
import inspect
import os
import sys
from typing import Optional, Union

import click
import pandas as pd
from sklearn import datasets

from brisk.training.Workflow import Workflow

@click.group()
def cli():
    """Brisk Command Line Interface"""
    pass


@cli.command()
@click.option(
    '-n',
    '--project_name',
    required=True,
    help='Name of the project directory.'
)
def create(project_name: str) -> None:
    """Create a new project directory.

    This command initializes a new project directory with the specified name.
    It creates the necessary configuration files, including .briskconfig,
    settings.py, algorithms.py, metrics.py, data.py, training.py, and a
    workflow file.

    Args:
        project_name (str): The name of the project directory to create.

    Example:
        python cli.py create -n my_project
    """
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)

    with open(
        os.path.join(project_dir, '.briskconfig'), 'w', encoding='utf-8') as f:
        f.write(f"project_name={project_name}\n")

    with open(
        os.path.join(project_dir, 'settings.py'), 'w', encoding='utf-8') as f:
        f.write("""# settings.py
from brisk.configuration.Configuration import Configuration, ConfigurationManager

def create_configuration() -> ConfigurationManager:
    config = Configuration(
        default_algorithms = ["linear"],
    )

    config.add_experiment_group(
        name="group_name",
    )
                
    return config.build()
                
WORKFLOW_CONFIG = {

}
""")

    with open(
        os.path.join(project_dir, 'algorithms.py'), 'w', encoding='utf-8') as f:
        f.write("""# algorithms.py
import brisk
                
ALGORITHM_CONFIG = [
    brisk.AlgorithmWrapper()
]        
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
    test_size = 0.2,
    n_splits = 5
)              
""")

    with open(
        os.path.join(project_dir, 'training.py'), 'w', encoding='utf-8') as f:
        f.write("""# training.py
from brisk.training.TrainingManager import TrainingManager
from metrics import METRIC_CONFIG
from settings import create_configuration
                                
config = create_configuration()

# Define the TrainingManager for experiments
manager = TrainingManager(
    metric_config=METRIC_CONFIG,
    data_managers=config.data_managers,
    experiments=config.experiment_queue,
    logfile=config.logfile,
    output_structure=config.output_structure
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

from brisk.training.Workflow import Workflow

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
@click.argument('extra_args', nargs=-1)
def run(workflow: str, extra_args: tuple) -> None:
    """Run experiments using the specified workflow.

    This command executes experiments based on the specified workflow file.
    It loads the workflow class and runs the experiments defined within it,
    passing any additional arguments provided by the user.

    Args:
        workflow (str): The name of the workflow file (without .py) to run.
        extra_args (tuple): Additional arguments to pass to the workflow.

    Example:
        python cli.py run -w my_workflow --arg1=value1 --arg2=value2
    """
    extra_arg_dict = parse_extra_args(extra_args)

    try:
        project_root = find_project_root()

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        manager = load_module_object(project_root, 'training.py', 'manager')

        workflow_config = load_module_object(
            project_root, 'settings.py', 'WORKFLOW_CONFIG'
            )

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
            workflow=workflow_class, workflow_config=workflow_config,
            **extra_arg_dict
            )

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
    """Load a dataset from sklearn into the project.

    This command loads a specified dataset from scikit-learn and saves it
    as a CSV file in the project's datasets directory. If a dataset name
    is provided, it will be used as the filename; otherwise, the default
    dataset name will be used.

    Args:
        dataset (str): The name of the dataset to load.
        dataset_name (str, optional): The name to save the dataset as.

    Example:
        python cli.py load_data --dataset iris --dataset_name my_iris_data
    """
    try:
        project_root = find_project_root()
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

    This command generates synthetic datasets for either classification or
    regression tasks based on the specified parameters. The generated dataset
    is saved as a CSV file in the project's datasets directory.

    Args:
        data_type (str): The type of synthetic dataset to create 
        ('classification' or 'regression').
        n_samples (int): The number of samples to generate.
        n_features (int): The number of features for the dataset.
        n_classes (int): The number of classes for classification data.
        random_state (int): The random state for reproducibility.
        dataset_name (str): The name of the dataset file to be saved.

    Example:
        python cli.py create_data --data_type classification --n_samples 200 
        --n_features 10 --n_classes 3 --dataset_name my_synthetic_data
    """
    try:
        project_root = find_project_root()
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
    """Load a dataset from sklearn by name."""
    datasets_map = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'diabetes': datasets.load_diabetes,
        'linnerud': datasets.oad_linnerud
    }
    if name in datasets_map:
        return datasets_map[name]()
    else:
        return None


def parse_extra_args(extra_args: tuple) -> dict:
    arg_dict = {}
    for arg in extra_args:
        key, value = arg.split('=')
        arg_dict[key] = value
    return arg_dict


def find_project_root(start_path: str = os.getcwd()) -> str:
    """Search for the .briskconfig file starting from the given directory.
    
    Args:
        start_path (str): Directory to start searching from 
        (defaults to current working directory).
    
    Returns:
        str: The project root directory containing the .briskconfig file.
    
    Raises:
        FileNotFoundError: If .briskconfig is not found in the directory tree.
    """
    current_dir = start_path

    # Stop when reaching the root
    while current_dir != os.path.dirname(current_dir):
        if os.path.isfile(os.path.join(current_dir, '.briskconfig')):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    raise FileNotFoundError(
        '.briskconfig not found. Please run the command from a project '
        'directory or specify the project path.'
        )


def load_module_object(
    project_root: str,
    module_filename: str,
    object_name: str,
    required: bool = True
) -> Union[object, None]:
    """
    Dynamically loads an object from a specified module file.

    Args:
        project_root (str): Path to the project's root directory.
        module_filename (str): The name of the module file (e.g. 'training.py').
        object_name (str): The name of the object to retrieve 
        (e.g. 'WORKFLOW_CONFIG').
        required (bool): Whether to raise an error if the object is not found.
        Defaults to True.

    Returns:
        object: The requested object from the module.

    Raises:
        AttributeError: If the object is not found in the module and required 
        is True.
        FileNotFoundError: If the module file is not found.
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
