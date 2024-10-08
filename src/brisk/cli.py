import importlib
import os
import sys

import click

# from brisk.training import TrainingManager

@click.group()
def cli():
    """Brisk Command Line Interface"""
    pass


@cli.command()
@click.option("-n", "--project_name", required=True, help="Name of the project directory.")
def create(project_name):
    """Create a new project directory."""
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)

    with open(os.path.join(project_dir, '.briskconfig'), 'w') as f:
        f.write(f"project_name={project_name}\n")

    with open(os.path.join(project_dir, 'settings.py'), 'w') as f:
        f.write("# settings.py\n")

    with open(os.path.join(project_dir, 'algorithms.py'), 'w') as f:
        f.write("""# algorithms.py
import brisk
                
ALGORITHM_CONFIG = {}        
""")

    with open(os.path.join(project_dir, 'metrics.py'), 'w') as f:
        f.write("""# metrics.py
import brisk
                
METRIC_CONFIG = brisk.MetricManager({})                       
""")

    with open(os.path.join(project_dir, 'splitter.py'), 'w') as f:
        f.write("""# splitter.py
from brisk.data.DataManager import DataManager                

SPLITTER = DataManager(
    test_size = 0.2,
    n_splits = 5
)              
""")

    with open(os.path.join(project_dir, 'training.py'), 'w') as f:
        f.write("""# training.py
from brisk.training.TrainingManager import TrainingManager
from .algorithms import ALGORITHM_CONFIG
from .metrics import METRIC_CONFIG
from .splitter import SPLITTER
                
# Define the TrainingManager for experiments
manager = TrainingManager(
    method_config=ALGORITHM_CONFIG,
    metric_config=METRIC_CONFIG,
    splitter=SPLITTER
)                 
""")
    
    datasets_dir = os.path.join(project_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    workflows_dir = os.path.join(project_dir, 'workflows')
    os.makedirs(workflows_dir, exist_ok=True)

    with open(os.path.join(workflows_dir, 'workflow.py'), 'w') as f:
        f.write("""# workflow.py
# Define the workflow for training and evaluating models

from brisk.training import Workflow

class MyWorkflow(Workflow):
    def workflow(self):
        pass           
""")

    print(f"A new project was created in: {project_dir}")


@cli.command()
@click.option('-w', '--workflow', required=True, help='Specify the workflow file (without .py) in workflows/')
@click.argument('extra_args', nargs=-1)
def run(workflow, extra_args):
    """Run experiments using the specified workflow."""
    
    extra_arg_dict = parse_extra_args(extra_args)   

    try:
        project_root = find_project_root()

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        manager = load_module_object(project_root, "training.py", "manager")

        WORKFLOW_CONFIG = load_module_object(
            project_root, "settings.py", "WORKFLOW_CONFIG"
            )

        workflow_module = importlib.import_module(f'workflows.{workflow}')
        workflow_class = getattr(workflow_module, 'MyWorkflow')

        manager.run_experiments(
            workflow=workflow_class, workflow_config=WORKFLOW_CONFIG,
            **extra_arg_dict
            )      

    except FileNotFoundError as e:
        print(f"Error: {e}")

    except (ImportError, AttributeError) as e:
        print(f"Error loading workflow: {workflow}. Error: {str(e)}")
        return

def parse_extra_args(extra_args):
    arg_dict = {}
    for arg in extra_args:
        key, value = arg.split('=')
        arg_dict[key] = value
    return arg_dict


def find_project_root(start_path: str = os.getcwd()) -> str:
    """Search for the .briskconfig file starting from the given directory and moving up the hierarchy.
    
    Args:
        start_path (str): Directory to start searching from (defaults to current working directory).
    
    Returns:
        str: The project root directory containing the .briskconfig file.
    
    Raises:
        FileNotFoundError: If .briskconfig is not found in the directory tree.
    """
    current_dir = start_path
    
    while current_dir != os.path.dirname(current_dir):  # Stop when reaching the root
        if os.path.isfile(os.path.join(current_dir, ".briskconfig")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    raise FileNotFoundError(
        ".briskconfig not found. Please run the command from a project directory or specify the project path."
        )


def load_module_object(
    project_root: str, 
    module_filename: str, 
    object_name: str, 
    required: bool = True
) -> object:
    """
    Dynamically loads an object (e.g., a class, instance, or variable) from a specified module file.

    Args:
        project_root (str): Path to the project's root directory.
        module_filename (str): The name of the module file (e.g., 'training.py', 'settings.py').
        object_name (str): The name of the object to retrieve (e.g., 'manager', 'WORKFLOW_CONFIG').
        required (bool): Whether to raise an error if the object is not found. Defaults to True.

    Returns:
        object: The requested object from the module.

    Raises:
        AttributeError: If the object is not found in the module and required is True.
        FileNotFoundError: If the module file is not found.
    """
    module_path = os.path.join(project_root, module_filename)

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"{module_filename} not found in {project_root}")


    module_name = os.path.splitext(module_filename)[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    spec.loader.exec_module(module)

    if hasattr(module, object_name):
        return getattr(module, object_name)
    elif required:
        raise AttributeError(f"The object '{object_name}' is not defined in {module_filename}")
    else:
        return None
    

if __name__ == '__main__':
    cli()
