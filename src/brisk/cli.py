import os
import importlib

import click

from brisk.training import TrainingManager

@click.group()
def cli():
    """Brisk Command Line Interface"""
    pass


@cli.command()
@click.option("-n", "--project_name", required=True, help="Name of the project directory.")
def create(project_name):
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)

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
                
METRIC_CONFIG = {}                       
""")

    with open(os.path.join(project_dir, 'splitter.py'), 'w') as f:
        f.write("""# splitter.py
from brisk.data import DataSplitter                

SPLITTER = DataSplitter(
    test_size = 0.2,
    n_splits = 5,
    split_method = "",
)              
""")

    with open(os.path.join(project_dir, 'training.py'), 'w') as f:
        f.write("""# training.py
from brisk.training import TrainingManager

# Define the TrainingManager for experiments
manager = TrainingManager()             
""")
    
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

    print(f"New ML project created in: {project_dir}")


@cli.command()
@click.option('-w', '--workflow', required=True, help='Specify the workflow file (without .py) in workflows/')
@click.argument('extra_args', nargs=-1)
def run(workflow, extra_args):
    """Run experiments using the specified workflow."""
    
    extra_arg_dict = parse_extra_args(extra_args)   

    try:
        workflow_module = importlib.import_module(f'workflows.{workflow}')
        workflow_class = getattr(workflow_module, 'MyWorkflow')
    except (ImportError, AttributeError) as e:
        print(f"Error loading workflow: {workflow}. Error: {str(e)}")
        return

    manager = TrainingManager()
    manager.run_experiments(workflow=workflow_class, **extra_arg_dict)


def parse_extra_args(extra_args):
    arg_dict = {}
    for arg in extra_args:
        key, value = arg.split('=')
        arg_dict[key] = value
    return arg_dict


if __name__ == '__main__':
    cli()
