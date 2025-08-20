"""Configuration container for experiment groups.

This module defines the ExperimentGroup class, which serves as a container for
configurations related to a group of experiments within the Brisk framework.
Default values for AlgorithmWrappers and DataManagers can be overidden within an
ExperimentGroup.

Examples
--------
>>> from brisk.configuration.experiment_group import ExperimentGroup
>>> group = ExperimentGroup(
...     name="baseline",
...     datasets=["data.csv"],
...     algorithms=["linear"],
...     data_config={"test_size": 0.25}
... )
>>>
"""

import dataclasses
import inspect
import pathlib
import textwrap
from typing import Dict, Any, List, Optional, Tuple

from brisk.data import data_manager
from brisk.configuration import project

@dataclasses.dataclass
class ExperimentGroup:
    """Container for experiment group configuration.

    Parameters
    ----------
    name : str
        Unique identifier for the experiment group
    datasets : list
        List of dataset filenames relative to project's datasets directory
    data_config : dict, optional
        Arguments for DataManager
    algorithms : list, optional
        List of algorithms to use
    algorithm_config : dict, optional
        Arguments to apply to AlgorithmWrappers
    description : str, optional
        Description for the experiment group
    workflow_args : dict, optional
        Arguments to pass to workflow as attributes
    """
    name: str
    datasets: List[str | Tuple[str, str]]
    data_config: Optional[Dict[str, Any]] = None
    algorithms: Optional[List[str]] = None
    algorithm_config: Optional[Dict[str, Dict[str, Any]]] = None
    description: Optional[str] = ""
    workflow_args: Optional[Dict[str, Any]] = None

    @property
    def dataset_paths(self) -> List[Tuple[pathlib.Path, str | None]]:
        """Get full paths to datasets relative to project root.

        Returns
        -------
        list of (pathlib.Path, str or None)
            Each tuple contains:

            * path : pathlib.Path
                Full path to the dataset file
            * table_name : str or None
                Name of table for database files, None for regular files

        Raises
        ------
        FileNotFoundError
            If project root (.briskconfig) cannot be found
        """
        project_root = project.find_project_root()
        datasets_dir = project_root / "datasets"
        return [
            (datasets_dir / dataset[0], dataset[1])
            if isinstance(dataset, tuple)
            else (datasets_dir / dataset, None)
            for dataset in self.datasets
        ]

    def __post_init__(self):
        """Validate experiment group configuration after initialization.

        Performs validation checks on:
        - Name format
        - Dataset existence
        - Algorithm configuration consistency
        - DataManager configuration parameters

        Raises
        ------
        ValueError
            If any validation check fails
        FileNotFoundError
            If datasets cannot be found
        """
        self._validate_name()
        self._validate_datasets()
        self._validate_algorithm_config()
        self._validate_data_config()
        self._validate_description()
        self._validate_workflow_args()

    def _validate_name(self):
        """Validate experiment group name.

        Ensures name is a non-empty string.

        Raises
        ------
        ValueError
            If name is empty or not a string
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Experiment group name must be a non-empty string")

    def _validate_datasets(self):
        """Validate dataset specifications.

        Checks:
        - At least one dataset is specified
        - All specified datasets exist in project's datasets directory

        Raises
        ------
        ValueError
            If no datasets are specified
        FileNotFoundError
            If any dataset file cannot be found
        """
        if not self.datasets:
            raise ValueError("At least one dataset must be specified")

        for dataset, path in zip(self.datasets, self.dataset_paths):
            if not path[0].exists():
                raise FileNotFoundError(
                    f"Dataset not found: {dataset}\n"
                    f"Expected location: {path}"
                )

    def _validate_algorithm_config(self):
        """Validate algorithm configuration.

        Ensures all algorithms in algorithm_config are present in the algorithms
        list.

        Raises
        ------
        ValueError
            If algorithm_config contains undefined algorithms
        """
        if self.algorithm_config:

            flat_algorithms = [
                algo for sublist in self.algorithms
                if isinstance(sublist, list)
                for algo in sublist
            ] if any(
                isinstance(x, list) for x in self.algorithms
            ) else self.algorithms

            invalid_algorithms = (
                set(self.algorithm_config.keys()) - set(flat_algorithms)
            )
            if invalid_algorithms:
                raise ValueError(
                    f"Algorithm config contains algorithms not in the list of "
                    f"algorithms: {invalid_algorithms}"
                )

    def _validate_data_config(self):
        """Validate DataManager configuration parameters.

        Ensures all parameters in data_config are valid DataManager parameters.
        Uses DataManager's __init__ signature to determine valid parameters.

        Raises
        ------
        ValueError
            If data_config contains invalid parameters
        """
        if self.data_config:
            valid_data_params = set(
                inspect.signature(
                    data_manager.DataManager.__init__
                ).parameters.keys()
            )
            valid_data_params.remove("self")

            invalid_params = set(self.data_config.keys()) - valid_data_params
            if invalid_params:
                raise ValueError(
                    f"Invalid DataManager parameters: {invalid_params}\n"
                    f"Valid parameters are: {valid_data_params}"
                )

    def _validate_description(self):
        """Validate and format experiment group description.

        Ensures description is a string and wraps text at 60 characters.

        Raises
        ------
        ValueError
            If description is not a string
        """
        if not isinstance(self.description, str):
            raise ValueError("Description must be a string")

        wrapped_description = textwrap.fill(
            self.description,
            width=60
        )
        self.description = wrapped_description

    def _validate_workflow_args(self):
        """Validate workflow arguments.

        Ensures workflow_args is a dictionary if provided.

        Raises
        ------
        ValueError
            If workflow_args is not a dictionary
        """
        if self.workflow_args and not isinstance(self.workflow_args, dict):
            raise ValueError("workflow_args must be a dict")
