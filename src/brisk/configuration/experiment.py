"""Data for individual experiment runs.

This module defines the Experiment class, which represents a single experiment
within the Brisk framework. Each Experiment instance contains the information
needed for one model training run.

Examples
--------
>>> from pathlib import Path
>>> from sklearn.linear_model import LinearRegression
>>> from brisk.utility.algorithm_wrapper import AlgorithmWrapper
>>>
>>> experiment = Experiment(
...     group_name="baseline",
...     algorithms={"model": AlgorithmWrapper("linear", LinearRegression)},
...     dataset_path=Path("data/example.csv"),
...     workflow_args={}
... )
>>> print(experiment.name)
'baseline_linear'
"""
import dataclasses
import pathlib
from typing import Dict, Optional, List, Any

from brisk.configuration import algorithm_wrapper

@dataclasses.dataclass
class Experiment:
    """Configuration for a single experiment run.

    Parameters
    ----------
    group_name : str
        Name of the experiment group for organization
    algorithms : dict
        Dictionary of AlgorithmWrapper instances with standardized keys:

        * Single model : dict
            {"model": wrapper}
        * Multiple models : dict
            {"model": wrapper1, "model2": wrapper2, ...}
    dataset_path : Path or str
        Path to the dataset file
    workflow_args : dict
        Arguments to pass to the workflow
    table_name : str, optional
        Name of table for database files
    categorical_features : list of str, optional
        Names of categorical features in the dataset

    Attributes
    ----------
    name : str
        Full descriptive name combining group and algorithms
    dataset_name : str
        Name of the dataset with optional table name
    algorithm_kwargs : dict
        Dictionary of instantiated algorithm objects
    algorithm_names : list
        List of algorithm names
    workflow_attributes : dict
        Combined workflow and algorithm arguments
    """
    group_name: str
    algorithms: Dict[str, algorithm_wrapper.AlgorithmWrapper]
    dataset_path: pathlib.Path
    workflow_args: Dict[str, Any]
    table_name: Optional[str | None]
    categorical_features: Optional[List[str] | None]

    @property
    def name(self) -> str:
        """Generate full descriptive name for logging and debugging.

        Returns
        -------
        str
            Name combining group name and algorithm names
            Example: 'baseline_linear_ridge'
        """
        algo_names = "_".join(
            algo.name for algo in self.algorithms.values()
        )
        return f"{self.group_name}_{algo_names}"

    @property
    def dataset_name(self) -> str:
        """Get the dataset name with optional table name.

        Returns
        -------
        str
            Dataset stem with optional table name
            Example: 'data_table1' or 'data'
        """
        dataset_name = (
            f"{self.dataset_path.stem}_{self.table_name}"
            if self.table_name else self.dataset_path.stem
        )
        return dataset_name

    @property
    def algorithm_kwargs(self) -> dict:
        """Get dictionary of instantiated algorithms.

        Returns
        -------
        dict
            Mapping of keys to algorithm instances
            Example: {'model': LinearRegression()}
        """
        algorithm_kwargs = {
            key: algo.instantiate() for key, algo in self.algorithms.items()
        }
        return algorithm_kwargs

    @property
    def algorithm_names(self) -> list:
        """Get list of algorithm names.

        Returns
        -------
        list
            Names of all algorithms in this experiment
            Example: ['linear', 'ridge']
        """
        algorithm_names = [algo.name for algo in self.algorithms.values()]
        return algorithm_names

    @property
    def workflow_attributes(self) -> Dict[str, Any]:
        """Get combined workflow and algorithm arguments to pass to Workflow.

        Returns
        -------
        dict
            Union of workflow_args and algorithm_kwargs
        """
        workflow_attributes = self.workflow_args | self.algorithm_kwargs
        return workflow_attributes

    def __post_init__(self):
        """Validate experiment configuration after initialization.

        Performs the following validations:
        1. Converts dataset to Path if it's a string
        2. Validates group_name is a string
        3. Validates algorithms is a non-empty dictionary
        4. Validates model naming convention:
           - Single model must use key "model"
           - Multiple models must use keys "model1", "model2", etc.

        Raises
        ------
        ValueError
            If any validation fails:
            - If group_name is not a string
            - If algorithms is not a dictionary
            - If algorithms is empty
            - If model keys don't follow naming convention
        """
        if not isinstance(self.dataset_path, pathlib.Path):
            self.dataset_path = pathlib.Path(self.dataset_path)

        if not isinstance(self.group_name, str):
            raise ValueError("Group name must be a string")

        if not isinstance(self.algorithms, dict):
            raise ValueError("Algorithms must be a dictionary")

        if not self.algorithms:
            raise ValueError("At least one algorithm must be provided")

        # Validate model naming convention
        if len(self.algorithms) == 1:
            if list(self.algorithms.keys()) != ["model"]:
                raise ValueError('Single model must use key "model"')
        else:
            expected_keys = [
                "model" if i == 0 else f"model{i+1}"
                for i in range(len(self.algorithms))
            ]
            if list(self.algorithms.keys()) != expected_keys:
                raise ValueError(
                    f"Multiple models must use keys {expected_keys}"
                )
