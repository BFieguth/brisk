"""experiment.py

This module defines the Experiment class, which represents a single experiment 
within the Brisk framework. The Experiment dataclass store the data path, group
name and algorithms to use.
"""

import dataclasses
import pathlib
from typing import Dict, Optional, List, Any

from brisk.utility import algorithm_wrapper

@dataclasses.dataclass
class Experiment:
    """Configuration for a single experiment run.
    
    Encapsulates all information needed for one experiment run, including 
    dataset path, model instances, and group organization. Provides validation 
    and consistent naming for experiment organization.
    
    Attributes:
        group_name: Name of the experiment group for organization
        dataset: Path to the dataset file
        algorithms: Dictionary of instantiated models with standardized keys:
                   - Single model: {"model": instance}
                   - Multiple models: {"model1": inst1, "model2": inst2, ...}
    
    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> experiment = Experiment(
        ...     group_name="baseline",
        ...     dataset=Path("data/example.csv"),
        ...     algorithms={"model": LinearRegression()}
        ... )
        >>> print(experiment.experiment_name)
        'baseline_a1b2c3d4'
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
        
        Returns:
            String combining group name and full model class names.
            Example: 'baseline_LinearRegression_RandomForestRegressor'
        """
        algo_names = "_".join(
            algo.name for algo in self.algorithms.values()
        )
        return f"{self.group_name}_{algo_names}"

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        dataset_name = (
            f"{self.dataset_path.stem}_{self.table_name}"
            if self.table_name else self.dataset_path.stem
        )
        return dataset_name

    @property
    def algorithm_kwargs(self) -> dict:
        """Dictionary with instantiated algorithms"""
        algorithm_kwargs = {
            key: algo.instantiate() for key, algo in self.algorithms.items()
        }
        return algorithm_kwargs

    @property
    def algorithm_names(self) -> list:
        """List of algorithm names"""
        algorithm_names = [algo.name for algo in self.algorithms.values()]
        return algorithm_names

    @property
    def workflow_attributes(self) -> Dict[str, Any]:
        """
        Combine the algorithm kwargs and the user defined workflow arguments
        into one dictionary that is passed to the Workflow instance.
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
        
        Raises:
            ValueError: If any validation fails:
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
