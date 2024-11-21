from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Dict, Any, List, Optional

from brisk.data.DataManager import DataManager
from brisk.utility.utility import find_project_root

@dataclass
class ExperimentGroup:
    """Container for experiment group configuration

    Stores and validates configuration for a group of related experiments,
    including datasets, algorithms, and their respective configurations.
    
    Attributes:
        name: Unique identifier for the experiment group
        datasets: List of dataset filenames relative to project's datasets directory
        data_config: Optional configuration for DataManager
        algorithms: Optional list of algorithms to use
        algorithm_config: Optional configuration for algorithms
    """
    name: str
    datasets: List[str]
    data_config: Optional[Dict[str, Any]] = None
    algorithms: Optional[List[str]] = None
    algorithm_config: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def dataset_paths(self) -> List[Path]:
        """Get full paths to datasets relative to project root.
        
        Returns:
            List of Path objects pointing to dataset files
        
        Raises:
            FileNotFoundError: If project root (.briskconfig) cannot be found
        """
        project_root = find_project_root()
        datasets_dir = project_root / 'datasets'
        return [datasets_dir / dataset for dataset in self.datasets]

    def __post_init__(self):
        """Validate experiment group configuration after initialization.
        
        Performs validation checks on:
        - Name format
        - Dataset existence
        - Algorithm configuration consistency
        - DataManager configuration parameters
        
        Raises:
            ValueError: If any validation check fails
            FileNotFoundError: If datasets cannot be found
        """
        self._validate_name()
        self._validate_datasets()
        self._validate_algorithm_config()
        self._validate_data_config()

    def _validate_name(self):
        """Validate experiment group name.
        
        Ensures name is a non-empty string.
        
        Raises:
            ValueError: If name is empty or not a string
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Experiment group name must be a non-empty string")
    
    def _validate_datasets(self):
        """Validate dataset specifications.
        
        Checks:
        - At least one dataset is specified
        - All specified datasets exist in project's datasets directory
        
        Raises:
            ValueError: If no datasets are specified
            FileNotFoundError: If any dataset file cannot be found
        """
        if not self.datasets:
            raise ValueError("At least one dataset must be specified")
        
        for dataset, path in zip(self.datasets, self.dataset_paths):
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {dataset}\n"
                    f"Expected location: {path}"
                )
            
    def _validate_algorithm_config(self):
        """Validate algorithm configuration.
        
        Ensures all algorithms in algorithm_config are present in the algorithms list.
        
        Raises:
            ValueError: If algorithm_config contains undefined algorithms
        """
        if self.algorithm_config:
            invalid_algorithms = (
                set(self.algorithm_config.keys()) - set(self.algorithms)
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
        
        Raises:
            ValueError: If data_config contains invalid parameters
        """
        if self.data_config:
            valid_data_params = set(
                signature(DataManager.__init__).parameters.keys()
            )
            valid_data_params.remove("self")

            invalid_params = set(self.data_config.keys()) - valid_data_params
            if invalid_params:
                raise ValueError(
                    f"Invalid DataManager parameters: {invalid_params}\n"
                    f"Valid parameters are: {valid_data_params}"
                )
            