"""configuration.py

This module defines the Configuration class, which serves as a user interface 
for defining experiment configurations within the Brisk framework. It allows 
users to create and manage experiment groups, specify the datasets to use, 
algorithms, and their configurations.

Usage Example:
    >>> config = Configuration(default_algorithms=["linear", "ridge"])
    >>> config.add_experiment_group(
    ...     name="baseline",
    ...     datasets=["data.csv"]
    ... )
    >>> manager = config.build()
"""

from typing import List, Dict, Optional, Any

from brisk.configuration.configuration_manager import ConfigurationManager
from brisk.configuration.ExperimentGroup import ExperimentGroup

class Configuration:
    """User interface for defining experiment configurations.
    
    This class provides a simple API for users to define experiment groups
    and their configurations. It handles default values and ensures unique
    group names.
    
    Attributes:
        experiment_groups: List of configured ExperimentGroup instances
        default_algorithms: List of algorithm names to use when none specified
    
    Example:
        >>> config = Configuration(default_algorithms=["linear", "ridge"])
        >>> config.add_experiment_group(
        ...     name="baseline",
        ...     datasets=["data.csv"]
        ... )
        >>> manager = config.build()
    """
    def __init__(self, default_algorithms: List[str]):
        """Initialize Configuration with default algorithms.
        
        Args:
            default_algorithms: List of algorithm names to use as defaults
        """
        self.experiment_groups: List[ExperimentGroup] = []
        self.default_algorithms = default_algorithms

    def add_experiment_group(
        self,
        *,
        name: str,
        datasets: List[str],
        data_config: Optional[Dict[str, Any]] = None,
        algorithms: Optional[List[str]] = None,
        algorithm_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Add a new experiment group configuration.
        
        Args:
            name: Unique identifier for the group
            datasets: List of dataset paths relative to project root
            data_config: Optional configuration for DataManager
            algorithms: Optional list of algorithms (uses defaults if None)
            algorithm_config: Optional algorithm-specific configurations
        
        Raises:
            ValueError: If group name already exists
        """
        if algorithms is None:
            algorithms = self.default_algorithms

        self._check_name_exists(name)
        self.experiment_groups.append(
            ExperimentGroup(
                name,
                datasets,
                data_config,
                algorithms,
                algorithm_config
            )
        )

    def build(self) -> ConfigurationManager:
        """Build and return a ConfigurationManager instance.
        
        Returns:
            ConfigurationManager containing processed experiment configurations
        """
        return ConfigurationManager(self.experiment_groups)

    def _check_name_exists(self, name: str) -> None:
        """Check if an experiment group name is already in use.
        
        Args:
            name: Group name to check
            
        Raises:
            ValueError: If name is already in use
        """
        if any(group.name == name for group in self.experiment_groups):
            raise ValueError(
                f"Experiment group with name '{name}' already exists"
            )
