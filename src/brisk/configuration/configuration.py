"""User interface for defining experiment configurations.

This module defines the Configuration class, which serves as a user interface
for defining experiment configurations within the Brisk framework. It allows
users to create and manage experiment groups, specify the datasets to use,
algorithms, as well as modify starting values and hyperparameters.

Examples
--------
    >>> config = Configuration(default_algorithms=["linear", "ridge"])
    >>> config.add_experiment_group(
    ...     name="baseline",
    ...     datasets=["data.csv"]
    ... )
    >>> manager = config.build()
"""

from typing import List, Dict, Optional, Any, Tuple

from brisk.configuration.configuration_manager import ConfigurationManager
from brisk.configuration.experiment_group import ExperimentGroup

class Configuration:
    """User interface for defining experiment configurations.

    This class provides a simple interface for users to define experiment groups
    and their configurations. It handles default values and ensures unique
    group names.

    Parameters
    ----------
    default_algorithms : list of str
        List of algorithm names to use as defaults
    categorical_features : dict, optional
        Dict mapping categorical feature names to datasets
    default_workflow_args : dict, optional
        Values to assign as attributes of the Workflow

    Attributes
    ----------
    experiment_groups : list
        List of ExperimentGroup instances
    default_algorithms : list
        List of algorithm names to use when none specified
    categorical_features : dict
        Dict mapping categorical feature names to datasets
    default_workflow_args : dict
        Values to assign as attributes of the Workflow
    """
    def __init__(
        self,
        default_algorithms: List[str],
        categorical_features: Optional[Dict[str, List[str]]] = None,
        default_workflow_args: Optional[Dict[str, Any]] = None
    ):
        self.experiment_groups: List[ExperimentGroup] = []
        self.default_algorithms = default_algorithms
        self.categorical_features = categorical_features or {}
        self.default_workflow_args = default_workflow_args or {}

    def add_experiment_group(
        self,
        *,
        name: str,
        datasets: List[str | Tuple[str, str]],
        data_config: Optional[Dict[str, Any]] = None,
        algorithms: Optional[List[str]] = None,
        algorithm_config: Optional[Dict[str, Dict[str, Any]]] = None,
        description: Optional[str] = "",
        workflow_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new ExperimentGroup.

        Parameters
        ----------
        name : str
            Unique identifier for the group
        datasets : list
            List of dataset paths relative to datasets directory
        data_config : dict, optional
            Arguments for DataManager used by this ExperimentGroup
        algorithms : list of str, optional
            List of algorithms (uses defaults if None)
        algorithm_config : dict, optional
            Algorithm-specific configurations, overides values set in
            algorithms.py
        description : str, optional
            Description for the experiment group
        workflow_args : dict, optional
            Values to assign as attributes in the Workflow

        Raises
        ------
        ValueError
            If group name already exists or workflow_args keys don't match
            default_workflow_args
        """
        if algorithms is None:
            algorithms = self.default_algorithms

        if workflow_args is None:
            workflow_args = self.default_workflow_args
        else:
            if self.default_workflow_args.keys() != workflow_args.keys():
                raise ValueError(
                    "workflow_args must have the same keys as defined in"
                    " default_workflow_args"
                )

        self._check_name_exists(name)
        self._check_datasets_type(datasets)
        self.experiment_groups.append(
            ExperimentGroup(
                name,
                datasets,
                data_config,
                algorithms,
                algorithm_config,
                description,
                workflow_args
            )
        )

    def build(self) -> ConfigurationManager:
        """Build and return a ConfigurationManager instance.

        Returns
        -------
        ConfigurationManager
            Processes ExperimentGroups and creates data splits.
        """
        return ConfigurationManager(
            self.experiment_groups, self.categorical_features
        )

    def _check_name_exists(self, name: str) -> None:
        """Check if an experiment group name is already in use.

        Parameters
        ----------
        name : str
            Group name to check

        Raises
        ------
        ValueError
            If name has already been used
        """
        if any(group.name == name for group in self.experiment_groups):
            raise ValueError(
                f"Experiment group with name '{name}' already exists"
            )

    def _check_datasets_type(self, datasets) -> None:
        """Validate the type of datasets parameter.

        Parameters
        ----------
        datasets : list
            List of dataset specifications

        Raises
        ------
        TypeError
            If datasets contains invalid types (must be strings or tuples of
            strings)
        """
        for dataset in datasets:
            if isinstance(dataset, str):
                continue
            if isinstance(dataset, tuple):
                for val in dataset:
                    if isinstance(val, str):
                        continue
            else:
                raise TypeError(
                    "datasets must be a list containing strings and/or tuples "
                    f"of strings. Got {type(datasets)}."
                    )
