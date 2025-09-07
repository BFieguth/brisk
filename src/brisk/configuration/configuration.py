"""Interface for defining experiment configurations.

This module defines the Configuration class, which serves as a user interface
for defining experiment configurations within the Brisk framework. It allows
users to create and manage experiment groups, specify the datasets to use,
algorithms, as well as modify starting values and hyperparameters.

Examples
--------
Create a basic configuration:
    >>> config = Configuration(
    ...     default_workflow="workflow",
    ...     default_algorithms=["linear", "ridge"]
    ... )
    >>> config.add_experiment_group(
    ...     name="baseline",
    ...     datasets=["data.csv"]
    ... )
    >>> manager = config.build()

Create configuration with custom settings:
    >>> config = Configuration(
    ...     default_workflow="custom_workflow",
    ...     default_algorithms=["svm", "rf"],
    ...     categorical_features={"data.csv": ["category1", "category2"]},
    ...     default_workflow_args={"param1": "value1"}
    ... )
"""
from typing import List, Dict, Optional, Any, Tuple

from brisk.configuration import configuration_manager
from brisk.configuration import experiment_group
from brisk.services import get_services
from brisk.theme import plot_settings as plot_settings_module

class Configuration:
    """User interface for defining experiment configurations.

    This class provides a simple interface for users to define experiment groups
    and their configurations. It handles default values, ensures unique group
    names, and provides validation for configuration parameters.

    Parameters
    ----------
    default_workflow : str
        Default workflow name to use for experiment groups
    default_algorithms : List[str]
        List of algorithm names to use as defaults when none specified
    categorical_features : Dict[str, List[str]], optional
        Dictionary mapping dataset identifiers to lists of categorical
        feature names, by default None
    default_workflow_args : Dict[str, Any], optional
        Default values to assign as attributes of the Workflow class,
        by default None
    plot_settings : PlotSettings, optional
        Plot configuration settings, by default None

    Attributes
    ----------
    default_workflow : str
        Default workflow name for experiment groups
    experiment_groups : List[ExperimentGroup]
        List of configured experiment groups
    default_algorithms : List[str]
        List of default algorithm names
    categorical_features : Dict[str, List[str]]
        Mapping of dataset identifiers to categorical feature lists
    default_workflow_args : Dict[str, Any]
        Default workflow arguments
    plot_settings : PlotSettings
        Plot configuration settings

    Notes
    -----
    The Configuration class serves as the main user interface for setting up
    experiments. It provides a fluent API for adding experiment groups and
    automatically handles validation and default value assignment.

    Examples
    --------
    Create a simple configuration:
        >>> config = Configuration(
        ...     default_workflow="workflow",
        ...     default_algorithms=["linear", "ridge"]
        ... )

    Add experiment groups:
        >>> config.add_experiment_group(
        ...     name="baseline",
        ...     datasets=["data1.csv", "data2.csv"],
        ...     algorithms=["linear", "svm"]
        ... )

    Build the configuration manager:
        >>> manager = config.build()
    """
    def __init__(
        self,
        default_workflow: str,
        default_algorithms: List[str],
        categorical_features: Optional[Dict[str, List[str]]] = None,
        default_workflow_args: Optional[Dict[str, Any]] = None,
        plot_settings: Optional[plot_settings_module.PlotSettings] = None
    ):
        """Initialize Configuration with default settings.

        Creates a new Configuration instance with the specified default
        workflow, algorithms, and optional configuration parameters.

        Parameters
        ----------
        default_workflow : str
            Default workflow name to use for experiment groups
        default_algorithms : List[str]
            List of algorithm names to use as defaults
        categorical_features : Dict[str, List[str]], optional
            Dictionary mapping dataset identifiers to categorical feature lists,
            by default None
        default_workflow_args : Dict[str, Any], optional
            Default values to assign as workflow attributes, by default None
        plot_settings : PlotSettings, optional
            Plot configuration settings, by default None

        Notes
        -----
        If plot_settings is not provided, a default PlotSettings instance
        will be created automatically.
        """
        self.default_workflow = default_workflow
        self.experiment_groups: List[experiment_group.ExperimentGroup] = []
        self.default_algorithms = default_algorithms
        self.categorical_features = categorical_features or {}
        self.default_workflow_args = default_workflow_args or {}
        self.plot_settings = plot_settings
        if self.plot_settings is None:
            self.plot_settings = plot_settings_module.PlotSettings()

    def add_experiment_group(
        self,
        *,
        name: str,
        datasets: List[str | Tuple[str, str]],
        data_config: Optional[Dict[str, Any]] = None,
        algorithms: Optional[List[str]] = None,
        algorithm_config: Optional[Dict[str, Dict[str, Any]]] = None,
        description: Optional[str] = "",
        workflow: Optional[str] = None,
        workflow_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new ExperimentGroup to the configuration.

        Adds a new experiment group with the specified parameters.
        Validates the group name uniqueness and dataset format before adding.

        Parameters
        ----------
        name : str
            Unique identifier for the experiment group
        datasets : List[str | Tuple[str, str]]
            List of dataset paths relative to datasets directory. Can be
            strings (dataset files) or tuples of (dataset_file, table_name)
            for multi-table databases
        data_config : Dict[str, Any], optional
            Arguments for DataManager used by this experiment group,
            by default None
        algorithms : List[str], optional
            List of algorithm names to use. If None, uses default_algorithms,
            by default None
        algorithm_config : Dict[str, Dict[str, Any]], optional
            Algorithm-specific configurations that override values set in
            algorithms.py, by default None
        description : str, optional
            Human-readable description for the experiment group,
            by default ""
        workflow : str, optional
            Name of the workflow file to use (without .py extension).
            If None, uses default_workflow, by default None
        workflow_args : Dict[str, Any], optional
            Values to assign as attributes in the Workflow class.
            Must have same keys as default_workflow_args, by default None

        Raises
        ------
        ValueError
            If group name already exists or workflow_args keys don't match
            default_workflow_args
        TypeError
            If datasets contains invalid types (must be strings or tuples)

        Notes
        -----
        The method performs several validation checks:
        1. Ensures group name is unique
        2. Validates dataset format (strings or tuples of strings)
        3. Validates workflow_args keys match default_workflow_args
        4. Converts string datasets to (dataset, None) tuples

        Examples
        --------
        Add a simple experiment group:
            >>> config.add_experiment_group(
            ...     name="baseline",
            ...     datasets=["data.csv"]
            ... )

        Add group with custom settings:
            >>> config.add_experiment_group(
            ...     name="advanced",
            ...     datasets=[("data.xlsx", "Sheet1"), "data2.csv"],
            ...     algorithms=["svm", "rf"],
            ...     data_config={"test_size": 0.3},
            ...     description="Advanced experiment with custom settings"
            ... )
        """
        if algorithms is None:
            algorithms = self.default_algorithms

        if workflow is None:
            workflow = self.default_workflow

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
        formated_datasets = self._convert_datasets_to_tuple(datasets)
        self.experiment_groups.append(
            experiment_group.ExperimentGroup(
                name,
                formated_datasets,
                workflow,
                data_config,
                algorithms,
                algorithm_config,
                description,
                workflow_args
            )
        )

    def build(self) -> configuration_manager.ConfigurationManager:
        """Build and return a ConfigurationManager instance.

        Processes all experiment groups and creates a ConfigurationManager
        that can execute the experiments. Exports configuration parameters
        for rerun functionality.

        Returns
        -------
        ConfigurationManager
            Fully configured manager ready to execute experiments

        Notes
        -----
        The build process:
        1. Exports configuration parameters for rerun functionality
        2. Creates a ConfigurationManager with all experiment groups
        3. Sets up data managers, algorithm configurations, and workflows
        4. Prepares the complete experiment execution environment

        Examples
        --------
        Build and use the configuration:
            >>> config = Configuration("workflow", ["linear", "ridge"])
            >>> config.add_experiment_group(name="test", datasets=["data.csv"])
            >>> manager = config.build()
            >>> # manager is ready to execute experiments
        """
        self.export_params()
        return configuration_manager.ConfigurationManager(
            self.experiment_groups, self.categorical_features,
            self.plot_settings
        )

    def _check_name_exists(self, name: str) -> None:
        """Check if an experiment group name is already in use.

        Validates that the provided group name is unique within the
        current configuration.

        Parameters
        ----------
        name : str
            Group name to check for uniqueness

        Raises
        ------
        ValueError
            If name has already been used in another experiment group

        Notes
        -----
        This method is called automatically when adding experiment groups
        to ensure all group names are unique within a configuration.
        """
        if any(group.name == name for group in self.experiment_groups):
            raise ValueError(
                f"Experiment group with name '{name}' already exists"
            )

    def _check_datasets_type(self, datasets) -> None:
        """Validate the type of datasets parameter.

        Ensures that all items in the datasets list are either strings
        or tuples of strings, which are the only valid dataset specifications.

        Parameters
        ----------
        datasets : list
            List of dataset specifications to validate

        Raises
        ------
        TypeError
            If datasets contains invalid types (must be strings or tuples
            of strings)

        Notes
        -----
        Valid dataset specifications:
        - String: "data.csv" (single dataset file)
        - Tuple: ("data.xlsx", "Sheet1") (dataset file with table name)

        Invalid specifications will raise TypeError with a descriptive message.
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

    def _convert_datasets_to_tuple(
        self,
        datasets: List[str | Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Convert datasets to tuples if they are strings.

        Normalizes dataset specifications to a consistent tuple format
        for internal processing.

        Parameters
        ----------
        datasets : List[str | Tuple[str, str]]
            List of dataset specifications (strings or tuples)

        Returns
        -------
        List[Tuple[str, str]]
            List of normalized dataset tuples where:
            - String datasets become (dataset, None)
            - Tuple datasets remain unchanged

        Notes
        -----
        This normalization ensures consistent handling of dataset specifications
        throughout the configuration system. All datasets are represented as
        (dataset_file, table_name) tuples internally.
        """
        formated_datasets = []
        for dataset in datasets:
            if isinstance(dataset, tuple):
                formated_datasets.append(dataset)
            else:
                formated_datasets.append((dataset, None))
        return formated_datasets

    def export_params(self) -> None:
        """Export configuration parameters for rerun functionality.

        Serializes the current configuration to a format that can be
        used to recreate the experiment setup during rerun operations.
        This includes all experiment groups, categorical features, and
        plot settings.

        Notes
        -----
        The exported parameters include:
        - Default workflow and algorithms
        - Categorical features mapping
        - Plot settings configuration
        - All experiment group configurations
        - Dataset metadata for validation

        This data is used by the rerun system to ensure experiments
        can be reproduced with identical configurations.

        Examples
        --------
        Export is called automatically during build():
            >>> config = Configuration("workflow", ["linear"])
            >>> config.add_experiment_group(name="test", datasets=["data.csv"])
            >>> manager = config.build()  # export_params() called automatically
        """
        services = get_services()

        # flatten categorical_features to a list of items
        categorical_items = []
        for key, features in (self.categorical_features or {}).items():
            if isinstance(key, tuple):
                dataset, table_name = key
            else:
                dataset, table_name = key, None
            categorical_items.append({
                "dataset": dataset,
                "table_name": table_name,
                "features": list(features or []),
            })

        configuration_json = {
            "default_workflow": self.default_workflow,
            "default_algorithms": list(self.default_algorithms),
            "default_workflow_args": dict(self.default_workflow_args or {}),
            "categorical_features": categorical_items,
            "plot_settings": self.plot_settings.export_params(),
        }

        groups_json = []
        for group in self.experiment_groups:
            # datasets converted to tuples; keep them as (path, table_name)
            datasets = []
            for dataset in group.datasets:
                if isinstance(dataset, tuple):
                    datasets.append(
                        {"dataset": dataset[0], "table_name": dataset[1]}
                    )
                else:
                    datasets.append({"dataset": dataset, "table_name": None})

            groups_json.append({
                "name": group.name,
                "datasets": datasets,
                "workflow": group.workflow,
                "data_config": dict(group.data_config or {}),
                "algorithms": list(group.algorithms or []),
                "algorithm_config": dict(group.algorithm_config or {}),
                "description": group.description,
                "workflow_args": dict(group.workflow_args or {}),
            })

        services.rerun.add_configuration(configuration_json)
        services.rerun.add_experiment_groups(groups_json)
        services.rerun.collect_dataset_metadata(groups_json)
