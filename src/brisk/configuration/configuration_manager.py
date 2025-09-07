"""Manage experiment configurations and DataManager instances.

This module defines the ConfigurationManager class, which is responsible for
managing experiment configurations and creating DataManager instances. The
ConfigurationManager processes configurations for experiment groups, ensuring
that DataManager instances are created efficiently and reused when
configurations match.
"""

import collections
from typing import List, Dict, Tuple

from brisk.data import data_manager
from brisk.configuration import (
    experiment_group, experiment_factory, project, algorithm_collection
)
from brisk.reporting import formatting
from brisk.services import get_services
from brisk.theme import plot_settings as plot_settings_module

class ConfigurationManager:
    """Manage experiment configurations and DataManager instances.

    This class processes ExperimentGroup configurations and creates the minimum
    necessary DataManager instances, reusing them when configurations match.

    Parameters
    ----------
    experiment_groups : List[ExperimentGroup]
        List of experiment group configurations to process
    categorical_features : Dict[str, List[str]]
        Dictionary mapping dataset identifiers to lists of categorical
        feature names
    plot_settings : PlotSettings
        Plot configuration settings for the experiments

    Attributes
    ----------
    experiment_groups : List[ExperimentGroup]
        List of experiment group configurations
    data_managers : Dict[str, DataManager]
        Mapping of group names to DataManager instances
    categorical_features : Dict[str, List[str]]
        Mapping of dataset identifiers to categorical feature lists
    project_root : Path
        Root directory of the project
    algorithm_config : AlgorithmCollection
        Collection of algorithm configurations loaded from algorithms.py
    base_data_manager : DataManager
        Base configuration for data management loaded from data.py
    experiment_queue : collections.deque
        Queue of experiments ready to run
    output_structure : Dict[str, Dict[str, Tuple[str, str]]]
        Directory structure for experiment outputs
    description_map : Dict[str, str]
        Mapping of group names to descriptions
    workflow_map : Dict[str, Type]
        Mapping of workflow names to workflow classes
    logfile : str
        Markdown documentation of the configuration

    Notes
    -----
    The ConfigurationManager optimizes memory usage by reusing DataManager
    instances when experiment groups have identical data configurations.
    This is particularly important when running many experiments with
    similar data processing requirements.

    Examples
    --------
    Create a configuration manager:
        >>> from brisk.configuration import ConfigurationManager
        >>> manager = ConfigurationManager(
        ...     experiment_groups=groups,
        ...     categorical_features=categorical_features,
        ...     plot_settings=plot_settings
        ... )
    """
    def __init__(
        self,
        experiment_groups: List[experiment_group.ExperimentGroup],
        categorical_features: Dict[str, List[str]],
        plot_settings: plot_settings_module.PlotSettings
    ):
        """Initialize ConfigurationManager with experiment groups and settings.

        Sets up the complete configuration for experiment execution including
        loading algorithm configurations, creating data managers, building
        experiment queues, and preparing output structures.

        Parameters
        ----------
        experiment_groups : List[ExperimentGroup]
            List of experiment group configurations to process
        categorical_features : Dict[str, List[str]]
            Dictionary mapping dataset identifiers to lists of categorical
            feature names
        plot_settings : PlotSettings
            Plot configuration settings for the experiments

        Notes
        -----
        The initialization process:
        1. Loads algorithm configuration from algorithms.py
        2. Loads base data manager configuration from data.py
        3. Creates optimized data manager instances
        4. Builds experiment queue from all groups
        5. Creates data splits for all datasets
        6. Generates configuration documentation
        7. Sets up output directory structure
        """
        self.services = get_services()
        self.services.io.set_io_settings(plot_settings.get_io_settings())
        self.services.utility.set_plot_settings(plot_settings)
        self.experiment_groups = experiment_groups
        self.categorical_features = categorical_features
        self.workflow_map = {}
        self.project_root = project.find_project_root()
        self.algorithm_config = self._load_algorithm_config()
        self.base_data_manager = self._load_base_data_manager()
        self.data_managers = self._create_data_managers()
        self.experiment_queue = self._create_experiment_queue()
        self._create_data_splits()
        self._create_logfile()
        self.output_structure = self._get_output_structure()
        self.description_map = self._create_description_map()

    def _load_base_data_manager(self) -> data_manager.DataManager:
        """Load default DataManager configuration from project's data.py.

        Loads the base DataManager configuration that serves as the template
        for all experiment groups. This configuration defines default
        parameters for data processing, splitting, and preprocessing.

        Returns
        -------
        DataManager
            Configured DataManager instance loaded from data.py

        Raises
        ------
        FileNotFoundError
            If data.py is not found in project root
        ImportError
            If data.py cannot be loaded or BASE_DATA_MANAGER is not defined

        Notes
        -----
        The data.py file must define:
        BASE_DATA_MANAGER = DataManager(...)

        This base configuration is used as a template for creating
        group-specific data managers with custom parameters.
        """
        data_file = self.project_root / "data.py"
        base_data_manager = self.services.io.load_base_data_manager(data_file)
        return base_data_manager

    def _load_algorithm_config(
        self
    ) -> algorithm_collection.AlgorithmCollection:
        """Load algorithm configuration from project's algorithms.py.

        Loads the complete algorithm configuration that defines all
        available algorithms, their default parameters, and hyperparameter
        grids for the experiments.

        Returns
        -------
        AlgorithmCollection
            Collection of AlgorithmWrapper instances from algorithms.py

        Raises
        ------
        FileNotFoundError
            If algorithms.py is not found in project root
        ImportError
            If algorithms.py cannot be loaded or ALGORITHM_CONFIG is not defined

        Notes
        -----
        The algorithms.py file must define:
        ALGORITHM_CONFIG = AlgorithmCollection(...)

        This configuration is used by the ExperimentFactory to create
        experiment instances with the appropriate algorithms.
        """
        algo_file = self.project_root / "algorithms.py"
        algo_config = self.services.io.load_algorithms(algo_file)
        return algo_config

    def _get_base_params(self) -> Dict:
        """Get parameters from base DataManager instance.

        Extracts all initialization parameters from the base DataManager
        to use as a template for creating group-specific data managers.

        Returns
        -------
        Dict
            Dictionary of parameter names and values from the base DataManager

        Notes
        -----
        This method uses introspection to extract parameter names from
        the DataManager's __init__ method, excluding 'self'. The resulting
        dictionary can be used to create new DataManager instances with
        the same base configuration plus any group-specific overrides.
        """
        return {
            name: getattr(self.base_data_manager, name)
            for name in self.base_data_manager.__init__.__code__.co_varnames
            if name != "self"
        }

    def _create_data_managers(self) -> Dict[str, data_manager.DataManager]:
        """Create minimal set of DataManager instances.

        Groups ExperimentGroups by their data_config and creates one
        DataManager instance per unique configuration. This optimization
        reduces memory usage by reusing data managers when configurations
        are identical.

        Returns
        -------
        Dict[str, DataManager]
            Dictionary mapping group names to DataManager instances

        Notes
        -----
        The method groups experiment groups by their data configuration:
        - Groups with identical data_config share the same DataManager
        - Groups with no data_config use the base DataManager
        - Preprocessor configurations are handled specially to ensure
          proper grouping based on preprocessor types

        This optimization is particularly important when running many
        experiments with similar data processing requirements.
        """
        config_groups = collections.defaultdict(list)
        for group in self.experiment_groups:
            # Create a hashable key for grouping similar configurations
            data_config = group.data_config or {}
            if "preprocessors" in data_config:
                # Create a key based on preprocessor types and other config
                preprocessor_types = tuple(
                    type(p).__name__ for p in data_config["preprocessors"]
                )
                other_config = {
                    k: v for k, v in data_config.items() if k != "preprocessors"
                }
                config_key = (
                    preprocessor_types, frozenset(other_config.items())
                )
            else:
                config_key = frozenset(data_config.items())

            config_groups[config_key].append(group.name)

        managers = {}
        for config_key, group_names in config_groups.items():
            first_group = next(
                g for g in self.experiment_groups
                if g.name in group_names
            )

            if not first_group.data_config:
                manager = self.base_data_manager
            else:
                base_params = self._get_base_params()
                base_params.update(first_group.data_config)
                manager = data_manager.DataManager(**base_params)

            for name in group_names:
                self.services.reporting.add_data_manager(name, manager)
                managers[name] = manager

        return managers

    def _create_experiment_queue(self) -> collections.deque:
        """Create queue of experiments from all ExperimentGroups.

        Creates an ExperimentFactory with loaded algorithm configuration,
        then processes each ExperimentGroup to create Experiment instances.
        Loads workflow classes and creates the complete experiment queue.

        Returns
        -------
        collections.deque
            Queue of Experiment instances ready to run

        Notes
        -----
        The method:
        1. Creates an ExperimentFactory with algorithm configuration
        2. Loads workflow classes for each experiment group
        3. Determines the number of data splits for each group
        4. Creates experiment instances for all algorithm-dataset combinations
        5. Adds all experiments to the execution queue

        The experiment queue is processed during experiment execution,
        with each experiment running independently.
        """
        factory = experiment_factory.ExperimentFactory(
            self.algorithm_config, self.categorical_features
        )

        all_experiments = collections.deque()
        for group in self.experiment_groups:
            workflow_class = self.services.io.load_workflow(group.workflow)
            self.workflow_map[group.workflow] = workflow_class
            n_splits = group.data_config.get(
                "n_splits", self.base_data_manager.n_splits
            )
            experiments = factory.create_experiments(group, n_splits)
            all_experiments.extend(experiments)

        return all_experiments

    def _create_data_splits(self) -> None:
        """Create DataSplitInfo instances for all datasets.

        Creates data splits for each dataset in each experiment group using
        the appropriate DataManager instance. This prepares all datasets
        for cross-validation and train/test splitting.

        Notes
        -----
        The method processes each experiment group and:
        1. Gets the appropriate DataManager for the group
        2. For each dataset in the group:
           - Determines categorical features for the dataset
           - Creates data splits using the DataManager
           - Associates splits with the group and dataset

        This ensures that all datasets are properly prepared for
        experiment execution with the correct feature categorization.
        """
        for group in self.experiment_groups:
            group_data_manager = self.data_managers[group.name]
            for dataset_path, table_name in group.dataset_paths:
                lookup_key = (
                    (dataset_path.name, table_name)
                    if table_name
                    else dataset_path.name
                )
                categorical_features = self.categorical_features.get(
                    lookup_key, None
                )
                group_data_manager.split(
                    data_path=str(dataset_path),
                    categorical_features=categorical_features,
                    group_name=group.name,
                    table_name=table_name,
                    filename=dataset_path.stem
                )

    def _create_logfile(self) -> None:
        """Create a markdown string describing the configuration.

        Generates comprehensive documentation of the experiment configuration
        including algorithm settings, experiment group details, data manager
        configurations, and dataset information.

        Notes
        -----
        The generated markdown includes:
        - Default algorithm configurations with parameters and grids
        - Experiment group descriptions and settings
        - DataManager configurations for each group
        - Dataset information including feature categorization
        - Algorithm-specific configurations for each group

        This documentation is saved as part of the experiment results
        and provides a complete record of the configuration used.
        """
        md_content = [
            "## Default Algorithm Configuration"
        ]

        # Add default algorithm configurations
        for algo in self.algorithm_config:
            md_content.append(algo.to_markdown())
            md_content.append("")

        # Add experiment group configurations
        for group in self.experiment_groups:
            md_content.extend([
                f"## Experiment Group: {group.name}",
                f"#### Description: {group.description}",
                ""
            ])

            # Add group-specific algorithm configurations
            if group.algorithm_config:
                md_content.extend([
                    "### Algorithm Configurations",
                    "```python",
                    formatting.format_dict(group.algorithm_config),
                    "```",
                    ""
                ])

            # Add DataManager configuration
            group_data_manager = self.data_managers[group.name]
            md_content.extend([
                "### DataManager Configuration",
                group_data_manager.to_markdown(),
                ""
            ])

            # Add dataset information
            md_content.append("### Datasets")
            for dataset_path, table_name in group.dataset_paths:
                lookup_key = (
                    (dataset_path.name, table_name)
                    if table_name
                    else dataset_path.name
                )
                categorical_features = self.categorical_features.get(
                    lookup_key, None
                )
                split_info = group_data_manager.split(
                    data_path=str(dataset_path),
                    categorical_features=categorical_features,
                    table_name=table_name,
                    group_name=group.name,
                    filename=dataset_path.stem
                ).get_split(0)

                md_content.extend([
                    f"#### {dataset_path.name}",
                    "Features:",
                    "```python",
                    f"Categorical: {split_info.categorical_features}",
                    f"Continuous: {split_info.continuous_features}",
                    "```",
                    ""
                ])

        self.logfile = "\n".join(md_content)

    def _get_output_structure(self) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """Get the directory structure for experiment outputs.

        Creates a nested dictionary structure that maps experiment groups
        to their datasets and provides the necessary path information
        for organizing experiment results.

        Returns
        -------
        Dict[str, Dict[str, Tuple[str, str]]]
            Nested dictionary structure where:
            - Top level keys are experiment group names
            - Second level maps dataset names to (path, table_name) tuples

        Notes
        -----
        The output structure is used to organize experiment results
        in a hierarchical manner:
        - Group level: Each experiment group gets its own directory
        - Dataset level: Each dataset within a group gets its own subdirectory
        - File level: Results are organized by dataset and table name

        This structure ensures that results from different experiments
        are properly separated and organized.
        """
        output_structure = {}

        for group in self.experiment_groups:
            dataset_info = {}

            for dataset_path, table_name in group.dataset_paths:
                dataset_name = (
                    f"{dataset_path.stem}_{table_name}"
                    if table_name else dataset_path.stem
                )
                dataset_info[dataset_name] = (
                    str(dataset_path), table_name
                    )

            output_structure[group.name] = dataset_info

        return output_structure

    def _create_description_map(self) -> Dict[str, str]:
        """Create a mapping of group names to descriptions.

        Creates a simple mapping of experiment group names to their
        descriptions, filtering out empty descriptions.

        Returns
        -------
        Dict[str, str]
            Mapping of group names to their descriptions, excluding empty
            descriptions

        Notes
        -----
        This mapping is used for generating reports and documentation
        where group descriptions are needed. Empty descriptions are
        filtered out to avoid cluttering the output with meaningless
        entries.
        """
        return {
            group.name: group.description
            for group in self.experiment_groups
            if group.description != ""
        }
