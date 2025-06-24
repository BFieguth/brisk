"""Manage experiment configurations and DataManager instances.

This module defines the ConfigurationManager class, which is responsible for 
managing experiment configurations and creating DataManager instances. The 
ConfigurationManager processes configurations for experiment groups, ensuring 
that DataManager instances are created efficiently and reused when 
configurations match.
"""

import ast
import collections
from importlib import util
from pathlib import Path
from typing import List, Dict, Tuple

from brisk.data import data_manager
from brisk.configuration import experiment_group, experiment_factory, project
from brisk.configuration import algorithm_wrapper
from brisk.reporting import formatting

class ConfigurationManager:
    """Manage experiment configurations and DataManager instances.
    
    This class processes ExperimentGroup configurations and creates the minimum
    necessary DataManager instances, reusing them when configurations match.
    
    Parameters
    ----------
    experiment_groups : list of ExperimentGroup
        List of experiment group configurations
    categorical_features : dict
        Dict mapping categorical features to dataset

    Attributes
    ----------
    experiment_groups : list
        List of experiment group configurations
    data_managers : dict
        Mapping of unique configurations to DataManager instances
    categorical_features : dict
        Mapping of features to their datasets
    project_root : Path
        Root directory of the project
    algorithm_config : AlgorithmCollection
        Collection of algorithm configurations
    base_data_manager : DataManager
        Base configuration for data management
    experiment_queue : collections.deque
        Queue of experiments to run
    output_structure : dict
        Directory structure for experiment outputs
    description_map : dict
        Mapping of group names to descriptions
    """
    def __init__(
        self,
        experiment_groups: List[experiment_group.ExperimentGroup],
        categorical_features: Dict[str, List[str]]
    ):
        """Initialize ConfigurationManager.
        
        Args:
            experiment_groups: List of experiment group configurations
            categorical_features: Dict mapping categorical features to dataset
        """
        self.experiment_groups = experiment_groups
        self.categorical_features = categorical_features
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
        
        Parameters
        ----------
        None

        Returns
        -------
        DataManager
            Configured instance from data.py
            
        Raises
        ------
        FileNotFoundError
            If data.py is not found in project root
        ImportError
            If data.py cannot be loaded or BASE_DATA_MANAGER is not defined
            
        Notes
        -----
        data.py must define BASE_DATA_MANAGER = DataManager(...)
        """
        data_file = self.project_root / "data.py"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please create data.py with BASE_DATA_MANAGER configuration"
            )

        spec = util.spec_from_file_location("data", data_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load data module from {data_file}")

        data_module = util.module_from_spec(spec)
        spec.loader.exec_module(data_module)

        if not hasattr(data_module, "BASE_DATA_MANAGER"):
            raise ImportError(
                f"BASE_DATA_MANAGER not found in {data_file}\n"
                f"Please define BASE_DATA_MANAGER = DataManager(...)"
            )
        self._validate_single_variable(data_file, "BASE_DATA_MANAGER")
        if not isinstance(
            data_module.BASE_DATA_MANAGER, data_manager.DataManager
        ):
            raise ValueError(
                f"BASE_DATA_MANAGER in {data_file} is not a valid DataManager instance"
            )
        return data_module.BASE_DATA_MANAGER

    def _validate_single_variable(
        self,
        file_path: Path,
        variable_name: str
    ) -> None:
        """Validate that only a variable name is defined only once in a file.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to check
        variable_name : str
            Name of the variable to check
        
        Raises
        ------
        ValueError
            If the variable is defined multiple times
        SyntaxError
            If the file contains invalid Python syntax
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code, filename=str(file_path))
            
            assignments = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == variable_name
                        ):
                            assignments.append(node.lineno)
            
            if len(assignments) > 1:
                lines_str = ", ".join(map(str, assignments))
                raise ValueError(
                    f"{variable_name} is defined multiple times in {file_path} "
                    f"on lines: {lines_str}. Please define it exactly once to avoid ambiguity."
                )
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax in {file_path}: {e}")

    def _load_algorithm_config(
        self
    ) -> algorithm_wrapper.AlgorithmCollection:
        """Load algorithm configuration from project's algorithms.py.
        
        Parameters
        ----------
        None

        Returns
        -------
        list
            List of AlgorithmWrapper instances from algorithms.py
            
        Raises
        ------
        FileNotFoundError
            If algorithms.py is not found in project root
        ImportError
            If algorithms.py cannot be loaded or ALGORITHM_CONFIG is not defined
            
        Notes
        -----
        algorithms.py must define ALGORITHM_CONFIG = AlgorithmCollection()
        """
        algo_file = self.project_root / "algorithms.py"

        if not algo_file.exists():
            raise FileNotFoundError(
                f"algorithms.py file not found: {algo_file}\n"
                f"Please create algorithms.py and define an AlgorithmCollection"
            )

        spec = util.spec_from_file_location("algorithms", algo_file)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Failed to load algorithms module from {algo_file}"
                )

        algo_module = util.module_from_spec(spec)
        spec.loader.exec_module(algo_module)

        if not hasattr(algo_module, "ALGORITHM_CONFIG"):
            raise ImportError(
                f"ALGORITHM_CONFIG not found in {algo_file}\n"
                f"Please define ALGORITHM_CONFIG = AlgorithmCollection()"
            )
        self._validate_single_variable(algo_file, "ALGORITHM_CONFIG")
        if not isinstance(
            algo_module.ALGORITHM_CONFIG, algorithm_wrapper.AlgorithmCollection
        ):
            raise ValueError(
                f"ALGORITHM_CONFIG in {algo_file} is not a valid AlgorithmCollection instance"
            )
        return algo_module.ALGORITHM_CONFIG

    def _get_base_params(self) -> Dict:
        """Get parameters from base DataManager instance.
        
        Returns
        -------
        dict
            Dictionary of current parameter values from base DataManager
        """
        return {
            name: getattr(self.base_data_manager, name)
            for name in self.base_data_manager.__init__.__code__.co_varnames
            if name != "self"
        }

    def _create_data_managers(self) -> Dict[str, data_manager.DataManager]:
        """Create minimal set of DataManager instances.
        
        Groups ExperimentGroups by their data_config and creates one
        DataManager instance per unique configuration.
        
        Returns
        -------
        dict
            Dictionary mapping group names to DataManager instances
        
        Notes
        -----
        Reuses DataManager instances when configurations match to minimize
        memory usage
        """
        config_groups = collections.defaultdict(list)
        for group in self.experiment_groups:
            # Convert data_config to frozendict for hashable key
            config_key = frozenset(
                (group.data_config or {}).items()
            )
            config_groups[config_key].append(group.name)

        managers = {}
        for config, group_names in config_groups.items():
            if not config:
                manager = self.base_data_manager
            else:
                base_params = self._get_base_params()
                new_params = dict(config)
                base_params.update(new_params)
                manager = data_manager.DataManager(**base_params)

            for name in group_names:
                managers[name] = manager

        return managers

    def _create_experiment_queue(self) -> collections.deque:
        """Create queue of experiments from all ExperimentGroups.
        
        Creates an ExperimentFactory with loaded algorithm configuration,
        then processes each ExperimentGroup to create Experiment instances.
        
        Returns
        -------
        collections.deque
            Queue of Experiment instances ready to run
        """
        factory = experiment_factory.ExperimentFactory(
            self.algorithm_config, self.categorical_features
        )

        all_experiments = collections.deque()
        for group in self.experiment_groups:
            experiments = factory.create_experiments(group)
            all_experiments.extend(experiments)

        return all_experiments

    def _create_data_splits(self) -> None:
        """Create DataSplitInfo instances for all datasets.
        
        Creates splits for each dataset in each experiment group using the
        appropriate DataManager instance.
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
                    table_name=table_name,
                    group_name=group.name,
                    filename=dataset_path.stem
                )

    def _create_logfile(self) -> None:
        """Create a markdown string describing the configuration.
        
        Creates a detailed markdown document containing:
        - Default algorithm configurations
        - Experiment group configurations
        - DataManager settings
        - Dataset information
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
                )

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
        
        Returns
        -------
        dict
            Nested dictionary structure where:
            - Top level keys are experiment group names
            - Second level maps dataset names to (path, table_name) tuples
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
        
        Returns
        -------
        dict
            Mapping of group names to their descriptions, excluding empty
            descriptions
        """
        return {
            group.name: group.description
            for group in self.experiment_groups
            if group.description != ""
        }
