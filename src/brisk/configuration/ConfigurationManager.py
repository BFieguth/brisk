import dataclasses
import collections
from typing import List, Dict, Optional
from importlib import util

from brisk.data.DataManager import DataManager
from brisk.configuration.ExperimentGroup import ExperimentGroup
from brisk.configuration.ExperimentFactory import ExperimentFactory
from brisk.utility.utility import find_project_root
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

class ConfigurationManager:
    """Manages experiment configurations and DataManager instances.
    
    This class processes ExperimentGroup configurations and creates the minimum
    necessary DataManager instances, reusing them when configurations match.
    
    Attributes:
        experiment_groups: List of experiment group configurations
        data_managers: Mapping of unique configurations to DataManager instances
    """
    def __init__(self, experiment_groups: List[ExperimentGroup]):
        """Initialize ConfigurationManager.
        
        Args:
            experiment_groups: List of experiment group configurations
        """
        self.experiment_groups = experiment_groups
        self.project_root = find_project_root()
        self.base_data_manager = self._load_base_data_manager()
        self.data_managers = self._create_data_managers()
        self.experiment_queue = self._create_experiment_queue()
        self._create_data_splits()

    def _load_base_data_manager(self) -> DataManager:
        """Load default DataManager configuration from project's data.py.
        
        Looks for data.py in project root and loads BASE_DATA_MANAGER.
        
        Returns:
            DataManager: Configured instance from data.py
            
        Raises:
            FileNotFoundError: If data.py is not found in project root
            ImportError: If data.py cannot be loaded or BASE_DATA_MANAGER is not defined
            
        Note:
            data.py must define BASE_DATA_MANAGER = DataManager(...)
        """
        data_file = self.project_root / 'data.py'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please create data.py with BASE_DATA_MANAGER configuration"
            )
        
        spec = util.spec_from_file_location('data', data_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load data module from {data_file}")
            
        data_module = util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
        
        if not hasattr(data_module, 'BASE_DATA_MANAGER'):
            raise ImportError(
                f"BASE_DATA_MANAGER not found in {data_file}\n"
                f"Please define BASE_DATA_MANAGER = DataManager(...)"
            )
        
        return data_module.BASE_DATA_MANAGER

    def _load_algorithm_config(self) -> List[AlgorithmWrapper]:
        """Load algorithm configuration from project's algorithms.py.
        
        Looks for algorithms.py in project root and loads ALGORITHM_CONFIG.
        
        Returns:
            List of AlgorithmWrapper instances from algorithms.py
            
        Raises:
            FileNotFoundError: If algorithms.py is not found in project root
            ImportError: If algorithms.py cannot be loaded or ALGORITHM_CONFIG is not defined
            
        Note:
            algorithms.py must define ALGORITHM_CONFIG = [...]
        """
        algo_file = self.project_root / 'algorithms.py'
        
        if not algo_file.exists():
            raise FileNotFoundError(
                f"Algorithm config file not found: {algo_file}\n"
                f"Please create algorithms.py with ALGORITHM_CONFIG list"
            )
        
        spec = util.spec_from_file_location('algorithms', algo_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load algorithms module from {algo_file}")
            
        algo_module = util.module_from_spec(spec)
        spec.loader.exec_module(algo_module)
        
        if not hasattr(algo_module, 'ALGORITHM_CONFIG'):
            raise ImportError(
                f"ALGORITHM_CONFIG not found in {algo_file}\n"
                f"Please define ALGORITHM_CONFIG = [...]"
            )
        
        return algo_module.ALGORITHM_CONFIG

    def _get_base_params(self) -> Dict:
        """Get parameters from base DataManager instance.
        
        Returns:
            Dictionary of current parameter values
        """
        return {
            name: getattr(self.base_data_manager, name)
            for name in self.base_data_manager.__init__.__code__.co_varnames
            if name != 'self'
        }
    
    def _create_data_managers(self) -> Dict[str, DataManager]:
        """Create minimal set of DataManager instances.
        
        Groups ExperimentGroups by their data_config and creates one
        DataManager instance per unique configuration.
        
        Returns:
            Dictionary mapping group names to DataManager instances
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
                manager = DataManager(**base_params)

            for name in group_names:
                managers[name] = manager

        return managers
    
    def _create_experiment_queue(self) -> collections.deque:
        """Create queue of experiments from all ExperimentGroups.
        
        Creates an ExperimentFactory with loaded algorithm configuration,
        then processes each ExperimentGroup to create Experiment instances.
        All experiments are combined into a single queue.
        
        Returns:
            Deque of Experiment instances ready to run
        """
        algorithm_config = self._load_algorithm_config()
        factory = ExperimentFactory(algorithm_config)
        
        all_experiments = collections.deque()
        for group in self.experiment_groups:
            experiments = factory.create_experiments(group)
            all_experiments.extend(experiments)
            
        return all_experiments

    def _create_data_splits(self):
        """Create DataSplitInfo instances for all datasets in experiment groups."""
        for group in self.experiment_groups:
            data_manager = self.data_managers[group.name]
            for dataset_path in group.dataset_paths:
                # Create split with group name and filename for caching
                data_manager.split(
                    data_path=str(dataset_path),
                    group_name=group.name,
                    filename=dataset_path.stem
                )

