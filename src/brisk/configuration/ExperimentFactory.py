from collections import deque
from typing import List, Dict, Any, Deque, Union

from brisk.configuration.Experiment import Experiment
from brisk.configuration.ExperimentGroup import ExperimentGroup
from brisk.utility.AlgorithmWrapper import AlgorithmWrapper

class ExperimentFactory:
    """Factory for creating Experiment instances from ExperimentGroup configurations.
    
    Handles:
    - Model instantiation from AlgorithmWrapper configs
    - Application of algorithm-specific configurations
    - Dataset path resolution
    """
    
    def __init__(self, algorithm_config: List[AlgorithmWrapper]):
        """Initialize factory with algorithm configuration.
        
        Args:
            algorithm_config: List of AlgorithmWrapper instances defining available algorithms
        """
        self.algorithm_config = {
            wrapper.name: wrapper for wrapper in algorithm_config
        }
    
    def create_experiments(self, group: ExperimentGroup) -> Deque[Experiment]:
        """Create queue of experiments from an experiment group.
        
        Args:
            group: ExperimentGroup configuration
            
        Returns:
            Deque of Experiment instances ready to run
            
        Example:
            >>> factory = ExperimentFactory(ALGORITHM_CONFIG)
            >>> group = ExperimentGroup(name="baseline", datasets=["data.csv"], algorithms=["linear"])
            >>> experiments = factory.create_experiments(group)
        """
        experiments = deque()
        group_algo_config = group.algorithm_config or {}
        
        algorithm_groups = self._normalize_algorithms(group.algorithms)

        for dataset_path in group.dataset_paths:
            dataset_name = dataset_path.stem
            experiment_group = f"{group.name}_{dataset_name}"

            for algo_group in algorithm_groups:
                models = {}
                hyperparam_grids = {}
                
                if len(algo_group) == 1:
                    algo_name = algo_group[0]
                    wrapper = self._get_algorithm_wrapper(
                        algo_name, 
                        group_algo_config.get(algo_name)
                    )
                    models["model"] = wrapper.instantiate()
                    hyperparam_grids["model"] = wrapper.hyperparam_grid
                else:
                    for i, algo_name in enumerate(algo_group):
                        model_key = f"model{i+1}"
                        wrapper = self._get_algorithm_wrapper(
                            algo_name, 
                            group_algo_config.get(algo_name)
                        )
                        models[model_key] = wrapper.instantiate()
                        hyperparam_grids[model_key] = wrapper.hyperparam_grid
                
                experiment = Experiment(
                    group_name=experiment_group,
                    dataset=dataset_path,
                    algorithms=models,
                    hyperparameters=hyperparam_grids
                )
                experiments.append(experiment)
        
        return experiments
    
    def _get_algorithm_wrapper(
        self, 
        algo_name: str, 
        config: Dict[str, Any] | None = None
    ) -> AlgorithmWrapper:
        """Get algorithm wrapper with updated configuration."""
        if algo_name not in self.algorithm_config:
            raise KeyError(f"Algorithm '{algo_name}' not found in configuration")
            
        wrapper = self.algorithm_config[algo_name]
        
        if config:
            wrapper = AlgorithmWrapper(
                name=wrapper.name,
                display_name=wrapper.display_name,
                algorithm_class=wrapper.algorithm_class,
                default_params=wrapper.default_params.copy(),
                hyperparam_grid=wrapper.hyperparam_grid.copy()
            )
            wrapper.hyperparam_grid.update(config)
            
        return wrapper

    def _normalize_algorithms(self, algorithms: List[Union[str, List[str]]]) -> List[List[str]]:
        """Normalize algorithm specification to list of lists.
        
        Examples:
            ["algo1", "algo2"] -> [["algo1"], ["algo2"]]
            [["algo1", "algo2"]] -> [["algo1", "algo2"]]
            ["algo1", ["algo2", "algo3"]] -> [["algo1"], ["algo2", "algo3"]]
        """
        normalized = []
        for item in algorithms:
            if isinstance(item, str):
                normalized.append([item])
            elif isinstance(item, list):
                normalized.append(item)
        return normalized
    