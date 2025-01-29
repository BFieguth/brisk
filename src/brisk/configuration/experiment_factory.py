"""experiment_factory.py

This module defines the ExperimentFactory class, which is responsible for 
creating Experiment instances from ExperimentGroup configurations within the 
Brisk framework. The ExperimentFactory applies experiment-specific settings to
algorithms, and resolves dataset paths for experiments.

Usage Example:
    >>> from brisk.configuration.experiment_factory import ExperimentFactory
    >>> factory = ExperimentFactory(ALGORITHM_CONFIG)
    >>> group = ExperimentGroup(
    ...     name="baseline", 
    ...     datasets=["data.csv"], 
    ...     algorithms=["linear"]
    ... )
    >>> experiments = factory.create_experiments(group)
"""
import collections
from typing import List, Dict, Any, Deque, Union

from brisk.configuration import experiment
from brisk.configuration import experiment_group
from brisk.utility import algorithm_wrapper

class ExperimentFactory:
    """Factory for creating Experiment instances from ExperimentGroups.
    
    Handles:
    - Model instantiation from AlgorithmWrapper configs
    - Application of algorithm-specific configurations
    - Dataset path resolution
    """

    def __init__(
        self,
        algorithm_config: List[algorithm_wrapper.AlgorithmWrapper],
        categorical_features: Dict[str, List[str]]
    ):
        """Initialize factory with algorithm configuration.
        
        Args:
            algorithm_config: List of AlgorithmWrapper instances defining 
            available algorithms
            categorical_features: Dict mapping categorical features to dataset
        """
        self.algorithm_config = {
            wrapper.name: wrapper for wrapper in algorithm_config
        }
        self.categorical_features = categorical_features

    def create_experiments(
        self,
        group: experiment_group.ExperimentGroup
    ) -> Deque[experiment.Experiment]:
        """Create queue of experiments from an experiment group.
        
        Args:
            group: ExperimentGroup configuration
            
        Returns:
            Deque of Experiment instances ready to run
            
        Example:
            >>> factory = ExperimentFactory(ALGORITHM_CONFIG)
            >>> group = ExperimentGroup(
                            name="baseline", 
                            datasets=["data.csv"], algorithms=["linear"]
                        )
            >>> experiments = factory.create_experiments(group)
        """
        experiments = collections.deque()
        group_algo_config = group.algorithm_config or {}

        algorithm_groups = self._normalize_algorithms(group.algorithms)

        for dataset_path, table_name in group.dataset_paths:
            for algo_group in algorithm_groups:
                models = {}

                if len(algo_group) == 1:
                    algo_name = algo_group[0]
                    wrapper = self._get_algorithm_wrapper(
                        algo_name,
                        group_algo_config.get(algo_name)
                    )
                    models["model"] = wrapper
                else:
                    for i, algo_name in enumerate(algo_group):
                        if i == 0:
                            model_key = "model"
                        else:
                            model_key = f"model{i+1}"
                        wrapper = self._get_algorithm_wrapper(
                            algo_name,
                            group_algo_config.get(algo_name)
                        )
                        models[model_key] = wrapper

                lookup_key = (
                    (dataset_path.name, table_name)
                    if table_name
                    else dataset_path.name
                )
                categorical_feature_names = self.categorical_features.get(
                    lookup_key, None
                )
                exp = experiment.Experiment(
                    group_name=group.name,
                    algorithms=models,
                    dataset_path=dataset_path,
                    table_name=table_name,
                    categorical_features=categorical_feature_names
                )
                experiments.append(exp)

        return experiments

    def _get_algorithm_wrapper(
        self,
        algo_name: str,
        config: Dict[str, Any] | None = None
    ) -> algorithm_wrapper.AlgorithmWrapper:
        """Get algorithm wrapper with updated configuration."""
        if algo_name not in self.algorithm_config:
            raise KeyError(
                f"Algorithm '{algo_name}' not found in configuration"
            )

        original_wrapper = self.algorithm_config[algo_name]

        wrapper = algorithm_wrapper.AlgorithmWrapper(
            name=original_wrapper.name,
            display_name=original_wrapper.display_name,
            algorithm_class=original_wrapper.algorithm_class,
            default_params=original_wrapper.default_params.copy(),
            hyperparam_grid=original_wrapper.hyperparam_grid.copy()
        )
        if config:
            wrapper.hyperparam_grid.update(config)

        return wrapper

    def _normalize_algorithms(
        self,
        algorithms: List[Union[str, List[str]]]
    ) -> List[List[str]]:
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
