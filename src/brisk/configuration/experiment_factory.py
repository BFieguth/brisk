"""Create Experiment instances from ExperimentGroup instances.

This module defines the ExperimentFactory class, which is responsible for 
creating Experiment instances from ExperimentGroup configurations within the 
Brisk framework. The ExperimentFactory applies experiment-specific settings to
algorithms, and resolves dataset paths for experiments.

Examples
--------
>>> from brisk.utility.algorithm_wrapper import AlgorithmCollection
>>> from brisk.configuration.experiment_group import ExperimentGroup
>>> from brisk.configuration.experiment_factory import ExperimentFactory
>>> 
>>> algorithms = AlgorithmCollection([
...     AlgorithmWrapper(
...         name="linear",
...         display_name="Linear Regression",
...         algorithm_class=LinearRegression
...     )
... ])
>>> 
>>> categorical_features = {
...     "data.csv": ["category1", "category2"]
... }
>>> 
>>> factory = ExperimentFactory(
...     algorithm_config=algorithms,
...     categorical_features=categorical_features
... )
>>> 
"""
import collections
from typing import List, Dict, Any, Deque, Union

from brisk.configuration import experiment
from brisk.configuration import experiment_group
from brisk.configuration import algorithm_wrapper

class ExperimentFactory:
    """Factory for creating Experiment instances from ExperimentGroups.
    
    Takes a list of ExperimentGroup and creates a queue of Experiment instances.
    Applies specific configuration for each ExperimentGroup when creating the
    Experiment instances.

    Parameters
    ----------
    algorithm_config : AlgorithmCollection
        List of AlgorithmWrapper instances defining available algorithms
    categorical_features : dict
        Dict mapping categorical features to dataset names

    Attributes
    ----------
    algorithm_config : AlgorithmCollection
        Available algorithms
    categorical_features : dict
        Mapping of categorical features to datasets
    """

    def __init__(
        self,
        algorithm_config: algorithm_wrapper.AlgorithmCollection,
        categorical_features: Dict[str, List[str]]
    ):
        if not isinstance(
            algorithm_config, algorithm_wrapper.AlgorithmCollection
        ):
            raise TypeError(
                "algorithm_config must be an AlgorithmCollection, "
                f"got {type(algorithm_config)}"
            )
        self.algorithm_config = algorithm_config
        self.categorical_features = categorical_features

    def create_experiments(
        self,
        group: experiment_group.ExperimentGroup,
        n_splits: int
    ) -> Deque[experiment.Experiment]:
        """Create queue of experiments from an experiment group.
        
        Parameters
        ----------
        group : ExperimentGroup
            Configuration for the experiment group
        
        n_splits: int
            The number of data splits to create for this ExperimentGroup
            
        Returns
        -------
        collections.deque
            Queue of Experiment instances ready to run
            
        Examples
        --------
        >>> from brisk.utility.algorithm_wrapper import AlgorithmCollection
        >>> from brisk.configuration.experiment_group import ExperimentGroup
        >>> from brisk.configuration.experiment_factory import ExperimentFactory
        >>> 
        >>> algorithms = AlgorithmCollection([
        ...     AlgorithmWrapper(
        ...         name="linear",
        ...         display_name="Linear Regression",
        ...         algorithm_class=LinearRegression
        ...     )
        ... ])
        >>> 
        >>> categorical_features = {
        ...     "data.csv": ["category1", "category2"]
        ... }
        >>> 
        >>> factory = ExperimentFactory(
        ...     algorithm_config=algorithms,
        ...     categorical_features=categorical_features
        ... )
        >>> 
        >>> group = ExperimentGroup(
        ...     name="baseline", 
        ...     datasets=["data.csv"], 
        ...     algorithms=["linear"]
        ... )
        >>> 
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
                for index in range(0, n_splits):
                    experiments.append(experiment.Experiment(
                        group_name=group.name,
                        algorithms=models,
                        dataset_path=dataset_path,
                        workflow_args=group.workflow_args,
                        split_index=index,
                        table_name=table_name,
                        categorical_features=categorical_feature_names
                    ))

        return experiments

    def _get_algorithm_wrapper(
        self,
        algo_name: str,
        config: Dict[str, Any] | None = None
    ) -> algorithm_wrapper.AlgorithmWrapper:
        """Get algorithm wrapper with updated configuration.
        
        Parameters
        ----------
        algo_name : str
            Name of the algorithm to retrieve
        config : dict, optional
            Configuration updates to apply to the algorithm

        Returns
        -------
        AlgorithmWrapper
            New wrapper instance with updated configuration
        """
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
        
        Parameters
        ----------
        algorithms : list
            List of algorithm names or nested lists of names

        Returns
        -------
        list of list
            Normalized list where each inner list represents one experiment

        Examples
        --------
        >>> factory._normalize_algorithms(["algo1", "algo2"])
        [["algo1"], ["algo2"]]
        >>> factory._normalize_algorithms([["algo1", "algo2"]])
        [["algo1", "algo2"]]
        >>> factory._normalize_algorithms(["algo1", ["algo2", "algo3"]])
        [["algo1"], ["algo2", "algo3"]]
        """
        normalized = []
        if not isinstance(algorithms, list):
            raise TypeError(
                f"algorithms must be a list, got {type(algorithms)}"
            )
        for item in algorithms:
            if isinstance(item, str):
                normalized.append([item])
            elif isinstance(item, list):
                if not all(isinstance(i, str) for i in item):
                    raise TypeError(
                        "nested algorithm lists must contain strings, "
                        f"got {item}"
                    )
                normalized.append(item)
            else:
                raise TypeError(
                    "algorithms must contain strings or lists of strings, "
                    f"got {type(item)}"
                )
        return normalized
