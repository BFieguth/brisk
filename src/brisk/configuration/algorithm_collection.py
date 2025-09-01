from typing import Any, Dict, Union

from brisk.configuration import algorithm_wrapper
from brisk.defaults import regression_algorithms, classification_algorithms

class AlgorithmCollection(list):
    """A collection for managing AlgorithmWrapper instances.

    Provides both list-like and dict-like access to AlgorithmWrapper objects,
    with name-based lookup functionality.

    Parameters
    ----------
    *args : AlgorithmWrapper
        Initial AlgorithmWrapper instances

    Raises
    ------
    TypeError
        If non-AlgorithmWrapper instance is added
    ValueError
        If duplicate algorithm names are found
    """
    def __init__(self, *args):
        super().__init__()
        for item in args:
            self.append(item)

    def append(self, item: algorithm_wrapper.AlgorithmWrapper) -> None:
        """Add an AlgorithmWrapper to the collection.

        Parameters
        ----------
        item : AlgorithmWrapper
            Algorithm wrapper to add

        Raises
        ------
        TypeError
            If item is not an AlgorithmWrapper
        ValueError
            If algorithm name already exists in collection
        """
        if not isinstance(item, algorithm_wrapper.AlgorithmWrapper):
            raise TypeError(
                "AlgorithmCollection only accepts AlgorithmWrapper instances"
            )
        if any(wrapper.name == item.name for wrapper in self):
            raise ValueError(
                f"Duplicate algorithm name: {item.name}"
            )
        super().append(item)

    def __getitem__(self, key: Union[int, str]) -> algorithm_wrapper.AlgorithmWrapper:
        """Get algorithm by index or name.

        Parameters
        ----------
        key : int or str
            Index or name of algorithm to retrieve

        Returns
        -------
        AlgorithmWrapper
            The requested algorithm wrapper

        Raises
        ------
        KeyError
            If string key doesn't match any algorithm name
        TypeError
            If key is neither int nor str
        """
        if isinstance(key, int):
            return super().__getitem__(key)

        if isinstance(key, str):
            for wrapper in self:
                if wrapper.name == key:
                    return wrapper
            raise KeyError(f"No algorithm found with name: {key}")

        raise TypeError(
            f"Index must be an integer or string, got {type(key).__name__}"
        )

    def export_params(self) -> list[Dict[str, Any]]:
        """
        Export algorithm configuration data for rerun functionality.
        
        Detects built-in algorithm collections and exports custom algorithms with
        their class and parameter definitions to enable exact rerun functionality.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of algorithm configurations that can be used to recreate the AlgorithmCollection
        """        
        regression_names = {wrapper.name for wrapper in regression_algorithms.REGRESSION_ALGORITHMS}
        classification_names = {wrapper.name for wrapper in classification_algorithms.CLASSIFICATION_ALGORITHMS}
        
        found_regression_algorithms = set()
        found_classification_algorithms = set()
        custom_algorithms = []
        
        for wrapper in self:
            if wrapper.name in regression_names:
                found_regression_algorithms.add(wrapper.name)
            elif wrapper.name in classification_names:
                found_classification_algorithms.add(wrapper.name)
            else:
                custom_algorithms.append(wrapper.export_config())
        
        result = []
        
        if found_regression_algorithms == regression_names:
            result.append({
                "type": "builtin_collection",
                "collection": "brisk.REGRESSION_ALGORITHMS"
            })
        elif found_regression_algorithms:
            for name in found_regression_algorithms:
                result.append({
                    "type": "builtin_algorithm",
                    "collection": "brisk.REGRESSION_ALGORITHMS",
                    "name": name
                })
        
        if found_classification_algorithms == classification_names:
            result.append({
                "type": "builtin_collection", 
                "collection": "brisk.CLASSIFICATION_ALGORITHMS"
            })
        elif found_classification_algorithms:
            for name in found_classification_algorithms:
                result.append({
                    "type": "builtin_algorithm",
                    "collection": "brisk.CLASSIFICATION_ALGORITHMS", 
                    "name": name
                })
        
        for custom_config in custom_algorithms:
            custom_config["type"] = "custom_algorithm"
            result.append(custom_config)
        
        return result
