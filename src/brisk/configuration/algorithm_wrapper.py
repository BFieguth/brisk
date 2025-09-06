"""Provides the AlgorithmWrapper class for managing machine learning algorithms.

This module provides classes for managing machine learning algorithms, their
parameters, and hyperparameter grids. It includes functionality for model
instantiation and parameter tuning.
"""

from typing import Any, Dict, Optional, Type

from brisk.reporting import formatting

class AlgorithmWrapper:
    """A wrapper class for machine learning algorithms.

    Provides methods to instantiate models with default or tuned parameters
    and manages hyperparameter grids for model tuning.

    Parameters
    ----------
    name : str
        Identifier for the algorithm
    display_name : str
        Human-readable name for display purposes
    algorithm_class : Type
        The class of the algorithm to be instantiated
    default_params : dict, optional
        Default parameters for model instantiation, by default None
    hyperparam_grid : dict, optional
        Grid of parameters for hyperparameter tuning, by default None

    Attributes
    ----------
    name : str
        Algorithm identifier
    display_name : str
        Human-readable name
    algorithm_class : Type
        The algorithm class
    default_params : dict
        Current default parameters
    hyperparam_grid : dict
        Current hyperparameter grid
    """
    def __init__(
        self,
        name: str,
        display_name: str,
        algorithm_class: Type,
        default_params: Optional[Dict[str, Any]] = None,
        hyperparam_grid: Optional[Dict[str, Any]] = None
    ):
        """Initializes the AlgorithmWrapper with a algorithm class.

        Args:
            name (str): The name of the algorithm.

            algorithm_class (Type): The class of the algorithm to be
            instantiated.

            default_params (Optional[Dict[str, Any]]): The default parameters to
            pass to the algorithm during instantiation.

            hyperparam_grid (Optional[Dict[str, Any]]): The hyperparameter grid
            for model tuning.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(display_name, str):
            raise TypeError("display_name must be a string")
        if not isinstance(algorithm_class, Type):
            raise TypeError("algorithm_class must be a class")
        if not algorithm_class.__module__.startswith("sklearn"):
            raise ValueError("algorithm_class must be from sklearn")

        self.name = name
        self.display_name = display_name
        self.algorithm_class = algorithm_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

        if not isinstance(self.default_params, dict):
            raise TypeError("default_params must be a dictionary")
        if not isinstance(self.hyperparam_grid, dict):
            raise TypeError("hyperparam_grid must be a dictionary")

    def __setitem__(self, key: str, value: dict) -> None:
        """Update parameter dictionaries.

        Parameters
        ----------
        key : str
            Either 'default_params' or 'hyperparam_grid'
        value : dict
            Parameters to update

        Raises
        ------
        KeyError
            If key is not 'default_params' or 'hyperparam_grid'
        """
        if not isinstance(value, dict):
            raise TypeError(f"value must be a dict, got {type(value)}")

        if key == "default_params":
            self.default_params.update(value)
        elif key == "hyperparam_grid":
            self.hyperparam_grid.update(value)
        else:
            raise KeyError(
                f"Invalid key: {key}. "
                "Allowed keys: 'default_params', 'hyperparam_grid'"
            )

    def instantiate(self) -> Any:
        """Instantiate model with default parameters.

        Returns
        -------
        Any
            Model instance with default parameters and wrapper name attribute
        """
        model = self.algorithm_class(**self.default_params)
        setattr(model, "wrapper_name", self.name)
        return model

    def instantiate_tuned(self, best_params: Dict[str, Any]) -> Any:
        """Instantiate model with tuned parameters.

        Parameters
        ----------
        best_params : dict
            Tuned hyperparameters

        Returns
        -------
        Any
            Model instance with tuned parameters and wrapper name attribute

        Notes
        -----
        If a parameter is set in default_params but not in hyperparam_grid,
        the default value will be preserved in the tuned parameters.
        """
        if not isinstance(best_params, dict):
            raise TypeError("best_params must be a dictionary")
        missing_defaults = [
            param for param in self.default_params
            if param not in best_params.keys()
        ]
        for param in missing_defaults:
            best_params[param] = self.default_params[param]
        model = self.algorithm_class(**best_params)
        setattr(model, "wrapper_name", self.name)
        return model

    def get_hyperparam_grid(self) -> Dict[str, Any]:
        """Get the hyperparameter grid.

        Returns
        -------
        dict
            Current hyperparameter grid
        """
        return self.hyperparam_grid

    def to_markdown(self) -> str:
        """Create markdown representation of algorithm configuration.

        Returns
        -------
        str
            Markdown formatted string containing algorithm name and class,
            default parameters, and hyperparameter grid.
        """
        md = [
            f"### {self.display_name} (`{self.name}`)",
            "",
            f"- **Algorithm Class**: `{self.algorithm_class.__name__}`",
            "",
            "**Default Parameters:**",
            "```python",
            formatting.format_dict(self.default_params),
            "```",
            "",
            "**Hyperparameter Grid:**",
            "```python",
            formatting.format_dict(self.hyperparam_grid),
            "```"
        ]
        return "\n".join(md)

    def export_config(self) -> Dict[str, Any]:
        """
        Export this AlgorithmWrapper's configuration for rerun functionality.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary that can be used to recreate this AlgorithmWrapper
        """
        config = {
            "name": self.name,
            "display_name": self.display_name,
            "algorithm_class_module": self.algorithm_class.__module__,
            "algorithm_class_name": self.algorithm_class.__name__,
            "default_params": self._serialize_params(self.default_params),
            "hyperparam_grid": self._serialize_params(self.hyperparam_grid)
        }
        
        return config

    def _serialize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize parameters, handling complex objects like sklearn estimators.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to serialize
            
        Returns
        -------
        Dict[str, Any]
            Serialized parameters
        """
        serialized = {}
        
        for key, value in params.items():
            if hasattr(value, '__module__') and hasattr(value, '__class__'):
                if hasattr(value, 'get_params'):
                    serialized[key] = {
                        "_brisk_object_type": "sklearn_estimator",
                        "module": value.__class__.__module__,
                        "class_name": value.__class__.__name__,
                        "params": value.get_params()
                    }
                else:
                    serialized[key] = {
                        "_brisk_object_type": "object",
                        "module": value.__class__.__module__,
                        "class_name": value.__class__.__name__,
                        "repr": repr(value)
                    }
            elif isinstance(value, list):
                serialized[key] = self._serialize_list(value)
            else:
                serialized[key] = value
                
        return serialized

    def _serialize_list(self, lst: list) -> list:
        """
        Serialize a list, handling tuples with sklearn estimators.
        
        Parameters
        ----------
        lst : list
            List to serialize
            
        Returns
        -------
        list
            Serialized list
        """
        serialized_list = []
        
        for item in lst:
            if isinstance(item, tuple) and len(item) == 2:
                name, estimator = item
                if hasattr(estimator, '__module__') and hasattr(estimator, 'get_params'):
                    serialized_list.append([
                        name,
                        {
                            "_brisk_object_type": "sklearn_estimator",
                            "module": estimator.__class__.__module__,
                            "class_name": estimator.__class__.__name__,
                            "params": estimator.get_params()
                        }
                    ])
                else:
                    serialized_list.append(list(item))
            else:
                serialized_list.append(item)
                
        return serialized_list
