"""Provides the AlgorithmWrapper class for managing and instantiating machine learning algorithms.

Exports:
    - AlgorithmWrapper: A class to handle model instantiation and tuning, 
        allowing for easy access to models and their hyperparameter grids.
"""

from typing import Any, Dict, Optional, Type

from brisk.utility.utility import format_dict

class AlgorithmWrapper:
    """A wrapper class for machine learning algorithms.

    This class provides methods to easily instantiate models with default parameters 
    or tuned hyperparameters. It also manages hyperparameter grids for tuning.

    Attributes:
        name (str): The name of the algorithm.
        algorithm_class (Type): The class of the algorithm.
        default_params (dict): The default parameters for the algorithm.
        hyperparam_grid (dict): The hyperparameter grid for tuning the algorithm.
    """
    def __init__(
        self, 
        name: str, 
        display_name: str,
        algorithm_class: Type, 
        default_params: Optional[Dict[str, Any]] = None, 
        hyperparam_grid: Optional[Dict[str, Any]] = None
    ):
        """Initializes the AlgorithmWrapper with a model class.

        Args:
            name (str): The name of the model.
            algorithm_class (Type): The class of the algorithm to be instantiated.
            default_params (Optional[Dict[str, Any]]): The default parameters to 
                pass to the model during instantiation.
            hyperparam_grid (Optional[Dict[str, Any]]): The hyperparameter grid 
                for model tuning.
        """
        self.name = name
        self.display_name = display_name
        self.algorithm_class = algorithm_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

    def __setitem__(self, key, value):
        """Override item setting to update default_params or hyperparam_grid."""
        if key == 'default_params':
            self.default_params.update(value)
        elif key == 'hyperparam_grid':
            self.hyperparam_grid.update(value)
        else:
            raise KeyError(f"Invalid key: {key}. Allowed keys: 'default_params', 'hyperparam_grid'")

    def instantiate(self) -> Any:
        """Instantiates the model with the default parameters.

        Returns:
            Any: An instance of the model with the provided default parameters.
        """
        return self.algorithm_class(**self.default_params)

    def instantiate_tuned(self, best_params: Dict[str, Any]) -> Any:
        """Instantiates a new model with the tuned parameters, includes max_iter if present.

        Args:
            best_params (Dict[str, Any]): The tuned hyperparameters.

        Returns:
            Any: A new instance of the model with the tuned hyperparameters.
        """
        if 'max_iter' in self.default_params:
            best_params['max_iter'] = self.default_params['max_iter']
        return self.algorithm_class(**best_params)

    def get_hyperparam_grid(self) -> Dict[str, Any]:
        """Returns the hyperparameter grid for the model.

        Returns:
            Dict[str, Any]: A dictionary representing the hyperparameter grid.
        """
        return self.hyperparam_grid

    def to_markdown(self) -> str:
        """Creates a markdown representation of the algorithm configuration.
        
        Returns:
            str: Markdown formatted string describing the algorithm
        """
        md = [
            f"### {self.display_name} (`{self.name}`)",
            "",
            f"- **Algorithm Class**: `{self.algorithm_class.__name__}`",
            "",
            "**Default Parameters:**",
            "```python",
            format_dict(self.default_params),
            "```",
            "",
            "**Hyperparameter Grid:**",
            "```python",
            format_dict(self.hyperparam_grid),
            "```"
        ]
        return "\n".join(md)
    