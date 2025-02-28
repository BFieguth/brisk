"""Provides the AlgorithmWrapper class for managing and instantiating machine 
learning algorithms.

Exports:
    - AlgorithmWrapper: A class to handle model instantiation and tuning, 
        allowing for easy access to models and their hyperparameter grids.
"""

from typing import Any, Dict, Optional, Type, Union

from brisk.utility.utility import format_dict

class AlgorithmWrapper:
    """A wrapper class for machine learning algorithms.

    This class provides methods to easily instantiate models with default 
    parameters or tuned hyperparameters. It also manages hyperparameter grids 
    for tuning.

    Attributes:
        name (str): The name of the algorithm.
        
        algorithm_class (Type): The class of the algorithm.
        
        default_params (dict): The default parameters for the algorithm.
        
        hyperparam_grid (dict): The hyperparameter grid for tuning the 
        algorithm.
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
        self.name = name
        self.display_name = display_name
        self.algorithm_class = algorithm_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

    def __setitem__(self, key, value):
        """Override item setting to update default_params or hyperparam_grid."""
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
        """Instantiates the model with the default parameters and wrapper name.

        Returns:
            Any: An instance of the model with the provided default parameters.
        """
        model = self.algorithm_class(**self.default_params)
        setattr(model, "wrapper_name", self.name)
        return model

    def instantiate_tuned(self, best_params: Dict[str, Any]) -> Any:
        """Instantiates a new model with the tuned parameters.
        
        If max_iter is specified in the default_params, it will be included in 
        the tuned parameters.

        Args:
            best_params (Dict[str, Any]): The tuned hyperparameters.

        Returns:
            Any: A new instance of the model with the tuned hyperparameters.
        """
        if "max_iter" in self.default_params:
            best_params["max_iter"] = self.default_params["max_iter"]
        model = self.algorithm_class(**best_params)
        setattr(model, "wrapper_name", self.name)
        return model

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


class AlgorithmCollection(list):
    """
    A custom collection for AlgorithmWrapper objects that allows list and
    dict-like access.
    """
    def __init__(self, *args):
        super().__init__()
        for item in args:
            self.append(item)

    def append(self, item: AlgorithmWrapper) -> None:
        """Enforce type checks before appending an item."""
        if not isinstance(item, AlgorithmWrapper):
            raise TypeError(
                "AlgorithmCollection only accepts AlgorithmWrapper instances"
            )
        if any(wrapper.name == item.name for wrapper in self):
            raise ValueError(
                f"Duplicate algorithm name: {item.name}"
            )
        super().append(item)

    def __getitem__(self, key: Union[int, str]) -> AlgorithmWrapper:
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
