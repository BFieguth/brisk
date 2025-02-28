"""Provides the AlgorithmWrapper class for managing machine learning algorithms.

This module provides classes for managing machine learning algorithms, their
parameters, and hyperparameter grids. It includes functionality for model
instantiation and parameter tuning.
"""

from typing import Any, Dict, Optional, Type, Union

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
        self.name = name
        self.display_name = display_name
        self.algorithm_class = algorithm_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

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
        If max_iter is specified in default_params, it will be preserved
        in the tuned parameters.
        """
        if "max_iter" in self.default_params:
            best_params["max_iter"] = self.default_params["max_iter"]
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

    def append(self, item: AlgorithmWrapper) -> None:
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
