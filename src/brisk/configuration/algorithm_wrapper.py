"""Provides the AlgorithmWrapper class for managing machine learning algorithms.

This module provides classes for managing machine learning algorithms, their
parameters, and hyperparameter grids. It includes functionality for model
instantiation, parameter tuning, and configuration serialization for
reproducible experiments.

Classes
-------
AlgorithmWrapper
    A wrapper class for machine learning algorithms that provides methods
    to instantiate models with default or tuned parameters and manages
    hyperparameter grids for model tuning.
"""

from typing import Any, Dict, Optional, Type

from brisk.reporting import formatting

class AlgorithmWrapper:
    """A wrapper class for machine learning algorithms.

    Provides methods to instantiate models with default or tuned parameters
    and manages hyperparameter grids for model tuning. This class serves as
    a bridge between Brisk's configuration system and scikit-learn algorithms,
    enabling consistent parameter management and model instantiation.

    Parameters
    ----------
    name : str
        Unique identifier for the algorithm used in configurations
    display_name : str
        Human-readable name for display purposes in reports and UI
    algorithm_class : Type
        The scikit-learn algorithm class to be instantiated
    default_params : dict, optional
        Default parameters for model instantiation, by default None
    hyperparam_grid : dict, optional
        Grid of parameters for hyperparameter tuning, by default None

    Attributes
    ----------
    name : str
        Algorithm identifier used in configurations
    display_name : str
        Human-readable name for display purposes
    algorithm_class : Type
        The scikit-learn algorithm class
    default_params : dict
        Default parameters for model instantiation
    hyperparam_grid : dict
        Hyperparameter grid for tuning

    Notes
    -----
    The AlgorithmWrapper class enforces that all algorithm classes must
    be from scikit-learn to ensure compatibility with Brisk's evaluation
    and reporting systems.

    Examples
    --------
    Create a wrapper for Linear Regression:
        >>> from sklearn.linear_model import LinearRegression
        >>> wrapper = AlgorithmWrapper(
        ...     name="ridge",
        ...     display_name="Ridge Regression",
        ...     algorithm_class=Ridge,
        ...     default_params={"fit_intercept": True},
        ...     hyperparam_grid={"alpha": range(0, 1, 0.1)}
        ... )

    Instantiate a model:
        >>> model = wrapper.instantiate()

    Raises
    ------
    TypeError
        If name, display_name, default_params, or hyperparam_grid are not
        of the expected types
    ValueError
        If algorithm_class is not from scikit-learn
    """
    def __init__(
        self,
        name: str,
        display_name: str,
        algorithm_class: Type,
        default_params: Optional[Dict[str, Any]] = None,
        hyperparam_grid: Optional[Dict[str, Any]] = None
    ):
        """Initialize the AlgorithmWrapper with an algorithm class.

        Creates a new AlgorithmWrapper instance with validation of all
        parameters. Ensures the algorithm class is from scikit-learn and
        validates parameter types.

        Parameters
        ----------
        name : str
            Unique identifier for the algorithm used in configurations
        display_name : str
            Human-readable name for display purposes in reports and UI
        algorithm_class : Type
            The scikit-learn algorithm class to be instantiated
        default_params : dict, optional
            Default parameters for model instantiation, by default None
        hyperparam_grid : dict, optional
            Grid of parameters for hyperparameter tuning, by default None

        Raises
        ------
        TypeError
            If name, display_name, default_params, or hyperparam_grid are not
            of the expected types
        ValueError
            If algorithm_class is not from scikit-learn module

        Notes
        -----
        The constructor performs comprehensive validation:
        - Ensures name and display_name are strings
        - Validates algorithm_class is a class from scikit-learn
        - Converts None parameters to empty dictionaries
        - Validates parameter dictionaries are actually dictionaries
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
        """Update parameter dictionaries using dictionary-style assignment.

        Provides a convenient way to update either default_params or
        hyperparam_grid using dictionary-style syntax.

        Parameters
        ----------
        key : str
            Either 'default_params' or 'hyperparam_grid' to specify
            which parameter dictionary to update
        value : dict
            Dictionary of parameters to merge into the specified
            parameter dictionary

        Raises
        ------
        TypeError
            If value is not a dictionary
        KeyError
            If key is not 'default_params' or 'hyperparam_grid'

        Examples
        --------
        Update default parameters:
            >>> wrapper["default_params"] = {"max_iter": 1000}

        Update hyperparameter grid:
            >>> wrapper["hyperparam_grid"] = {"C": [0.1, 1.0, 10.0]}

        Notes
        -----
        This method uses dict.update() to merge the provided parameters
        with existing parameters, allowing for incremental updates.
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

        Creates a new instance of the algorithm class using the current
        default parameters. Adds a wrapper_name attribute to the model
        for identification purposes.

        Returns
        -------
        Any
            Model instance with default parameters and wrapper_name attribute

        Notes
        -----
        The instantiated model will have a 'wrapper_name' attribute set
        to the algorithm's name, which can be useful for identification
        in evaluation and reporting contexts.

        Examples
        --------
        Create and instantiate a model:
            >>> wrapper = AlgorithmWrapper(
            ...     name="linear",
            ...     display_name="Linear Regression",
            ...     algorithm_class=LinearRegression
            ... )
            >>> model = wrapper.instantiate()
            >>> print(model.wrapper_name)  # "linear"
        """
        model = self.algorithm_class(**self.default_params)
        setattr(model, "wrapper_name", self.name)
        return model

    def instantiate_tuned(self, best_params: Dict[str, Any]) -> Any:
        """Instantiate model with tuned parameters.

        Creates a new instance of the algorithm class using tuned
        hyperparameters. Automatically includes any default parameters
        that were not part of the tuning process.

        Parameters
        ----------
        best_params : dict
            Tuned hyperparameters from a hyperparameter search process

        Returns
        -------
        Any
            Model instance with tuned parameters and wrapper_name attribute

        Raises
        ------
        TypeError
            If best_params is not a dictionary

        Notes
        -----
        If a parameter is set in default_params but not in hyperparam_grid,
        the default value will be preserved in the tuned parameters. This
        ensures that all necessary parameters are included in the final model.

        Examples
        --------
        Instantiate with tuned parameters:
            >>> best_params = {"C": 1.0, "gamma": "scale"}
            >>> model = wrapper.instantiate_tuned(best_params)
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

        Returns the current hyperparameter grid dictionary used for
        hyperparameter tuning.

        Returns
        -------
        Dict[str, Any]
            Current hyperparameter grid dictionary

        Examples
        --------
        Get hyperparameter grid for tuning:
            >>> grid = wrapper.get_hyperparam_grid()
            >>> print(grid)
            >>> {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
        """
        return self.hyperparam_grid

    def to_markdown(self) -> str:
        """Create markdown representation of algorithm configuration.

        Generates a formatted markdown string containing the algorithm's
        name, class, default parameters, and hyperparameter grid. Useful
        for documentation and reporting purposes.

        Returns
        -------
        str
            Markdown formatted string containing algorithm information

        Notes
        -----
        The markdown output includes:
        - Algorithm display name and identifier
        - Algorithm class name
        - Formatted default parameters code block
        - Formatted hyperparameter grid code block

        Examples
        --------
        Generate markdown documentation:
            >>> md = wrapper.to_markdown()
            >>> print(md)
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
        """Export this AlgorithmWrapper's configuration for rerun functionality.
        
        Creates a serialized configuration dictionary that can be used to
        recreate this AlgorithmWrapper instance. Handles complex objects
        like scikit-learn estimators by serializing their parameters.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary that can be used to recreate this
            AlgorithmWrapper instance

        Notes
        -----
        The exported configuration includes:
        - Algorithm name and display name
        - Algorithm class module and name
        - Serialized default parameters
        - Serialized hyperparameter grid

        Complex objects like scikit-learn estimators are serialized with
        their module, class name, and parameters for proper reconstruction.

        Examples
        --------
        Export configuration for saving:
            >>> config = wrapper.export_config()
            >>> # Save config to file or database
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
        
        Recursively serializes parameter dictionaries, converting complex
        objects like scikit-learn estimators into serializable representations
        that can be reconstructed later.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters dictionary to serialize
            
        Returns
        -------
        Dict[str, Any]
            Serialized parameters dictionary

        Notes
        -----
        This method handles several types of objects:
        - Scikit-learn estimators: Serialized with module, class, and parameters
        - Other objects with __module__ and __class__: Serialized with module,
        class, and repr
        - Lists: Recursively serialized using _serialize_list
        - Simple types: Passed through unchanged

        The serialization preserves enough information to reconstruct
        the original objects during configuration loading.
        """
        serialized = {}
        for key, value in params.items():
            if hasattr(value, "__module__") and hasattr(value, "__class__"):
                if hasattr(value, "get_params"):
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
        """Serialize a list, handling tuples with sklearn estimators.
        
        Recursively serializes list items, with special handling for
        tuples containing scikit-learn estimators (common in pipeline
        configurations).

        Parameters
        ----------
        lst : list
            List to serialize
            
        Returns
        -------
        list
            Serialized list with complex objects converted to dictionaries

        Notes
        -----
        This method handles:
        - Tuples with (name, estimator) pairs: Converts to
        [name, serialized_estimator]
        - Regular items: Passes through unchanged
        - Nested lists: Recursively processes

        The special handling for tuples is important for pipeline
        configurations where estimators are often stored as (name, estimator)
        pairs.
        """
        serialized_list = []
        for item in lst:
            if isinstance(item, tuple) and len(item) == 2:
                name, estimator = item
                if (
                    hasattr(estimator, "__module__") and
                    hasattr(estimator, "get_params")
                ):
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
