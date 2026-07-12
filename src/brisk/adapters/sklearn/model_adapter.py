"""Sklearn adapter for the algorithm system.

Wraps scikit-learn estimators behind the AlgorithmWrapperPort interface,
enabling sklearn models to be used through brisk's port abstractions.
"""

from typing import Any, cast

from sklearn import base

from brisk.ports import algorithm
from brisk.reporting import formatting


class SklearnAlgorithmWrapper:
    """A wrapper for scikit-learn algorithm implementations.

    Provides methods to instantiate models with default or tuned parameters
    and manages hyperparameter grids for model tuning. Enforces that all
    algorithm classes are scikit-learn BaseEstimator subclasses.

    Parameters
    ----------
    name : str
        Unique identifier for the algorithm used in configurations.
    display_name : str
        Human-readable name for display purposes in reports and UI.
    algorithm_class : type[base.BaseEstimator]
        The scikit-learn algorithm class to be instantiated.
    default_params : dict, optional
        Default parameters for model instantiation.
    hyperparam_grid : dict, optional
        Grid of parameters for hyperparameter tuning.

    Raises
    ------
    TypeError
        If name, display_name, default_params, or hyperparam_grid are not
        of the expected types.
    ValueError
        If algorithm_class is not a scikit-learn BaseEstimator subclass.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> wrapper = SklearnAlgorithmWrapper(
    ...     name="ridge",
    ...     display_name="Ridge Regression",
    ...     algorithm_class=Ridge,
    ...     default_params={"fit_intercept": True},
    ...     hyperparam_grid={"alpha": [0.1, 0.5, 1.0]},
    ... )
    >>> model = wrapper.instantiate()
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        algorithm_class: type[base.BaseEstimator],
        default_params: dict[str, Any] | None = None,
        hyperparam_grid: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(display_name, str):
            raise TypeError("display_name must be a string")
        if not isinstance(algorithm_class, type):
            raise TypeError("algorithm_class must be a class")
        if not issubclass(algorithm_class, base.BaseEstimator):
            raise ValueError(
                "'algorithm_class' is not a 'BaseEstimator' subclass"
            )

        self.name = name
        self.display_name = display_name
        self.algorithm_class = algorithm_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

        if not isinstance(self.default_params, dict):
            raise TypeError("default_params must be a dictionary")
        if not isinstance(self.hyperparam_grid, dict):
            raise TypeError("hyperparam_grid must be a dictionary")

    def instantiate(self) -> algorithm.ModelPort:
        """Instantiate model with default parameters.

        Returns
        -------
        algorithm.ModelPort
            A fitted-ready model with ``wrapper_name`` set to ``self.name``.

        Examples
        --------
        >>> model = wrapper.instantiate()
        >>> print(model.wrapper_name)
        'ridge'
        """
        model = self.algorithm_class(**self.default_params)
        setattr(model, "wrapper_name", self.name)
        return cast(algorithm.ModelPort, model)

    def instantiate_tuned(
        self,
        best_params: dict[str, Any],
    ) -> algorithm.ModelPort:
        """Instantiate model with tuned hyperparameters.

        Default parameters not present in ``best_params`` are preserved.

        Parameters
        ----------
        best_params : dict
            Tuned hyperparameters from a hyperparameter search.

        Returns
        -------
        algorithm.ModelPort
            A fitted-ready model with ``wrapper_name`` set to ``self.name``.
        """
        merged = {
            key: value for key, value in self.default_params.items()
            if key not in best_params
        }
        merged.update(best_params)
        model = self.algorithm_class(**merged)
        setattr(model, "wrapper_name", self.name)
        return cast(algorithm.ModelPort, model)

    def get_hyperparam_grid(self) -> dict[str, Any]:
        """Return a copy of the hyperparameter grid for tuning.

        Returns
        -------
        dict[str, Any]
            Mapping of parameter names to candidate value lists.
        """
        return self.hyperparam_grid.copy()

    def to_markdown(self) -> str:
        """Generate a markdown representation of this algorithm's configuration.

        Returns
        -------
        str
            Markdown-formatted string with name, class, parameters, and grid.
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
            "```",
        ]
        return "\n".join(md)

    def export_config(self) -> dict[str, Any]:
        """Export a serializable configuration for rerun functionality.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary containing algorithm name, class
            info, default parameters, and hyperparameter grid.
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "algorithm_class_module": self.algorithm_class.__module__,
            "algorithm_class_name": self.algorithm_class.__name__,
            "default_params": self._serialize_params(self.default_params),
            "hyperparam_grid": self._serialize_params(self.hyperparam_grid),
        }

    def _serialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serialize parameters, handling nested sklearn estimators.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters dictionary to serialize.

        Returns
        -------
        dict[str, Any]
            Serialized parameters with complex objects converted to dicts.
        """
        serialized: dict[str, Any] = {}
        for key, value in params.items():
            if hasattr(value, "__module__") and hasattr(value, "__class__"):
                if hasattr(value, "get_params"):
                    serialized[key] = {
                        "_brisk_object_type": "sklearn_estimator",
                        "module": value.__class__.__module__,
                        "class_name": value.__class__.__name__,
                        "params": value.get_params(deep=True),
                    }
                else:
                    serialized[key] = {
                        "_brisk_object_type": "object",
                        "module": value.__class__.__module__,
                        "class_name": value.__class__.__name__,
                        "repr": repr(value),
                    }
            elif isinstance(value, list):
                serialized[key] = self._serialize_list(value)
            else:
                serialized[key] = value

        return serialized

    def _serialize_list(self, lst: list[Any]) -> list[Any]:
        """Serialize a list, handling tuples with sklearn estimators.

        Parameters
        ----------
        lst : list[Any]
            List to serialize.

        Returns
        -------
        list[Any]
            Serialized list with estimator tuples converted to dicts.
        """
        serialized_list: list[Any] = []
        for item in lst:
            if isinstance(item, tuple) and len(item) == 2:
                name, estimator = item
                if (
                    hasattr(estimator, "__module__")
                    and hasattr(estimator, "get_params")
                ):
                    serialized_list.append([
                        name,
                        {
                            "_brisk_object_type": "sklearn_estimator",
                            "module": estimator.__class__.__module__,
                            "class_name": estimator.__class__.__name__,
                            "params": estimator.get_params(deep=True),
                        },
                    ])
                else:
                    serialized_list.append(list(item))
            else:
                serialized_list.append(item)

        return serialized_list

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        """Update default_params or hyperparam_grid via bracket syntax.

        Parameters
        ----------
        key : str
            Either ``'default_params'`` or ``'hyperparam_grid'``.
        value : dict[str, Any]
            Dictionary of parameters to merge into the target.

        Raises
        ------
        TypeError
            If value is not a dictionary.
        KeyError
            If key is not ``'default_params'`` or ``'hyperparam_grid'``.

        Examples
        --------
        >>> wrapper["default_params"] = {"max_iter": 1000}
        >>> wrapper["hyperparam_grid"] = {"C": [0.1, 1.0, 10.0]}
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
