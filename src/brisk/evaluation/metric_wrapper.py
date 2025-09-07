"""Provides the MetricWrapper class for wrapping metric functions.

This module provides the MetricWrapper class, which wraps metric functions from
scikit-learn or defined by the user. It allows for easy application of default
parameters to metrics and provides additional metadata for use in the Brisk
framework.
"""
import copy
import functools
import inspect
import importlib
from typing import Callable, Any, Optional, Dict

from sklearn import metrics

class MetricWrapper:
    """A wrapper for metric functions with default parameters and metadata.

    Wraps metric functions and provides methods to update parameters and
    retrieve the metric function with applied parameters. Also handles display
    names and abbreviations for reporting. Supports both scikit-learn metrics
    and custom user-defined functions.

    Parameters
    ----------
    name : str
        Name of the metric
    func : Callable
        Metric function to wrap
    display_name : str
        Human-readable name for display in reports and plots
    greater_is_better : bool
        Whether higher values indicate better performance
    abbr : str, optional
        Abbreviation for the metric, by default None
    **default_params : Any
        Default parameters for the metric function

    Attributes
    ----------
    name : str
        Name of the metric
    func : Callable
        The wrapped metric function (may be modified to accept split_metadata)
    display_name : str
        Human-readable display name
    abbr : str
        Abbreviation (defaults to name if not provided)
    greater_is_better : bool
        Whether higher values indicate better performance
    params : dict
        Current parameters for the metric
    _func_with_params : Callable
        Metric function with parameters applied
    scorer : Callable
        Scikit-learn scorer created from the metric

    Notes
    -----
    The MetricWrapper automatically ensures that wrapped functions can accept
    a split_metadata parameter, even if the original function doesn't support
    it. This allows for consistent parameter passing across all metrics.

    Examples
    --------
    Create a wrapper for mean squared error:
        >>> from sklearn.metrics import mean_squared_error
        >>> wrapper = MetricWrapper(
        ...     name="mse",
        ...     func=mean_squared_error,
        ...     display_name="Mean Squared Error",
        ...     greater_is_better=False
        ... )

    Create a custom metric wrapper:
        >>> def custom_metric(y_true, y_pred):
        ...     return sum(abs(y_true - y_pred)) / len(y_true)
        >>> wrapper = MetricWrapper(
        ...     name="custom_mae",
        ...     func=custom_metric,
        ...     display_name="Custom MAE",
        ...     greater_is_better=False
        ... )
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        display_name: str,
        greater_is_better: bool,
        abbr: Optional[str] = None,
        **default_params: Any
    ):
        """Initialize MetricWrapper with metric function and parameters.

        Parameters
        ----------
        name : str
            Name of the metric
        func : Callable
            Metric function to wrap
        display_name : str
            Human-readable name for display
        greater_is_better : bool
            Whether higher values indicate better performance
        abbr : str, optional
            Abbreviation for the metric, by default None
        **default_params : Any
            Default parameters for the metric function

        Notes
        -----
        The constructor automatically ensures the function can accept
        split_metadata and sets up the initial parameter configuration.
        """
        self.name = name
        self._original_func = func
        self.func = self._ensure_split_metadata_param(func)
        self.display_name = display_name
        self.abbr = abbr if abbr else name
        self.greater_is_better = greater_is_better
        self.params = default_params
        self.params["split_metadata"] = {}
        self._apply_params()

    def _apply_params(self) -> None:
        """Apply current parameters to function and scorer.

        Creates a partial function with the current parameters and updates the
        scikit-learn scorer. This method is called whenever parameters are
        updated to ensure consistency between the function and scorer.

        Returns
        -------
        None

        Notes
        -----
        This method uses functools.partial to create a new function with
        the current parameters applied, and creates a new scikit-learn
        scorer with the same parameters.
        """
        self._func_with_params = functools.partial(self.func, **self.params)
        self.scorer = metrics.make_scorer(
            self.func,
            greater_is_better=self.greater_is_better,
            **self.params
        )

    def set_params(self, **params: Any) -> None:
        """Update parameters for the metric function and scorer.

        Updates the internal parameter dictionary and reapplies parameters
        to both the function and scorer.

        Parameters
        ----------
        **params : Any
            New parameters to update or add

        Returns
        -------
        None

        Notes
        -----
        This method updates the internal params dictionary and then calls
        _apply_params to ensure both the function and scorer are updated
        with the new parameters.
        """
        self.params.update(params)
        self._apply_params()

    def get_func_with_params(self) -> Callable:
        """Get the metric function with current parameters applied.

        Returns a deep copy of the metric function with all current
        parameters applied. This ensures that the returned function
        is independent and can be safely used in parallel operations.

        Returns
        -------
        Callable
            Deep copy of the metric function with parameters applied

        Notes
        -----
        The returned function is a deep copy to prevent issues with
        shared state in parallel or concurrent operations.
        """
        return copy.deepcopy(self._func_with_params)

    def _ensure_split_metadata_param(self, func: Callable) -> Callable:
        """Ensure metric function accepts split_metadata as a keyword argument.

        Wraps the function if necessary to accept the split_metadata parameter
        without affecting the original functionality. This allows for consistent
        parameter passing across all metrics in the Brisk framework.

        Parameters
        ----------
        func : Callable
            Function to check/wrap

        Returns
        -------
        Callable
            Original or wrapped function that accepts split_metadata

        Notes
        -----
        If the original function already accepts split_metadata, it is
        returned unchanged. Otherwise, a wrapper function is created that
        accepts split_metadata but ignores it, passing only the original
        parameters to the underlying function.

        The wrapper preserves the original function's name, qualname, and
        docstring for proper introspection.
        """
        sig = inspect.signature(func)

        if "split_metadata" not in sig.parameters:
            def wrapped_func(y_true, y_pred, split_metadata=None, **kwargs): # pylint: disable=unused-argument
                return func(y_true, y_pred, **kwargs)

            wrapped_func.__name__ = func.__name__
            wrapped_func.__qualname__ = func.__qualname__
            wrapped_func.__doc__ = func.__doc__
            return wrapped_func
        return func

    def export_config(self) -> Dict[str, Any]:
        """Export this MetricWrapper's configuration for rerun functionality.
        
        Exports the complete configuration needed to recreate this MetricWrapper
        instance. Handles both built-in scikit-learn functions and custom
        user-defined functions by detecting the function source and exporting
        appropriate reconstruction information.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary that can be used to recreate this
            MetricWrapper instance

        Notes
        -----
        The export process intelligently detects the function type:
        - "imported": For functions from external libraries (e.g., sklearn)
        - "local": For custom functions defined in the project
        - "unknown": For functions that cannot be properly identified

        For imported functions, it exports module and function names.
        For local functions, it exports the source code.
        For unknown functions, it exports basic identification information.

        The split_metadata parameter is excluded from the exported params
        as it's runtime-specific and not needed for reconstruction.
        """
        config = {
            "name": self.name,
            "display_name": self.display_name,
            "abbr": self.abbr,
            "greater_is_better": self.greater_is_better,
            "params": dict(self.params)
        }

        if "split_metadata" in config["params"]:
            del config["params"]["split_metadata"]

        original_func = self._original_func
        try:
            module_name = original_func.__module__
            if module_name and not module_name.startswith("__"):
                if (
                    module_name in ["metrics", "__main__"] or
                    module_name.endswith("metrics")
                ):
                    config["func_type"] = "local"
                    config["func_source"] = inspect.getsource(original_func)
                else:
                    # Try to import as external library function
                    try:
                        module = importlib.import_module(module_name)
                        imported_func = getattr(module, original_func.__name__)
                        if id(imported_func) == id(original_func):
                            config["func_type"] = "imported"
                            config["func_module"] = module_name
                            config["func_name"] = original_func.__name__
                        else:
                            config["func_type"] = "local"
                            config["func_source"] = inspect.getsource(
                                original_func
                            )

                    except (ImportError, AttributeError):
                        # Can't import - treat as local function
                        config["func_type"] = "local"
                        config["func_source"] = inspect.getsource(original_func)
            else:
                config["func_type"] = "local"
                config["func_source"] = inspect.getsource(original_func)

        except (OSError, TypeError):
            if (
                hasattr(original_func, "__module__") and
                hasattr(original_func, "__name__")
            ):
                module_name = original_func.__module__
                if (
                    module_name and not module_name.startswith("__") and
                    not module_name.endswith("metrics")
                ):
                    config["func_type"] = "imported"
                    config["func_module"] = module_name
                    config["func_name"] = original_func.__name__
                else:
                    config["func_type"] = "unknown"
                    config["func_info"] = {
                        "name": getattr(original_func, "__name__", "unknown"),
                        "module": getattr(
                            original_func, "__module__", "unknown"
                        ),
                        "qualname": getattr(
                            original_func, "__qualname__", "unknown"
                        )
                    }
            else:
                config["func_type"] = "unknown"
                config["func_info"] = {
                    "name": getattr(original_func, "__name__", "unknown"),
                    "module": getattr(original_func, "__module__", "unknown"),
                    "qualname": getattr(
                        original_func, "__qualname__", "unknown"
                    )
                }
        return config
