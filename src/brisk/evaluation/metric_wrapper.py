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

from sklearn import metrics
from typing import Callable, Any, Optional, Dict

class MetricWrapper:
    """A wrapper for metric functions with default parameters and metadata.

    Wraps metric functions and provides methods to update parameters and
    retrieve the metric function with applied parameters. Also handles display
    names and abbreviations for reporting.

    Parameters
    ----------
    name : str
        Name of the metric
    func : callable
        Metric function to wrap
    display_name : str
        Human-readable name for display
    abbr : str, optional
        Abbreviation for the metric, by default None
    **default_params : Any
        Default parameters for the metric function

    Attributes
    ----------
    name : str
        Name of the metric
    func : callable
        The wrapped metric function
    display_name : str
        Human-readable display name
    abbr : str
        Abbreviation (defaults to name if not provided)
    params : dict
        Current parameters for the metric
    _func_with_params : callable
        Metric function with parameters applied
    scorer : callable
        Scikit-learn scorer created from the metric
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
        self.name = name
        self._original_func = func
        self.func = self._ensure_split_metadata_param(func)
        self.display_name = display_name
        self.abbr = abbr if abbr else name
        self.greater_is_better = greater_is_better
        self.params = default_params
        self.params["split_metadata"] = {}
        self._apply_params()

    def _apply_params(self):
        """Apply current parameters to function and scorer.

        Creates a partial function with the current parameters and updates the
        scikit-learn scorer.
        """
        self._func_with_params = functools.partial(self.func, **self.params)
        self.scorer = metrics.make_scorer(
            self.func,
            greater_is_better=self.greater_is_better,
            **self.params
        )

    def set_params(self, **params: Any):
        """Update parameters for the metric function and scorer.

        Parameters
        ----------
        **params : Any
            New parameters to update or add
        """
        self.params.update(params)
        self._apply_params()

    def get_func_with_params(self) -> Callable:
        """Get the metric function with current parameters applied.

        Returns
        -------
        callable
            Deep copy of the metric function with parameters
        """
        return copy.deepcopy(self._func_with_params)

    def _ensure_split_metadata_param(self, func: Callable) -> Callable:
        """Ensure metric function accepts split_metadata as a keyword argument.

        Wraps the function if necessary to accept the split_metadata parameter
        without affecting the original functionality.

        Parameters
        ----------
        func : callable
            Function to check/wrap

        Returns
        -------
        callable
            Original or wrapped function that accepts split_metadata
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
        """
        Export this MetricWrapper's configuration for rerun functionality.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary that can be used to recreate this MetricWrapper
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
            if module_name and not module_name.startswith('__'):
                if module_name in ['metrics', '__main__'] or module_name.endswith('metrics'):
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
                            # Same name but different function - treat as local/custom
                            config["func_type"] = "local"
                            config["func_source"] = inspect.getsource(original_func)
                            
                    except (ImportError, AttributeError):
                        # Can't import - treat as local function
                        config["func_type"] = "local"
                        config["func_source"] = inspect.getsource(original_func)
            else:
                config["func_type"] = "local"
                config["func_source"] = inspect.getsource(original_func)
                
        except (OSError, TypeError):
            if hasattr(original_func, '__module__') and hasattr(original_func, '__name__'):
                module_name = original_func.__module__
                if module_name and not module_name.startswith('__') and not module_name.endswith('metrics'):
                    config["func_type"] = "imported"
                    config["func_module"] = module_name
                    config["func_name"] = original_func.__name__
                else:
                    config["func_type"] = "unknown"
                    config["func_info"] = {
                        "name": getattr(original_func, '__name__', 'unknown'),
                        "module": getattr(original_func, '__module__', 'unknown'),
                        "qualname": getattr(original_func, '__qualname__', 'unknown')
                    }
            else:
                config["func_type"] = "unknown"
                config["func_info"] = {
                    "name": getattr(original_func, '__name__', 'unknown'),
                    "module": getattr(original_func, '__module__', 'unknown'),
                    "qualname": getattr(original_func, '__qualname__', 'unknown')
                }
        
        return config
