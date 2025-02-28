"""metric_wrapper.py

This module provides the MetricWrapper class, which is designed to wrap 
metric functions from the scikit-learn library. The MetricWrapper allows 
for the easy application of default parameters to metrics, making it 
simpler to manage and use various metrics in the Brisk framework.
"""
import copy
import functools
import inspect

from sklearn import metrics
from typing import Callable, Any, Optional

class MetricWrapper:
    """A wrapper for metric functions to facilitate the application of 
    default parameters and provide additional metadata.

    This class allows users to define a metric function along with its 
    default parameters, a human-readable display name, and an optional 
    abbreviation. It also provides methods to update parameters and 
    retrieve the metric function with the applied parameters.

    Attributes:
        name (str): The name of the metric.
        func (Callable): The metric function.
        display_name (str): A human-readable name for the metric.
        abbr (str): An abbreviation for the metric.
        params (dict): Default parameters for the metric function.
        _func_with_params (Callable): The metric function with applied 
            parameters.
        scorer (Callable): A scikit-learn scorer created from the metric 
            function and parameters.

    Args:
        name (str): The name of the metric.
        func (Callable): The metric function.
        display_name (str): A human-readable name for the metric.
        abbr (Optional[str]): An abbreviation for the metric.
        **default_params: Default parameters for the metric function.
    """
    def __init__(
        self,
        name: str,
        func: Callable,
        display_name: str,
        abbr: Optional[str] = None,
        **default_params: Any
    ):
        """Initializes the MetricWrapper with a metric function and default 
        parameters.

        Args:
            name (str): The name of the metric.
            func (Callable): The metric function.
            display_name (Optional[str]): A human-readable name for the metric.
            abbr (Optional[str]): An abbreviation for the metric.
            **default_params: Default parameters for the metric function.
        """
        self.name = name
        self.func = self._ensure_split_metadata_param(func)
        self.display_name = display_name
        self.abbr = abbr if abbr else name
        self.params = default_params
        self.params["split_metadata"] = {}
        self._apply_params()

    def _apply_params(self):
        """Applies the parameters to both the function and scorer."""
        self._func_with_params = functools.partial(self.func, **self.params)
        self.scorer = metrics.make_scorer(self.func, **self.params)

    def set_params(self, **params: Any):
        """Updates the parameters for the metric function and scorer.

        Args:
            **params: Parameters to update.
        """
        self.params.update(params)
        self._apply_params()

    def get_func_with_params(self):
        """Returns the metric function with applied parameters."""
        return copy.deepcopy(self._func_with_params)

    def _ensure_split_metadata_param(self, func: Callable) -> Callable:
        """Wraps the provided function to ensure it accepts split_metadata as 
        a kwarg."""
        sig = inspect.signature(func)

        if "split_metadata" not in sig.parameters:
            def wrapped_func(y_true, y_pred, split_metadata=None, **kwargs): # pylint: disable=unused-argument
                return func(y_true, y_pred, **kwargs)

            wrapped_func.__name__ = func.__name__
            wrapped_func.__qualname__ = func.__qualname__
            wrapped_func.__doc__ = func.__doc__
            return wrapped_func
        return func
