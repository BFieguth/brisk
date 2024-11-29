import copy
from functools import partial

from sklearn.metrics import make_scorer
from typing import Callable, Any, Optional

class MetricWrapper:
    def __init__(
        self,
        name: str,
        func: Callable,
        display_name: str,
        abbr: Optional[str] = None,
        **default_params: Any
    ):
        """Initializes the MetricWrapper with a metric function and default parameters.

        Args:
            name (str): The name of the metric.
            func (Callable): The metric function.
            display_name (Optional[str]): A human-readable name for the metric.
            abbr (Optional[str]): An abbreviation for the metric.
            **default_params: Default parameters for the metric function.
        """
        self.name = name
        self.func = func
        self.display_name = display_name
        self.abbr = abbr if abbr else name
        self.params = default_params
        self._apply_params()

    def _apply_params(self):
        """Applies the parameters to both the function and scorer."""
        self._func_with_params = partial(self.func, **self.params)
        self.scorer = make_scorer(self.func, **self.params)

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
