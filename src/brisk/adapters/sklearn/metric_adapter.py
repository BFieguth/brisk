"""Sklearn adapter for the metric system.

Wraps scikit-learn metric functions behind the MetricWrapperPort interface
and provides an sklearn-specific MetricManager with scorer support.
"""

import copy
import functools
import importlib
import inspect
from typing import Any, Callable, cast

from sklearn import metrics

from brisk.adapters.base import metric_manager
from brisk.ports import metric


class SklearnMetricWrapper:
    """Wraps a scikit-learn or user-defined metric function.

    Provides methods to retrieve the metric with applied parameters, create
    an sklearn scorer, and export the configuration for reproducibility.

    Parameters
    ----------
    name : str
        Canonical identifier for the metric.
    func : Callable
        Metric function ``(y_true, y_pred, **kwargs) -> float``.
    display_name : str
        Human-readable name for reports and plots.
    greater_is_better : bool
        Whether higher values indicate better performance.
    abbr : str or None
        Short abbreviation, defaults to ``name`` if not provided.
    **default_params : Any
        Default keyword arguments forwarded to ``func``.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> wrapper = SklearnMetricWrapper(
    ...     name="mse",
    ...     func=mean_squared_error,
    ...     display_name="Mean Squared Error",
    ...     greater_is_better=False,
    ...     abbr="MSE",
    ... )
    >>> func = wrapper.get_func_with_params()
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., float],
        display_name: str,
        greater_is_better: bool,
        abbr: str | None = None,
        **default_params: Any,
    ) -> None:
        self.name = name
        self._original_func = func
        self.func = self._ensure_split_metadata_param(func)
        self.display_name = display_name
        self.greater_is_better = greater_is_better
        self.abbr = abbr if abbr else name
        self.default_params = default_params

    def get_func_with_params(self) -> metric.MetricPort:
        """Return the metric function with default parameters applied.

        Returns
        -------
        metric.MetricPort
            A deep-copied partial of the metric function.
        """
        return copy.deepcopy(
            functools.partial(self.func, **self.default_params)
        )

    def set_params(self, **params: Any) -> None:
        """Merge additional parameters into the defaults.

        Parameters
        ----------
        **params : Any
            Parameters to merge.
        """
        self.default_params.update(params)

    def get_scorer(self) -> Callable[..., float]:
        """Create an sklearn scorer from this metric.

        Returns
        -------
        Callable
            Scorer compatible with sklearn cross-validation functions.
        """
        return cast(
            Callable[..., float],
            metrics.make_scorer(
                self.func,
                greater_is_better=self.greater_is_better,
                **self.default_params,
            ),
        )

    def export_config(self) -> dict[str, Any]:
        """Export a serializable configuration for rerun functionality.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary containing metric metadata and
            function source information for reconstruction.

        Notes
        -----
        Detects whether the function is an importable library function,
        a locally-defined function, or unknown, and serializes accordingly.
        The ``split_metadata`` parameter is excluded as it is
        runtime-specific.
        """
        config: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "abbr": self.abbr,
            "greater_is_better": self.greater_is_better,
            "default_params": dict(self.default_params),
        }

        if "split_metadata" in config["default_params"]:
            del config["default_params"]["split_metadata"]

        original_func = self._original_func
        try:
            module_name = original_func.__module__
            if module_name and not module_name.startswith("__"):
                if (
                    module_name in ["metrics", "__main__"]
                    or module_name.endswith("metrics")
                ):
                    config["func_type"] = "local"
                    config["func_source"] = inspect.getsource(original_func)
                else:
                    try:
                        module = importlib.import_module(module_name)
                        imported_func = getattr(
                            module, original_func.__name__
                        )
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
                        config["func_type"] = "local"
                        config["func_source"] = inspect.getsource(
                            original_func
                        )
            else:
                config["func_type"] = "local"
                config["func_source"] = inspect.getsource(original_func)

        except (OSError, TypeError):
            config.update(self._export_fallback_func_info(original_func))

        return config

    def _ensure_split_metadata_param(
        self, func: Callable[..., float],
    ) -> Callable[..., float]:
        """Ensure the metric function accepts a ``split_metadata`` kwarg.

        If the function's signature already includes ``split_metadata`` it is
        returned unchanged. Otherwise a thin wrapper is created that accepts
        and discards the parameter.

        Parameters
        ----------
        func : Callable
            The raw metric function.

        Returns
        -------
        Callable
            The original or wrapped function.
        """
        sig = inspect.signature(func)

        if "split_metadata" not in sig.parameters:
            def wrapped_func(  # pylint: disable=unused-argument
                y_true: Any,
                y_pred: Any,
                split_metadata: Any = None,
                **kwargs: Any,
            ) -> float:
                return func(y_true, y_pred, **kwargs)

            wrapped_func.__name__ = func.__name__
            wrapped_func.__qualname__ = func.__qualname__
            wrapped_func.__doc__ = func.__doc__
            return wrapped_func
        return func

    @staticmethod
    def _export_fallback_func_info(
        func: Callable[..., float],
    ) -> dict[str, Any]:
        """Build fallback function info when source inspection fails.

        Parameters
        ----------
        func : Callable
            The function to describe.

        Returns
        -------
        dict[str, Any]
            Either ``func_type: "imported"`` or ``func_type: "unknown"``.
        """
        if (
            hasattr(func, "__module__")
            and hasattr(func, "__name__")
        ):
            module_name = func.__module__
            if (
                module_name
                and not module_name.startswith("__")
                and not module_name.endswith("metrics")
            ):
                return {
                    "func_type": "imported",
                    "func_module": module_name,
                    "func_name": func.__name__,
                }

        return {
            "func_type": "unknown",
            "func_info": {
                "name": getattr(func, "__name__", "unknown"),
                "module": getattr(func, "__module__", "unknown"),
                "qualname": getattr(func, "__qualname__", "unknown"),
            },
        }


class SklearnMetricManager(metric_manager.BaseMetricManager):
    """Sklearn-specific metric manager with scorer and export support.

    Extends ``BaseMetricManager`` with ``get_scorer`` (wraps metrics via
    ``sklearn.metrics.make_scorer``) and ``export_params`` (detects
    built-in metric collections for compact serialization).

    Parameters
    ----------
    *metric_wrappers : SklearnMetricWrapper
        Sklearn metric wrappers to register.

    Examples
    --------
    >>> manager = SklearnMetricManager(mse_wrapper, mae_wrapper)
    >>> scorer = manager.get_scorer("mse")
    """

    def get_scorer(self, identifier: str) -> Callable[..., float]:
        """Retrieve an sklearn scorer by identifier.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric.

        Returns
        -------
        Callable
            Scorer compatible with sklearn cross-validation functions.
        """
        name = self._resolve_identifier(identifier)
        wrapper = self._wrappers[name]
        if not isinstance(wrapper, SklearnMetricWrapper):
            raise TypeError(
                f"Wrapper for '{name}' is not an SklearnMetricWrapper"
            )
        return wrapper.get_scorer()

    def export_params(self) -> list[dict[str, Any]]:
        """Export metric configurations for rerun functionality.

        Detects whether registered metrics match built-in collections
        and exports compact references where possible. Custom metrics
        are exported with full configuration.

        Returns
        -------
        list[dict[str, Any]]
            Serializable list of metric configurations.
        """
        # Imported lazily to avoid a circular import: the default metric
        # collections are built from SklearnMetricWrapper (this module).
        from brisk.defaults import ( # pylint: disable=import-outside-toplevel
            classification_metrics, regression_metrics,
        )

        regression_names = {
            wrapper.name for wrapper in regression_metrics.REGRESSION_METRICS
        }
        classification_names = {
            wrapper.name
            for wrapper in classification_metrics.CLASSIFICATION_METRICS
        }

        found_regression_metrics = set()
        found_classification_metrics = set()
        custom_metrics = []

        for name, wrapper in self._wrappers.items():
            if name in regression_names:
                found_regression_metrics.add(name)
            elif name in classification_names:
                found_classification_metrics.add(name)
            elif isinstance(wrapper, SklearnMetricWrapper):
                custom_metrics.append(wrapper.export_config())

        result = []
        if found_regression_metrics == regression_names:
            result.append({
                "type": "builtin_collection",
                "collection": "brisk.REGRESSION_METRICS"
            })
        elif found_regression_metrics:
            for name in found_regression_metrics:
                result.append({
                    "type": "builtin_metric",
                    "collection": "brisk.REGRESSION_METRICS",
                    "name": name
                })

        if found_classification_metrics == classification_names:
            result.append({
                "type": "builtin_collection",
                "collection": "brisk.CLASSIFICATION_METRICS"
            })
        elif found_classification_metrics:
            for name in found_classification_metrics:
                result.append({
                    "type": "builtin_metric",
                    "collection": "brisk.CLASSIFICATION_METRICS",
                    "name": name
                })

        for custom_config in custom_metrics:
            custom_config["type"] = "custom_metric"
            result.append(custom_config)

        return result
