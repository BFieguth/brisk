"""Reusable base metric manager for any ML framework.

Provides multi-identifier lookup (name, abbreviation, display name) and
the standard MetricManagerPort operations. Framework-specific adapters
subclass this to add scorer creation, export logic, etc.
"""

from typing import Any

from brisk.ports import metric


class BaseMetricManager:
    """Registry that resolves metrics by name, abbreviation, or display name.

    Stores metric wrappers keyed by canonical name and maintains a unified
    alias dict that maps every known identifier to that canonical name.

    Parameters
    ----------
    *metric_wrappers : metric.MetricWrapperPort
        Metric wrappers to register on construction.

    Examples
    --------
    >>> manager = BaseMetricManager(mse_wrapper, mae_wrapper)
    >>> func = manager.get_metric("mse")
    """

    def __init__(self, *metric_wrappers: metric.MetricWrapperPort) -> None:
        self._wrappers: dict[str, metric.MetricWrapperPort] = {}
        self._aliases: dict[str, str] = {}
        for wrapper in metric_wrappers:
            self._add_metric(wrapper)

    def _add_metric(self, wrapper: metric.MetricWrapperPort) -> None:
        """Register a metric wrapper, replacing any existing entry.

        Parameters
        ----------
        wrapper : metric.MetricWrapperPort
            Metric wrapper to register.
        """
        # Clean up stale aliases from a previous wrapper with the same name
        if wrapper.name in self._wrappers:
            old = self._wrappers[wrapper.name]
            self._aliases.pop(old.name, None)
            if old.abbr:
                self._aliases.pop(old.abbr, None)
            if old.display_name:
                self._aliases.pop(old.display_name, None)

        self._wrappers[wrapper.name] = wrapper
        self._aliases[wrapper.name] = wrapper.name
        if wrapper.abbr:
            self._aliases[wrapper.abbr] = wrapper.name
        if wrapper.display_name:
            self._aliases[wrapper.display_name] = wrapper.name

    def _resolve_identifier(self, identifier: str) -> str:
        """Resolve a metric identifier to its canonical name.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric.

        Returns
        -------
        str
            Canonical metric name.

        Raises
        ------
        ValueError
            If the identifier is not found in any mapping.
        """
        if identifier in self._aliases:
            return self._aliases[identifier]
        raise ValueError(f"Metric '{identifier}' not found.")

    def get_metric(self, identifier: str) -> metric.MetricPort:
        """Retrieve a metric function by identifier.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric.

        Returns
        -------
        metric.MetricPort
            The metric function with default parameters applied.
        """
        name = self._resolve_identifier(identifier)
        return self._wrappers[name].get_func_with_params()

    def get_name(self, identifier: str) -> str:
        """Retrieve a metric's display name.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric.

        Returns
        -------
        str
            The human-readable display name.
        """
        name = self._resolve_identifier(identifier)
        return self._wrappers[name].display_name

    def is_higher_better(self, identifier: str) -> bool:
        """Check whether higher values indicate better performance.

        Parameters
        ----------
        identifier : str
            Full name, abbreviation, or display name of the metric.

        Returns
        -------
        bool
            True if higher values are better for this metric.
        """
        name = self._resolve_identifier(identifier)
        return self._wrappers[name].greater_is_better

    def list_metrics(self) -> list[str]:
        """Return all registered metric names.

        Returns
        -------
        list[str]
            Canonical names of all registered metrics.
        """
        return list(self._wrappers.keys())

    def set_split_metadata(self, split_metadata: dict[str, Any]) -> None:
        """Propagate split metadata to all registered metrics.

        Parameters
        ----------
        split_metadata : dict[str, Any]
            Metadata about the current data split.
        """
        for wrapper in self._wrappers.values():
            wrapper.set_params(split_metadata=split_metadata)
