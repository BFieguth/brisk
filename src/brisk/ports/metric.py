"""Port interfaces for the metric system.

Defines structural contracts for metrics, metric wrappers, and the metric
registry.
"""

import dataclasses
from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt


@dataclasses.dataclass
class MetricResult:
    """Container for a computed metric result."""

    value: float
    display_name: str
    greater_is_better: bool


@runtime_checkable
class MetricPort(Protocol):
    """Callable that scores predictions against ground truth."""

    def __call__(
        self,
        y_true: npt.NDArray[Any],
        y_pred: npt.NDArray[Any],
        **kwargs: Any,
    ) -> float: ...


@runtime_checkable
class MetricWrapperPort(Protocol):
    """Wraps a metric callable with display metadata and default params."""

    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def greater_is_better(self) -> bool: ...

    @property
    def abbr(self) -> str | None: ...

    def get_func_with_params(self) -> MetricPort: ...

    def set_params(self, **params: Any) -> None: ...


@runtime_checkable
class MetricManagerPort(Protocol):
    """
    Registry that looks up metric wrappers by name, abbreviation, or display
    name.
    """

    def get_metric(self, identifier: str) -> MetricPort: ...

    def get_name(self, identifier: str) -> str: ...

    def is_higher_better(self, identifier: str) -> bool: ...

    def list_metrics(self) -> list[str]: ...

    def set_split_metadata(self, split_metadata: dict[str, Any]) -> None: ...
