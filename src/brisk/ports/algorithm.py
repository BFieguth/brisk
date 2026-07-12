"""Port interfaces for the algorithm system.

Defines structural contracts for machine learning models and algorithm wrappers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt


@runtime_checkable
class ModelPort(Protocol):
    """Core model interface: fit, predict, get/set params."""

    wrapper_name: str

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ... # pylint: disable = C0103

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> ModelPort: ... # pylint: disable = C0103

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> ModelPort: ...


@runtime_checkable
class ProbabilisticModelPort(ModelPort, Protocol):
    """Model that supports class probability estimates."""

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ... # pylint: disable = C0103


@runtime_checkable
class DecisionModelPort(ModelPort, Protocol):
    """Model that supports raw decision scores."""

    def decision_function(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ... # pylint: disable = C0103


@runtime_checkable
class AlgorithmWrapperPort(Protocol):
    """Factory that wraps an algorithm class with metadata and parameters."""

    name: str
    display_name: str
    default_params: dict[str, Any]
    hyperparam_grid: dict[str, Any]

    def instantiate(self) -> ModelPort: ...

    def instantiate_tuned(self, best_params: dict[str, Any]) -> ModelPort: ...

    def get_hyperparam_grid(self) -> dict[str, Any]: ...

    def export_config(self) -> dict[str, Any]: ...

    def to_markdown(self) -> str: ...

    def __setitem__(self, key: str, value: dict[str, Any]) -> None: ...
