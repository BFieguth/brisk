"""Port interfaces for the data preprocessing system.

Defines structural contracts for data preprocessors (scaling, encoding, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PreprocessorPort(Protocol):
    """Interface that all preprocessors must implement."""

    @property
    def is_fitted(self) -> bool: ...

    def get_feature_names(self, feature_names: list[str]) -> list[str]: ...

    def export_params(self) -> dict[str, Any]: ...

    def fit(
        self,
        X: pd.DataFrame, # pylint: disable = C0103
        y: pd.Series | None,
        categorical_features: list[str] | None = None,
    ) -> PreprocessorPort: ...

    def transform(
        self,
        X: pd.DataFrame, # pylint: disable = C0103
        y: pd.Series | None,
    ) -> tuple[pd.DataFrame, pd.Series | None]: ...

    def fit_transform(
        self,
        X: pd.DataFrame, # pylint: disable = C0103
        y: pd.Series | None,
        categorical_features: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]: ...
