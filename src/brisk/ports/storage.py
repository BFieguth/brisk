"""Port interfaces for the storage and module loading system.

Defines structural contracts for persistent storage operations and runtime
module loading.
"""

import pathlib
from typing import Any, Protocol

import pandas as pd


class StoragePort(Protocol):
    """Loads data, saves evaluation outputs, and manages output directories."""

    results_dir: pathlib.Path
    output_dir: pathlib.Path

    def load_data(
        self,
        data_path: str,
        table_name: str | None = None,
    ) -> pd.DataFrame: ...

    def set_output_dir(self, output_dir: pathlib.Path) -> None: ...

    def save_to_json(
        self,
        data: dict[str, Any],
        output_path: pathlib.Path | str,
        metadata: dict[str, Any],
    ) -> None: ...

    def save_plot(
        self,
        output_path: pathlib.Path,
        metadata: dict[str, Any] | None,
        plot: Any | None,
        **kwargs: Any,
    ) -> None: ...


class ModuleLoaderPort(Protocol):
    """Loads Python objects from user project files at runtime."""

    def load_module_object(
        self,
        project_root: str,
        module_filename: str,
        object_name: str,
        required: bool = True,
    ) -> object | None: ...
