"""Filesystem adapter implementing ``StoragePort``."""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from brisk.adapters.plotting.plot_renderer import PlotRenderer


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            val = float(o)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (float, int)) and (
            np.isnan(o) if isinstance(o, float) else False
        ):
            return None
        try:
            return super().default(o)
        except TypeError:
            return f"<{type(o).__module__}.{type(o).__name__}>"


class FilesystemStorageAdapter:
    """Filesystem-backed storage for JSON, plots, and tabular data."""

    results_dir: pathlib.Path
    output_dir: pathlib.Path

    def __init__(
        self,
        results_dir: pathlib.Path | str,
        output_dir: pathlib.Path | str,
        plot_renderer: PlotRenderer | None = None,
    ) -> None:
        self.results_dir = pathlib.Path(results_dir)
        self.output_dir = pathlib.Path(output_dir)
        self._plot_renderer = plot_renderer or PlotRenderer()

    def set_output_dir(self, output_dir: pathlib.Path | str) -> None:
        self.output_dir = pathlib.Path(output_dir)

    def save_to_json(
        self,
        data: dict[str, Any],
        output_path: pathlib.Path | str,
        metadata: dict[str, Any],
    ) -> None:
        path = pathlib.Path(output_path)
        if not path.parent.exists():
            os.makedirs(path.parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, cls=NumpyEncoder)

    def save_plot(
        self,
        output_path: pathlib.Path | str,
        metadata: dict[str, Any] | None,
        plot: Any | None,
        **kwargs: Any,
    ) -> None:
        path = pathlib.Path(output_path)
        if not path.parent.exists():
            os.makedirs(path.parent, exist_ok=True)

        self._plot_renderer.save(
            output_path=path,
            plot=plot,
            height=kwargs.get("height"),
            width=kwargs.get("width"),
        )

    def load_data(
        self,
        data_path: str,
        table_name: str | None = None,
    ) -> pd.DataFrame:
        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension == ".csv":
            return pd.read_csv(data_path)

        if file_extension in (".xls", ".xlsx"):
            return pd.read_excel(data_path)

        if file_extension in (".db", ".sqlite"):
            if table_name is None:
                raise ValueError(
                    "For SQL databases, 'table_name' must be provided."
                )
            conn = sqlite3.connect(data_path)
            try:
                return pd.read_sql(f"SELECT * FROM {table_name}", conn)
            finally:
                conn.close()

        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            "Supported formats are CSV, Excel, and SQL database."
        )
