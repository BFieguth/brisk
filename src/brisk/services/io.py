"""IO related utilities."""

from pathlib import Path
from typing import Optional, Any, Dict, Union
import json
import os
import io
import sys
import importlib

import matplotlib.pyplot as plt
import plotnine as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sqlite3

from brisk.services import base

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)


class IOService(base.BaseService):
    """IO service for saving and loading files.
    
    Parameters
    ----------
    name : str
        The name of the service

    results_dir : Path
        The root directory for all results, does not change at runtime.
        
    output_dir : Path
        The current output directory, will be changed at runtime.

    Attributes
    ----------
    results_dir : Path
        The root directory for all results, does not change at runtime.
    output_dir : Path
        The current output directory, will be changed at runtime.
    """
    def __init__(self, name: str, results_dir: Path, output_dir: Path):
        super().__init__(name)
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.format = "png"
        self.width = 10
        self.height = 8
        self.dpi = 300
        self.transparent = False

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the current output directory.

        Parameters
        ----------
        output_dir : Path
            The new output directory

        Returns
        -------
        None
        """
        self.output_dir = output_dir

    def save_to_json(
        self,
        data: Dict[str, Any],
        output_path: Union[Path, str],
        metadata: Dict[str, Any]
    ) -> None:
        """Save dictionary to JSON file with metadata.

        Parameters
        ----------
        data : dict
            Data to save

        output_path : str
            The path to the output file.

        metadata : dict
            Metadata to include, by default None

        Returns
        -------
        None
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        try:
            if metadata:
                data["_metadata"] = metadata

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, cls=NumpyEncoder)

            self._other_services["reporting"].store_table_data(
                data, metadata
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save JSON to {output_path}: {e}"
            )

    def save_plot(
        self,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        plot: Optional[pn.ggplot | go.Figure] = None,
        **kwargs
    ) -> None:
        """Save current plot to file with metadata.

        Parameters
        ----------
        output_path (Path): 
            The path to the output file.

        metadata (dict, optional): 
            Metadata to include, by default None

        plot (ggplot, optional): 
            Plotnine plot object, by default None

        Returns
        -------
        None
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)

        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        output_path = output_path.with_suffix(f".{self.format}")
        self._convert_to_svg(metadata, plot, height, width)

        try:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        metadata[key] = json.dumps(value)
            if plot and isinstance(plot, pn.ggplot):
                plot.save(
                    filename=output_path, format=self.format,
                    height=height, width=width, dpi=self.dpi,
                    transparent=self.transparent
                )
            elif plot and isinstance(plot, go.Figure):
                plot.write_image(
                    file=output_path, format=self.format
                )
            else:
                plt.savefig(
                    output_path, format=self.format,
                    dpi=self.dpi, transparent=self.transparent
                )
                plt.close()

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save plot to {output_path}: {e}"
            )

    def save_rerun_config(self, data: Dict, output_path: Union[Path, str]):
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save JSON to {output_path}: {e}"
            )

    def _convert_to_svg(
        self,
        metadata: Dict[str, Any],
        plot: Optional[pn.ggplot | go.Figure],
        height,
        width
    ) -> None:
        """Convert plot to SVG format for the report.

        Parameters
        ----------
        metadata : dict
            Metadata to include
        plot : ggplot
            Plotnine plot object
        height : int
            The plot height in inches
        width : int
            The plot width in inches

        Returns
        -------
        None
        """
        try:
            svg_buffer = io.BytesIO()
            if plot and isinstance(plot, pn.ggplot):
                plot.save(
                    svg_buffer, format="svg", height=height, width=width,
                    dpi=100
                )
            elif plot and isinstance(plot, go.Figure):
                plot.write_image(
                    file=svg_buffer, format="svg", width=width, height=height
                )
            else:
                plt.savefig(
                    svg_buffer, format="svg", bbox_inches="tight", dpi=100
                )

            svg_str = svg_buffer.getvalue().decode("utf-8")
            svg_buffer.close()
            self._other_services["reporting"].store_plot_svg(
                svg_str, metadata
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to convert plot to SVG: {e}"
            )

    def set_io_settings(self, io_settings: Dict[str, Any]) -> None:
        """Set settings to use when saving plots."""
        self.format = io_settings["format"]
        self.width = io_settings["width"]
        self.height = io_settings["height"]
        self.dpi = io_settings["dpi"]
        self.transparent = io_settings["transparent"]

    @staticmethod
    def load_data(
        data_path: str,
        table_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Loads data from a CSV, Excel file, or SQL database.

        Parameters
        ----------
        data_path : str
            Path to the dataset file
        table_name : str, optional
            Name of the table in SQL database. Required for SQL databases.

        Returns
        -------
        pd.DataFrame: The loaded dataset.

        Raises
        ------
        ValueError
            If file format is unsupported or table_name is missing for SQL
            database.
        """
        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension == ".csv":
            return pd.read_csv(data_path)

        elif file_extension in [".xls", ".xlsx"]:
            return pd.read_excel(data_path)

        elif file_extension in [".db", ".sqlite"]:
            if table_name is None:
                raise ValueError(
                    "For SQL databases, 'table_name' must be provided."
                )

            conn = sqlite3.connect(data_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            return df

        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                "Supported formats are CSV, Excel, and SQL database."
            )

    @staticmethod
    def load_module_object(
        project_root: str,
        module_filename: str,
        object_name: str,
        required: bool = True
    ) -> Union[object, None]:
        """
        Dynamically loads an object from a specified module file.

        Parameters
        ----------
        project_root : str
            Path to project root directory
        module_filename : str
            Name of the module file
        object_name : str
            Name of object to load
        required : bool, default=True
            Whether to raise error if object not found

        Returns
        -------
        object or None
            Loaded object or None if not found and not required

        Raises
        ------
        FileNotFoundError
            If module file not found
        AttributeError
            If required object not found in module
        """
        module_path = os.path.join(project_root, module_filename)

        if not os.path.exists(module_path):
            raise FileNotFoundError(
                f'{module_filename} not found in {project_root}'
            )

        module_name = os.path.splitext(module_filename)[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        spec.loader.exec_module(module)

        if hasattr(module, object_name):
            return getattr(module, object_name)
        elif required:
            raise AttributeError(
                f'The object \'{object_name}\' is not defined in {module_filename}'
            )
        else:
            return None
