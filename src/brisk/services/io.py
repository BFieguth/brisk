"""IO related utilities."""

from pathlib import Path
from typing import Optional, Any, Dict, Union
import json
import os
import io

import matplotlib.pyplot as plt
import plotnine as pn
import plotly.graph_objects as go
import numpy as np

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
        height: int = 6,
        width: int = 8
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

        height (int, optional): 
            The plot height in inches, by default 6

        width (int, optional): 
            The plot width in inches, by default 8

        Returns
        -------
        None
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)

        self._convert_to_svg(metadata, plot)

        try:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        metadata[key] = json.dumps(value)
            if plot and isinstance(plot, pn.ggplot):
                plot.save(
                    filename=output_path, format="png", metadata=metadata,
                    height=height, width=width, dpi=100
                )
            elif plot and isinstance(plot, go.Figure):
                plot.write_image(
                    file=output_path, format="png", width=width, height=height
                )
            else:
                plt.savefig(output_path, format="png", metadata=metadata)
                plt.close()

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save plot to {output_path}: {e}"
            )

    def _convert_to_svg(
        self,
        metadata: Dict[str, Any],
        plot: Optional[pn.ggplot | go.Figure] = None,
        height: int = 6,
        width: int = 8
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
                plot.save(svg_buffer, format="svg", height=height, width=width)
            elif plot and isinstance(plot, go.Figure):
                plot.write_image(
                    file=svg_buffer, format="svg", width=width, height=height
                )
            else:
                plt.savefig(svg_buffer, format="svg", bbox_inches="tight")

            svg_str = svg_buffer.getvalue().decode("utf-8")
            svg_buffer.close()
            self._other_services["reporting"].store_plot_svg(
                svg_str, metadata
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to convert plot to SVG: {e}"
            )
