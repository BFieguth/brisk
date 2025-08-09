"""IO related utilities for evaluators"""

from pathlib import Path
from typing import Optional, Any, Dict, Union
import json
import os
import io

import matplotlib.pyplot as plt
import plotnine as pn

from brisk.services.base import BaseService

class IOService(BaseService):
    """IO service for saving and loading files."""
    def __init__(self, name: str, results_dir: Path, output_dir: Path):
        """
        Parameters
        ----------
        name : str
            The name of the service

        results_dir : Path
            The root directory for all results, does not change at runtime.
            
        output_dir : Path
            The current output directory, will be changed at runtime.
        """

        super().__init__(name)
        self.results_dir = results_dir
        self.output_dir = output_dir

    def set_output_dir(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def save_to_json(
        self,
        data: Dict[str, Any],
        output_path: Union[Path, str],
        metadata: Dict[str, Any]
    ):
        """Save dictionary to JSON file with metadata.

        Parameters
        ----------
        data : dict
            Data to save

        output_path : str
            The path to the output file.

        metadata : dict
            Metadata to include, by default None
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        try:
            if metadata:
                data["_metadata"] = metadata

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            self._other_services["reporting"].store_table_data(
                data, metadata
            )

        except IOError as e:
            self._other_services["logging"].logger.info(f"Failed to save JSON to {output_path}: {e}")

    def save_plot(
        self,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        plot: Optional[pn.ggplot] = None,
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
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        
        self._convert_to_svg(metadata, plot)

        try:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        metadata[key] = json.dumps(value)
            if plot:
                plot.save(
                    filename=output_path, format="png", metadata=metadata,
                    height=height, width=width, dpi=100
                )
            else:
                plt.savefig(output_path, format="png", metadata=metadata)
                plt.close()
                # plt.close('all')

        except IOError as e:
            self._other_services["logging"].logger.info(f"Failed to save plot to {output_path}: {e}")
            # plt.close('all')

    def _convert_to_svg(
        self,
        metadata: Dict[str, Any],
        plot: Optional[pn.ggplot] = None,
        height: int = 6,
        width: int = 8
    ):
        try:
            svg_buffer = io.BytesIO()
            if plot: 
                plot.save(svg_buffer, format="svg", height=height, width=width)
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
