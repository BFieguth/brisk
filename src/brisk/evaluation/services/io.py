"""IO related utilities for evaluators"""

from pathlib import Path
from typing import Optional, Any, Dict, Union
import json

import matplotlib.pyplot as plt
import plotnine as pn

from brisk.evaluation.services.base import BaseService

class IOService(BaseService):
    """IO service for saving and loading files."""
    def __init__(self, name: str, output_dir: Path):
        super().__init__(name)
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
        try:
            if metadata:
                data["_metadata"] = metadata

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

        except IOError as e:
            self._other_services["logging"].logger.info(f"Failed to save JSON to {output_path}: {e}")

    def save_plot(
        self,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        plot: Optional[pn.ggplot] = None,
        height: int = 6,
        width: int = 8
    ) -> None:
        """Save current plot to file with metadata.

        Parameters
        ----------
        output_path (str): 
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
                # plt.close('all')

        except IOError as e:
            self._other_services["logging"].logger.info(f"Failed to save plot to {output_path}: {e}")
            # plt.close('all')
