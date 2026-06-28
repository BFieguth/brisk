"""Plotting adapter for rendering and serializing plots.

Encapsulates the external plotting-library mechanics (matplotlib, plotnine,
plotly) used to persist plots to disk and to convert them to SVG strings for
embedding in HTML reports. This is the "move, don't rewrite" extraction of the
plot-rendering portion of ``IOService``; the surrounding orchestration (output
directories, report-cache wiring) remains in the domain/service layer and is
expected to delegate here during the Phase 4 evaluator decoupling.

Supported plot objects:

* ``plotnine.ggplot``
* ``plotly.graph_objects.Figure``
* the current ``matplotlib.pyplot`` figure (when ``plot`` is ``None``)
"""

from __future__ import annotations

import io
import pathlib
import warnings
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotnine as pn


class PlotRenderer:
    """Render plot objects to files and SVG strings.

    Parameters
    ----------
    file_format : str, default="png"
        Default file format for saved plots.
    width : int, default=10
        Default plot width in inches.
    height : int, default=8
        Default plot height in inches.
    dpi : int, default=300
        Default plot DPI for file output.
    transparent : bool, default=False
        Whether to save plots with a transparent background.

    Examples
    --------
    >>> renderer = PlotRenderer()
    >>> renderer.save(my_ggplot, pathlib.Path("plot.png"))
    >>> svg = renderer.to_svg(my_ggplot)
    """

    def __init__(
        self,
        file_format: str = "png",
        width: int = 10,
        height: int = 8,
        dpi: int = 300,
        transparent: bool = False,
    ) -> None:
        self.format = file_format
        self.width = width
        self.height = height
        self.dpi = dpi
        self.transparent = transparent

    def configure(self, settings: dict[str, Any]) -> None:
        """Update render settings from a settings dictionary.

        Parameters
        ----------
        settings : dict[str, Any]
            Mapping with keys ``file_format``, ``width``, ``height``,
            ``dpi`` and ``transparent``.
        """
        self.format = settings["file_format"]
        self.width = settings["width"]
        self.height = settings["height"]
        self.dpi = settings["dpi"]
        self.transparent = settings["transparent"]

    def save(
        self,
        output_path: Any,
        plot: pn.ggplot | go.Figure | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Persist a plot to a file.

        Parameters
        ----------
        output_path : pathlib.Path or str
            Destination path. The suffix is normalised to ``self.format``.
        plot : plotnine.ggplot or plotly.graph_objects.Figure, optional
            Plot to save. If None, the current matplotlib figure is saved.
        height : int, optional
            Override for plot height in inches.
        width : int, optional
            Override for plot width in inches.
        """
        height = height if height is not None else self.height
        width = width if width is not None else self.width
        output_path = pathlib.Path(output_path).with_suffix(f".{self.format}")

        if plot is not None and isinstance(plot, pn.ggplot):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="plotnine"
                )
                plot.save(
                    filename=output_path, format=self.format,
                    height=height, width=width, dpi=self.dpi,
                    transparent=self.transparent,
                )
        elif plot is not None and isinstance(plot, go.Figure):
            plot.write_image(file=output_path, format=self.format)
        else:
            plt.savefig(
                output_path, format=self.format,
                dpi=self.dpi, transparent=self.transparent,
            )
            plt.close()

    def to_svg(
        self,
        plot: pn.ggplot | go.Figure | None = None,
        height: int | None = None,
        width: int | None = None,
        dpi: int = 100,
    ) -> str:
        """Render a plot to an SVG string.

        Parameters
        ----------
        plot : plotnine.ggplot or plotly.graph_objects.Figure, optional
            Plot to convert. If None, the current matplotlib figure is used.
        height : int, optional
            Override for plot height in inches.
        width : int, optional
            Override for plot width in inches.
        dpi : int, default=100
            DPI used for the SVG export.

        Returns
        -------
        str
            The SVG markup as a UTF-8 string.
        """
        height = height if height is not None else self.height
        width = width if width is not None else self.width

        svg_buffer = io.BytesIO()
        if plot is not None and isinstance(plot, pn.ggplot):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="plotnine"
                )
                plot.save(
                    svg_buffer, format="svg", height=height, width=width,
                    dpi=dpi,
                )
        elif plot is not None and isinstance(plot, go.Figure):
            plot.write_image(
                file=svg_buffer, format="svg", width=width, height=height
            )
        else:
            plt.savefig(
                svg_buffer, format="svg", bbox_inches="tight", dpi=dpi
            )

        svg_str = svg_buffer.getvalue().decode("utf-8")
        svg_buffer.close()
        return svg_str
