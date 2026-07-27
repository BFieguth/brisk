"""Unit tests for the plotting adapter.

Adapted from the plot-related cases in
``tests/integration/services/test_io.py`` (``save_plot`` and SVG conversion).
The orchestration logic stays in ``IOService``; the rendering mechanics now
live in ``PlotRenderer``, so these tests drive that class directly.
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotnine as pn
import pytest
import seaborn as sns

from brisk.adapters.plotting.plot_renderer import PlotRenderer

# pylint: disable=W0621


@pytest.fixture
def renderer():
    return PlotRenderer()


@pytest.fixture
def ggplot():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    return pn.ggplot(df, pn.aes("x", "y")) + pn.geom_point()


@pytest.mark.unit
class TestPlotRendererSave:
    def test_save_matplotlib(self, renderer, tmp_path):
        output_path = tmp_path / "matplotlib_plot.png"
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        renderer.save(output_path)
        assert output_path.exists()

    def test_save_seaborn(self, renderer, tmp_path):
        output_path = tmp_path / "seaborn_plot.png"
        plt.figure()
        sns.lineplot(x=[1, 2, 3], y=[1, 4, 9])
        renderer.save(output_path)
        assert output_path.exists()

    def test_save_plotnine(self, renderer, ggplot, tmp_path):
        output_path = tmp_path / "plotnine_plot.png"
        renderer.save(output_path, plot=ggplot)
        assert output_path.exists()

    @pytest.mark.slow
    def test_save_plotly(self, renderer, tmp_path):
        output_path = tmp_path / "plotly_plot.png"
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        renderer.save(output_path, plot=fig)
        assert output_path.exists()

    def test_save_normalises_suffix_to_format(self, tmp_path):
        renderer = PlotRenderer(file_format="svg")
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        renderer.save(tmp_path / "plot.png")
        assert (tmp_path / "plot.svg").exists()
        assert not (tmp_path / "plot.png").exists()


@pytest.mark.unit
class TestPlotRendererToSvg:
    def _assert_is_svg(self, svg_str):
        assert isinstance(svg_str, str)
        assert svg_str.startswith("<?xml") or svg_str.lstrip().startswith(
            "<svg"
        )

    def test_to_svg_matplotlib(self, renderer):
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        self._assert_is_svg(renderer.to_svg())

    def test_to_svg_seaborn(self, renderer):
        plt.figure()
        sns.lineplot(x=[1, 2, 3], y=[1, 4, 9])
        self._assert_is_svg(renderer.to_svg())

    def test_to_svg_plotnine(self, renderer, ggplot):
        self._assert_is_svg(renderer.to_svg(plot=ggplot))

    @pytest.mark.slow
    def test_to_svg_plotly(self, renderer):
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        self._assert_is_svg(renderer.to_svg(plot=fig))


@pytest.mark.unit
class TestPlotRendererConfigure:
    def test_configure_updates_settings(self, renderer):
        renderer.configure({
            "file_format": "svg",
            "width": 12,
            "height": 6,
            "dpi": 150,
            "transparent": True,
        })
        assert renderer.format == "svg"
        assert renderer.width == 12
        assert renderer.height == 6
        assert renderer.dpi == 150
        assert renderer.transparent is True
