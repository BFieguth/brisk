"""HTML reporter adapter implementing ``ReporterPort``.

Adapts the existing ``ReportingService`` (which collects evaluation results
and assembles a ``ReportData`` object) to the ``ReporterPort`` interface and
pairs it with the Jinja2 ``html_renderer.HTMLReportRenderer`` to emit an HTML report.

The collection logic is reused as-is from ``ReportingService`` ("wrap, never
rewrite"); this adapter only narrows the type surface to the port and adds a
``render_report`` convenience that drives the renderer.
"""

from __future__ import annotations

import pathlib

from brisk.adapters.reporting import html_renderer
from brisk.services import reporting


class HTMLReporter(reporting.ReportingService):
    """Reporter adapter that produces an interactive HTML report.

    Inherits the full result-collection behaviour of ``ReportingService`` and
    therefore satisfies ``ReporterPort``. Adds rendering via
    ``html_renderer.HTMLReportRenderer``.

    Parameters
    ----------
    name : str, default="reporting"
        Service name identifier.
    renderer : html_renderer.HTMLReportRenderer, optional
        Renderer used to emit HTML. A default instance is created when not
        provided.

    Examples
    --------
    >>> reporter = HTMLReporter("reporting")
    >>> reporter.set_metric_config(metric_manager)
    >>> reporter.set_evaluator_registry(registry)
    >>> # ... collect results ...
    >>> reporter.render_report(pathlib.Path("output"))
    """

    def __init__(
        self,
        name: str = "reporting",
        renderer: html_renderer.HTMLReportRenderer | None = None,
    ) -> None:
        super().__init__(name)
        self._renderer = renderer if renderer is not None else None

    @property
    def renderer(self) -> html_renderer.HTMLReportRenderer:
        """Lazily-constructed HTML renderer.

        Returns
        -------
        html_renderer.HTMLReportRenderer
            The renderer used to emit the report.
        """
        if self._renderer is None:
            self._renderer = html_renderer.HTMLReportRenderer()
        return self._renderer

    def render_report(self, output_path: pathlib.Path) -> None:
        """Assemble the report data and write ``report.html``.

        Parameters
        ----------
        output_path : pathlib.Path
            Directory where the HTML report will be written.
        """
        report_data = self.get_report_data()
        self.renderer.render(report_data, output_path)
