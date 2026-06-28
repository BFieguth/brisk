"""Unit tests for the Jinja2 HTML renderer adapter.

The original ``ReportRenderer`` had no dedicated test module; these tests
cover the adapter directly. They confirm assets load from the
``brisk.reporting`` package and that a real ``ReportData`` instance renders to
a self-contained ``report.html`` file.
"""

from unittest import mock

import pytest

from brisk.adapters.reporting.html_renderer import HTMLReportRenderer
from brisk.services import reporting

# pylint: disable=W0212


@pytest.fixture()
def report_data():
    """A real (empty) ``ReportData`` assembled by ReportingService."""
    service = reporting.ReportingService("test_reporting")
    service._other_services = {
        "utility": mock.MagicMock(),
        "logging": mock.MagicMock(),
    }
    service.set_metric_config(mock.MagicMock())
    service.set_evaluator_registry(mock.MagicMock())
    return service.get_report_data()


@pytest.fixture(scope="module")
def renderer():
    return HTMLReportRenderer()


@pytest.mark.unit
class TestHTMLReportRendererAssets:
    def test_css_loaded(self, renderer):
        assert renderer.css_content
        assert all(k.endswith("_css") for k in renderer.css_content)

    def test_page_templates_loaded(self, renderer):
        assert renderer.page_templates
        assert all(k.endswith("_template") for k in renderer.page_templates)

    def test_component_templates_loaded(self, renderer):
        assert renderer.component_templates
        assert all(
            k.endswith("_component") for k in renderer.component_templates
        )

    def test_javascript_loaded(self, renderer):
        assert isinstance(renderer.javascript, str)
        assert renderer.javascript.strip()

    def test_template_available(self, renderer):
        assert renderer.template is not None


@pytest.mark.unit
class TestHTMLReportRendererRender:
    def test_render_writes_report_html(self, renderer, report_data, tmp_path):
        renderer.render(report_data, tmp_path)
        output = tmp_path / "report.html"
        assert output.exists()
        assert output.stat().st_size > 0

    def test_render_output_is_html(self, renderer, report_data, tmp_path):
        renderer.render(report_data, tmp_path)
        text = (tmp_path / "report.html").read_text().lower()
        assert "<html" in text or "<!doctype" in text

    def test_render_embeds_report_json(self, renderer, report_data, tmp_path):
        renderer.render(report_data, tmp_path)
        text = (tmp_path / "report.html").read_text()
        # The serialized ReportData JSON is embedded for the client app.
        assert "navbar" in text
