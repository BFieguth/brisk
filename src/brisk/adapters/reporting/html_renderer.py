"""Jinja2 HTML renderer adapter.

Renders a ``ReportData`` instance into a single self-contained interactive
HTML report. All direct use of Jinja2 and the on-disk report assets (CSS,
HTML templates, JavaScript) lives here so the domain core stays free of
templating concerns.

The renderer loads its assets from the ``brisk.reporting`` package directory,
which holds the ``styles/``, ``pages/``, ``components/`` and ``js/`` folders
plus the top-level ``report.html`` template.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any

import jinja2

import brisk.reporting


class HTMLReportRenderer:
    """Render a ``ReportData`` instance to an HTML report.

    Loads CSS, HTML page/component templates, and JavaScript from the
    ``brisk.reporting`` package directory and uses Jinja2 to produce a single
    ``report.html`` file.

    Attributes
    ----------
    css_content : dict[str, str]
        Mapping of CSS variable names to CSS content.
    page_templates : dict[str, str]
        Mapping of template variable names to HTML page templates.
    component_templates : dict[str, str]
        Mapping of component variable names to HTML component templates.
    javascript : str
        Concatenated JavaScript code with comments stripped.
    env : jinja2.Environment
        Jinja2 environment for template rendering.
    template : jinja2.Template
        Main Jinja2 template for the report.
    """

    def __init__(self, report_dir: pathlib.Path | str | None = None) -> None:
        """Initialize the renderer and load all report assets.

        Parameters
        ----------
        report_dir : pathlib.Path or str, optional
            Directory containing the report assets. Defaults to the
            ``brisk.reporting`` package directory.
        """
        if report_dir is None:
            report_dir = os.path.dirname(
                os.path.abspath(brisk.reporting.__file__)
            )
        report_dir = str(report_dir)

        self.css_content = self._load_directory(
            pathlib.Path(report_dir, "styles"), ".css", "_css"
        )
        self.page_templates = self._load_directory(
            pathlib.Path(report_dir, "pages"), ".html", "_template"
        )
        self.component_templates = self._load_directory(
            pathlib.Path(report_dir, "components"), ".html", "_component"
        )
        self.javascript = self._load_javascript(
            pathlib.Path(report_dir, "js/renderers"),
            pathlib.Path(report_dir, "js/core/app.js"),
        )
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=report_dir)
        )
        self.template = self.env.get_template("report.html")

    def _load_directory(
        self,
        dir_path: pathlib.Path,
        file_extension: str,
        name_extension: str,
    ) -> dict[str, str]:
        """Load all files in a directory, keyed by derived variable name.

        Parameters
        ----------
        dir_path : pathlib.Path
            Directory to load files from.
        file_extension : str
            File extension to filter by (e.g. ``".css"``).
        name_extension : str
            String to replace the file extension with in variable names.

        Returns
        -------
        dict[str, str]
            Mapping of variable names to file contents.
        """
        content = {}
        files = [
            file for file in os.listdir(dir_path)
            if file.endswith(file_extension)
        ]
        for file in files:
            file_path = pathlib.Path(dir_path, file)
            variable_name = file.replace(file_extension, name_extension)
            with open(file_path, "r", encoding="utf-8") as f:
                content[variable_name] = f.read()
        return content

    def _load_javascript(
        self,
        renderer_path: pathlib.Path,
        app_path: pathlib.Path,
    ) -> str:
        """Load and concatenate JavaScript files with comments stripped.

        Parameters
        ----------
        renderer_path : pathlib.Path
            Directory containing JavaScript renderer files.
        app_path : pathlib.Path
            Path to the main ``app.js`` file (loaded last).

        Returns
        -------
        str
            Concatenated JavaScript code with comments stripped.
        """
        comment_pattern = re.compile(
            r"/\*\*[\s\S]*?\*/|/\*[\s\S]*?\*/|//.*?\n", re.MULTILINE | re.DOTALL
        )

        js_content = ""
        files = [
            pathlib.Path(renderer_path, file)
            for file in os.listdir(renderer_path)
            if file.endswith(".js")
        ]
        files.append(app_path)
        for js_file in files:
            with open(js_file, "r", encoding="utf-8") as f:
                content = f.read()
            cleaned_content = comment_pattern.sub("", content)
            js_content += f"\n// === {os.path.basename(js_file)} ===\n"
            js_content += cleaned_content + "\n"
        return js_content

    def render(self, data: Any, output_path: pathlib.Path) -> None:
        """Create an HTML report file from a ``ReportData`` instance.

        Parameters
        ----------
        data : Any
            The report data to render. Must expose ``model_dump()`` and
            ``model_dump_json()`` (the pydantic ``ReportData`` model).
        output_path : pathlib.Path
            Directory where ``report.html`` will be written.
        """
        html_output = self.template.render(
            report=data.model_dump(),
            report_json=data.model_dump_json(),
            javascript=self.javascript,
            **self.css_content,
            **self.page_templates,
            **self.component_templates,
        )
        output_file = pathlib.Path(output_path, "report.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_output)
