"""Generate report.html from a ReportData instance."""
import os
from pathlib import Path
from typing import Dict
import re

from jinja2 import Environment, FileSystemLoader

from brisk.reporting.report_data import ReportData

class ReportRenderer():
    """Render a ReportData instance to an HTML report.

    Attributes
    ----------
    css_content : Dict[str, str]
        A dictionary of CSS content.
    page_templates : Dict[str, str]
        A dictionary of HTML page templates.
    component_templates : Dict[str, str]
        A dictionary of HTML component templates.
    javascript : str
        A string of JavaScript code.
    env : jinja2.Environment
        A Jinja2 environment.
    template : jinja2.Template
        A Jinja2 template for the report.
    """
    def __init__(self):
        report_dir = os.path.dirname(os.path.abspath(__file__))
        self.css_content = self._load_directory(
            Path(report_dir, "styles"), ".css", "_css"
        )
        self.page_templates = self._load_directory(
            Path(report_dir, "pages"), ".html", "_template"
        )
        self.component_templates = self._load_directory(
            Path(report_dir, "components"), ".html", "_component"
        )
        self.javascript = self._load_javascript(
            Path(report_dir, "js/renderers"), Path(report_dir, "js/core/app.js")
        )
        self.env = Environment(
            loader=FileSystemLoader(searchpath=report_dir)
        )
        self.template = self.env.get_template("report.html")

    def _load_directory(
        self,
        dir_path: Path,
        file_extension: str,
        name_extension: str
    ) -> Dict[str, str]:
        """Load all files in a directory and assign variable name to each file.

        Parameters
        ----------
        dir_path : Path
            The path to the directory to load.
        file_extension : str
            The extension of the files to load.
        name_extension : str
            String to replace file extension with.

        Returns
        -------
        Dict[str, str]
            A dictionary of variable names and file contents.
        """
        content = {}
        files = [
            file for file in os.listdir(dir_path)
            if file.endswith(file_extension)
        ]
        for file in files:
            file_path = Path(dir_path, file)
            variable_name = file.replace(file_extension, name_extension)
            with open(file_path, "r", encoding="utf-8") as f:
                content[variable_name] = f.read()
        return content

    def _load_javascript(self, renderer_path: Path, app_path: Path) -> str:
        """Load JavaScript files, ensure app.js is loaded last.

        Strips any comments and JSdocs from the files.

        Parameters
        ----------
        renderer_path : Path
            The path to the directory containing the JavaScript rendering files.
        app_path : Path
            The path to the app.js file.

        Returns
        -------
        str
            A string of JavaScript code.
        """
        comment_pattern = re.compile(
            r"/\*\*[\s\S]*?\*/|/\*[\s\S]*?\*/|//.*?\n", re.MULTILINE | re.DOTALL
        )

        js_content = ""
        files = [
            Path(renderer_path, file) for file in os.listdir(renderer_path)
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

    def render(self, data: ReportData, output_path: Path) -> None:
        """Create an HTML report file from a ReportData instance.

        Parameters
        ----------
        data : ReportData
            The data to render.
        output_path : Path
            The path to the directory to write the report to.
        """
        html_output = self.template.render(
            report=data.model_dump(),
            report_json=data.model_dump_json(),
            javascript=self.javascript,
            **self.css_content,
            **self.page_templates,
            **self.component_templates
        )
        output_file = Path(output_path, "report.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_output)
