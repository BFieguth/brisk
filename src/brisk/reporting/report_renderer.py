"""Generate report.html from a ReportData instance.
"""
import os
from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader

from brisk.reporting.report_data import ReportData

class ReportRenderer():
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
            loader=FileSystemLoader(searchpath="./src/brisk/reporting")
        )
        self.template = self.env.get_template("report.html")

    def _load_directory(
        self,
        dir_path: Path,
        file_extension: str,
        name_extension: str
    ) -> Dict[str, str]:
        """
        Load all files in a directory and assign variable name to each file.
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
        """Load JavaScript files, ensure app.js is loaded last."""
        js_content = ""
        files = [
            Path(renderer_path, file) for file in os.listdir(renderer_path)
            if file.endswith(".js")
        ]
        files.append(app_path)
        for js_file in files:
            with open(js_file, "r", encoding="utf-8") as f:
                content = f.read()
            js_content += f"\n// === {os.path.basename(js_file)} ===\n"
            js_content += content + "\n"
        return js_content

    def render(self, data: ReportData, output_path: Path) -> None:
        """Create an HTML report"""
        html_output = self.template.render(
            report=data.model_dump(),
            report_json=data.model_dump_json(),
            javascript=self.javascript,
            **self.css_content,
            **self.page_templates,
            **self.component_templates
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_output)
