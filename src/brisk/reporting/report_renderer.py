"""Generate HTML reports from ReportData instances.

This module provides functionality to render machine learning experiment results
into interactive HTML reports. It uses Jinja2 templating to combine data with
HTML templates, CSS styles, and JavaScript functionality.

The ReportRenderer class handles loading of all necessary assets (CSS, HTML
templates, JavaScript) and renders them into a complete HTML report that can be
viewed in a web browser.

Examples
--------
>>> from brisk.reporting.report_renderer import ReportRenderer
>>> from brisk.reporting.report_data import ReportData
>>> from pathlib import Path
>>> 
>>> # Create a report renderer
>>> renderer = ReportRenderer()
>>> 
>>> # Render a report
>>> report_data = ReportData(...)
>>> output_path = Path("output")
>>> renderer.render(report_data, output_path)
>>> # This creates output/report.html
"""
import os
from pathlib import Path
from typing import Dict
import re

from jinja2 import Environment, FileSystemLoader

from brisk.reporting.report_data import ReportData

class ReportRenderer:
    """Render a ReportData instance to an HTML report.
    
    This class handles the complete process of converting machine learning
    experiment data into an interactive HTML report. It loads all necessary
    assets (CSS, HTML templates, JavaScript) and uses Jinja2 templating to
    generate the final report.
    
    The renderer automatically loads assets from the reporting directory
    structure:
    - CSS files from `styles/` directory
    - HTML page templates from `pages/` directory  
    - HTML component templates from `components/` directory
    - JavaScript files from `js/renderers/` and `js/core/app.js`
    
    Attributes
    ----------
    css_content : Dict[str, str]
        Dictionary mapping CSS variable names to CSS content
    page_templates : Dict[str, str]
        Dictionary mapping template variable names to HTML page templates
    component_templates : Dict[str, str]
        Dictionary mapping component variable names to HTML component templates
    javascript : str
        Concatenated JavaScript code with comments stripped
    env : jinja2.Environment
        Jinja2 environment for template rendering
    template : jinja2.Template
        Main Jinja2 template for the report
        
    Examples
    --------
    >>> renderer = ReportRenderer()
    >>> # Assets are automatically loaded during initialization
    >>> print(len(renderer.css_content))  # Number of CSS files loaded
    >>> print(len(renderer.page_templates))  # Number of page templates loaded
    """
    def __init__(self) -> None:
        """Initialize the ReportRenderer and load all necessary assets.
        
        This constructor automatically loads all CSS files, HTML templates,
        and JavaScript files from the reporting directory structure. It sets
        up the Jinja2 environment and loads the main report template.
        
        Notes
        -----
        The constructor assumes a specific directory structure:
        - `styles/` - Contains CSS files
        - `pages/` - Contains HTML page templates
        - `components/` - Contains HTML component templates
        - `js/renderers/` - Contains JavaScript renderer files
        - `js/core/app.js` - Main JavaScript application file
        - `report.html` - Main Jinja2 template file
        """
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
        """Load all files in a directory and assign variable names to each file.

        This method scans a directory for files with a specific extension,
        reads their contents, and creates a dictionary mapping variable names
        (derived from filenames) to file contents.

        Parameters
        ----------
        dir_path : Path
            The path to the directory to load files from
        file_extension : str
            The file extension to filter by (e.g., '.css', '.html')
        name_extension : str
            String to replace the file extension with in variable names
            (e.g., '_css', '_template', '_component')

        Returns
        -------
        Dict[str, str]
            Dictionary mapping variable names to file contents
            
        Examples
        --------
        >>> renderer = ReportRenderer()
        >>> css_files = renderer._load_directory(
        ...     Path("styles"), ".css", "_css"
        ... )
        >>> # If styles/ contains 'main.css' and 'theme.css'
        >>> # Result: {
        >>> #     "main_css": "/* CSS content */",
        >>> #     "theme_css": "/* CSS content */"
        >>> # }
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
        """Load JavaScript files and concatenate them with comments stripped.

        This method loads all JavaScript files from a renderer directory and
        the main app.js file, strips comments and JSDoc, and concatenates them
        into a single string. The app.js file is always loaded last to ensure
        proper initialization order.

        Parameters
        ----------
        renderer_path : Path
            The path to the directory containing JavaScript renderer files
        app_path : Path
            The path to the main app.js file

        Returns
        -------
        str
            Concatenated JavaScript code with comments stripped
            
        Notes
        -----
        The method uses regex to remove:
        - Multi-line comments (/* ... */)
        - JSDoc comments (/** ... */)
        - Single-line comments (// ...)
        
        Examples
        --------
        >>> renderer = ReportRenderer()
        >>> js_content = renderer._load_javascript(
        ...     Path("js/renderers"), Path("js/core/app.js")
        ... )
        >>> print(len(js_content))  # Total length of concatenated JS
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

        This method takes a ReportData instance and renders it into a complete
        HTML report using the loaded templates, CSS, and JavaScript. The report
        is saved as 'report.html' in the specified output directory.

        Parameters
        ----------
        data : ReportData
            The machine learning experiment data to render into HTML
        output_path : Path
            The directory path where the report.html file will be written
            
        Notes
        -----
        The method creates a single HTML file that includes:
        - All CSS styles embedded in <style> tags
        - All JavaScript code embedded in <script> tags
        - Complete HTML structure with data rendered via Jinja2 templates
        
        The output file will be named 'report.html' and placed in the
        specified output directory.
        
        Examples
        --------
        >>> from brisk.reporting.report_data import ReportData
        >>> from pathlib import Path
        >>> 
        >>> renderer = ReportRenderer()
        >>> report_data = ReportData(...)
        >>> output_dir = Path("output")
        >>> output_dir.mkdir(exist_ok=True)
        >>> 
        >>> renderer.render(report_data, output_dir)
        >>> # Creates output/report.html
        >>> print((output_dir / "report.html").exists())  # True
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
