import os
import json
from typing import Dict

from jinja2 import Environment, FileSystemLoader

from src.brisk.reporting.new_report.report_data import report_data

def load_css(styles_dir: str) -> Dict[str, str]:
    """
    Load the css files so the style section of main file can be updated.
    """
    css_content = {}
    css_files = [file for file in os.listdir(styles_dir) if file.endswith('.css')]
    for file in css_files:
        css_path = os.path.join(styles_dir, file)
        variable_name = file.replace(".css", "_css")

        with open(css_path, "r", encoding="utf-8") as f:
            css_content[variable_name] = f.read()
            print(f"DEBUG: Loaded {file} ({len(css_content[variable_name])} characters)")

    return css_content


def load_page_template(pages_dir: str) -> Dict[str, str]:
    """Read templates for pages."""
    templates = {}
    page_files = {
        "home_template": os.path.join(pages_dir, "home.html"),
        "experiment_template": os.path.join(pages_dir, "experiment.html"),
        "dataset_template": os.path.join(pages_dir, "dataset.html")
    }
    for page_type, template_path in page_files.items():
        with open(template_path, "r", encoding="utf-8") as f:
            templates[page_type] = f.read()
            print(f"DEBUG: Loaded {page_type} ({len(templates[page_type])} characters)")

    return templates


def load_component_templates(component_dir: str) -> Dict[str, str]:
    """Read templates for reusable components."""
    templates = {}
    component_files = [file for file in os.listdir(component_dir) if file.endswith('.html')]
    for file in component_files:
        html_path = os.path.join(component_dir, file)
        component_name =  file.replace(".html", "_component")

        with open(html_path, "r", encoding="utf-8") as f:
            templates[component_name] = f.read()
            print(f"DEBUG: Loaded {component_name} ({len(templates[component_name])} characters)")

    return templates


def load_javascript(js_dir: str) -> str:
    """
    Load and concatenate all JavaScript files in the correct dependency order.
    """
    js_content = ""
    
    js_files_order = [
        # Renderers (no dependencies)
        os.path.join(js_dir, "renderers", "table.js"),
        os.path.join(js_dir, "renderers", "experiment_group_card.js"), 
        os.path.join(js_dir, "renderers", "home.js"),
        os.path.join(js_dir, "renderers", "dataset_page.js"),
        os.path.join(js_dir, "renderers", "experiment_page.js"),
        # Core (depends on renderers)
        os.path.join(js_dir, "core", "app.js")
    ]
    
    for js_file in js_files_order:
        if os.path.exists(js_file):
            with open(js_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Remove ES6 import/export statements
                # content = remove_es6_imports_exports(content)
                js_content += f"\n// === {os.path.basename(js_file)} ===\n"
                js_content += content + "\n"
                print(f"DEBUG: Loaded {js_file} ({len(content)} characters)")
        else:
            print(f"WARNING: JS file not found: {js_file}")
    
    return js_content

if __name__ == "__main__":
    styles_dir = "./src/brisk/reporting/new_report/styles"
    css_content = load_css(styles_dir)

    pages_dir = "./src/brisk/reporting/new_report/pages"
    page_templates = load_page_template(pages_dir)

    components_dir = "./src/brisk/reporting/new_report/components"
    component_templates = load_component_templates(components_dir)

    js_dir = "./src/brisk/reporting/new_report/js"
    javascript = load_javascript(js_dir)

    env = Environment(
        loader=FileSystemLoader(searchpath="./src/brisk/reporting/new_report")
    )
    template = env.get_template("report.html")

    html_output = template.render(
        report=report_data.model_dump(),
        report_json=report_data.model_dump_json(),
        javascript=javascript,
        **css_content,
        **page_templates,
        **component_templates # NOTE used for table-template (table.html, experiment_group_card.html)
    )

    with open("dev_report.html", "w", encoding="utf-8") as f:
        f.write(html_output)
        print("report created succesfully!")
