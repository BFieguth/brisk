# Brisk documentation build configuration file, created using sphinx-quickstart
# on December 7, 2024.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import inspect
import sys
import pathlib

import brisk

project = 'Brisk'
copyright = '2024, Braeden Fieguth'
author = 'Braeden Fieguth'

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    version = brisk.__version__
else:
    version = "dev"

release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.linkcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc", # must be loaded after autodoc
    "sphinx_design",
]

# Add autosummary settings
autosummary_generate = True
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

numpydoc_show_class_members = False


# -- Setup descriptions -------------------------------------------------
docs_path = pathlib.Path(__file__).parent
sys.path.insert(0, str(docs_path))
from _descriptions import generate_list_table

def setup(app):
    """Generate RST files for object tables."""
    with open(docs_path / '_api_objects_table.rst', 'w') as f:
        f.write(generate_list_table())
    
    with open(docs_path / '_data_objects_table.rst', 'w') as f:
        f.write(generate_list_table(['DataManager', 'DataSplitInfo']))

    with open(docs_path / "_configuration_objects_table.rst", "w") as f:
        f.write(generate_list_table([
            "Configuration", "ExperimentFactory", "ExperimentGroup", 
            "Experiment", "ConfigurationManager", "AlgorithmWrapper",
            "find_project_root"
        ]))

    with open(docs_path / '_evaluation_objects_table.rst', 'w') as f:
        f.write(generate_list_table([
            'EvaluationManager', 'MetricManager', 'MetricWrapper'
        ]))

    with open(docs_path / '_reporting_objects_table.rst', 'w') as f:
        f.write(generate_list_table(['ReportManager']))

    with open(docs_path / '_training_objects_table.rst', 'w') as f:
        f.write(generate_list_table([
            'TrainingManager', 'Workflow', 'AlertMailer', "TqdmLoggingHandler",
            "FileFormatter"
        ]))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_css_files = [
    "css/brisk.css",
]

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "show_prev_next": False,
    "navbar_center": ["brisk_navbar.html"],
    "navbar_end": ["theme-switcher", "version-switcher", "brisk_icon_links"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 3,
    "switcher": {
        "json_url": "https://brisk.readthedocs.io/en/latest/_static/versions.json",
        "version_match": version
    }
}

# -- Linkcode settings ------------------------------------------------------
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object in Brisk.
    
    Adapted from matplotlib's implementation.
    """
    if domain != 'py':
        return None

    module_name = info['module']
    fullname = info['fullname']

    sub_module = sys.modules.get(module_name)
    if sub_module is None:
        return None

    obj = sub_module
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    if inspect.isfunction(obj):
        obj = inspect.unwrap(obj)

    try:
        source_file = inspect.getsourcefile(obj)
    except TypeError:
        source_file = None

    if not source_file or source_file.endswith('__init__.py'):
        try:
            source_file = inspect.getsourcefile(sys.modules[obj.__module__])
        except (TypeError, AttributeError, KeyError):
            source_file = None
    if not source_file:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}"
        if len(source) > 1:
            linespec = f"#L{lineno:d}-L{lineno + len(source) - 1:d}"
    except (OSError, TypeError) as e:
        linespec = ""

    startdir = pathlib.Path(brisk.__file__).parent.parent.parent
    source_file = os.path.relpath(source_file, start=startdir).replace(os.path.sep, '/')

    if not source_file.startswith('src/brisk/'):
        return None

    github_user = "BFieguth"
    github_repo = "brisk"
    github_branch = "main"

    url = (f"https://github.com/{github_user}/{github_repo}/blob/"
           f"{github_branch}/{source_file}{linespec}")
    return url
