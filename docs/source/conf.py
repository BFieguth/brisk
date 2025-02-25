# Brisk documentation build configuration file, created using sphinx-quickstart
# on December 7, 2024.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import pathlib

import brisk

project = 'Brisk'
copyright = '2024, Braeden Fieguth'
author = 'Braeden Fieguth'
version = brisk.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc", # must be loaded after autodoc
    "sphinx.ext.viewcode",
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
}
