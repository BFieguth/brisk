# Brisk documentation build configuration file, created using sphinx-quickstart
# on December 7, 2024.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
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
]

templates_path = ['_templates']
exclude_patterns = []



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
