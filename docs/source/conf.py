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

import logging

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
html_title = f"Brisk {version} Documentation"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.linkcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc", # must be loaded after autodoc
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
            'TrainingManager', 'Workflow', "TqdmLoggingHandler", "FileFormatter"
        ]))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_css_files = [
    "css/brisk.css",
]

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "show_prev_next": True,
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
sys.path.insert(0, os.path.abspath("../../src"))

def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object in Brisk.
    
    Adapted from matplotlib's implementation.
    """
    if domain != 'py':
        return None

    module_name = info['module']
    fullname = info['fullname']
    
    # Debug logging - this will help us understand what's happening
    if on_rtd:
        print(f"[LINKCODE DEBUG] Processing: {module_name}.{fullname}")

    # Get the module
    sub_module = sys.modules.get(module_name)
    if sub_module is None:
        if on_rtd:
            print(f"[LINKCODE DEBUG] Module {module_name} not found in sys.modules")
        try:
            # Try to import the module
            import importlib
            sub_module = importlib.import_module(module_name)
        except ImportError as e:
            if on_rtd:
                print(f"[LINKCODE DEBUG] Failed to import {module_name}: {e}")
            return None

    # Get the object
    obj = sub_module
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            if on_rtd:
                print(f"[LINKCODE DEBUG] Could not find attribute {part} in {obj}")
            return None

        if inspect.isfunction(obj):
            obj = inspect.unwrap(obj)

    # Get source file
    try:
        source_file = inspect.getsourcefile(obj)
    except (TypeError, OSError) as e:
        if on_rtd:
            print(f"[LINKCODE DEBUG] Could not get source file for {obj}: {e}")
        source_file = None

    # Fallback for __init__.py files
    if not source_file or source_file.endswith('__init__.py'):
        try:
            source_file = inspect.getsourcefile(sys.modules[obj.__module__])
        except (TypeError, AttributeError, KeyError) as e:
            if on_rtd:
                print(f"[LINKCODE DEBUG] Fallback source file lookup failed: {e}")
            source_file = None
    
    if not source_file:
        if on_rtd:
            print(f"[LINKCODE DEBUG] No source file found for {module_name}.{fullname}")
        return None

    # Get line numbers
    try:
        source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}"
        if len(source) > 1:
            linespec = f"#L{lineno:d}-L{lineno + len(source) - 1:d}"
    except (OSError, TypeError) as e:
        if on_rtd:
            print(f"[LINKCODE DEBUG] Could not get source lines: {e}")
        linespec = ""

    # Path resolution - this is likely where the issue is
    if on_rtd:
        print(f"[LINKCODE DEBUG] Raw source file path: {source_file}")
    
    # Try different approaches for path resolution
    try:
        # Method 1: Use brisk module location
        startdir = pathlib.Path(brisk.__file__).parent.parent.parent
        if on_rtd:
            print(f"[LINKCODE DEBUG] Start directory (brisk): {startdir}")
        
        source_file_rel = os.path.relpath(source_file, start=startdir).replace(os.path.sep, '/')
        if on_rtd:
            print(f"[LINKCODE DEBUG] Relative path (method 1): {source_file_rel}")
        
        # Check if path is valid
        if source_file_rel.startswith('src/brisk/'):
            source_file = source_file_rel
        else:
            # Method 2: Try from current working directory
            cwd = pathlib.Path.cwd()
            if on_rtd:
                print(f"[LINKCODE DEBUG] Current working directory: {cwd}")
            
            source_file_rel2 = os.path.relpath(source_file, start=cwd).replace(os.path.sep, '/')
            if on_rtd:
                print(f"[LINKCODE DEBUG] Relative path (method 2): {source_file_rel2}")
            
            if source_file_rel2.startswith('src/brisk/'):
                source_file = source_file_rel2
            else:
                # Method 3: Extract from absolute path
                if 'src/brisk/' in source_file:
                    # Find the src/brisk/ part and use everything from there
                    src_index = source_file.find('src/brisk/')
                    source_file = source_file[src_index:]
                    if on_rtd:
                        print(f"[LINKCODE DEBUG] Extracted path (method 3): {source_file}")
                else:
                    if on_rtd:
                        print(f"[LINKCODE DEBUG] Could not resolve path for {source_file}")
                    return None
    
    except Exception as e:
        if on_rtd:
            print(f"[LINKCODE DEBUG] Path resolution failed: {e}")
        return None

    # Final validation
    if not source_file.startswith('src/brisk/'):
        if on_rtd:
            print(f"[LINKCODE DEBUG] Final path validation failed: {source_file}")
        return None

    # Build GitHub URL
    github_user = "BFieguth"
    github_repo = "brisk" 
    github_branch = "main"

    url = (f"https://github.com/{github_user}/{github_repo}/blob/"
           f"{github_branch}/{source_file}{linespec}")
    
    if on_rtd:
        print(f"[LINKCODE DEBUG] Generated URL: {url}")
    
    return url
