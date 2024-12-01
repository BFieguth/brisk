"""version.py

Exports:
    - __version__: The current version of the brisk-ml package.
"""
import os
import toml

current_dir = os.path.dirname(__file__)
pyproject_path = os.path.join(current_dir, "..", "..", "pyproject.toml")
pyproject = toml.load(pyproject_path)
__version__ = pyproject["tool"]["poetry"]["version"]
