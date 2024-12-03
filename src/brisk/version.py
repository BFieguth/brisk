"""version.py

Exports:
    - __version__: The current version of the brisk-ml package.
"""
import pkg_resources

__version__ = pkg_resources.get_distribution("brisk-ml").version
