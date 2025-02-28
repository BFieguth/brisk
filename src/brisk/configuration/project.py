"""Project configuration utilities for Brisk.

This module provides utilities for locating and managing project-level
configuration files and directory structures.
"""

import pathlib
import functools

@functools.lru_cache
def find_project_root() -> pathlib.Path:
    """Find the project root directory containing .briskconfig.

    Searches current directory and parent directories for .briskconfig file.
    Result is cached to avoid repeated filesystem operations.

    Returns
    -------
    pathlib.Path
        Path to project root directory

    Raises
    ------
    FileNotFoundError
        If .briskconfig cannot be found in any parent directory

    Examples
    --------
    >>> root = find_project_root()
    >>> datasets_dir = root / 'datasets'
    >>> config_file = root / '.briskconfig'
    """
    current = pathlib.Path.cwd()
    while current != current.parent:
        if (current / ".briskconfig").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find .briskconfig in any parent directory"
    )
