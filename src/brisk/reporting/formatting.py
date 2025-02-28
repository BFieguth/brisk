"""Formatting utilities for Brisk reports and logs.

This module provides utilities for formatting data structures into
human-readable output for reports, logs and other display purposes.
"""

def format_dict(d: dict) -> str:
    """Format dictionary with each key-value pair on a new line.

    Parameters
    ----------
    d : dict
        Dictionary to format

    Returns
    -------
    str
        Formatted string representation of dictionary

    Examples
    --------
    >>> d = {'a': 1, 'b': 2}
    >>> print(format_dict(d))
    'a': 1,
    'b': 2,
    """
    if not d:
        return "{}"
    return "\n".join(f"{key!r}: {value!r}," for key, value in d.items())
