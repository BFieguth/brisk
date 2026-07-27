"""Port interfaces for the data loader system.

Defines structural contracts for data loaders.
"""

from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class DataLoaderPort(Protocol):
    """Contract that all data loader adapters must follow."""

    def supports(self, data_path: str | Path) -> bool:
        """Return True if this adapter can load the given file."""
        ...

    def load(self, data_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load the file and return it as a pandas DataFrame."""
        ...