"""Registry for selecting data loader adapters.

This module contains the DataLoaderRegistry class, which stores available data
loader adapters and selects the correct adapter for a given input file. The
registry uses each adapter's supports method to determine whether that adapter
can load the provided file, then delegates loading to the selected adapter.

The registry is intentionally lightweight. It is responsible for adapter
selection, while file-specific validation and loading logic are handled by the
individual adapters.

Exports:
    DataLoaderRegistry: A class for registering data loader adapters and loading
    files through the appropriate adapter.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from brisk.ports.data_loader import DataLoaderPort
from brisk.data.csv_adapter import CSVAdapter
from brisk.data.excel_adapter import ExcelAdapter
from brisk.data.parquet_adapter import ParquetAdapter
from brisk.data.sqlite_adapter import SQLiteAdapter


class DataLoaderRegistry:
    """A registry that stores and selects data loader adapters.

    The registry maintains a list of available data loader adapters. When a file
    is loaded, the registry checks each adapter in order and uses the first one
    whose supports method returns True for the provided file path.

    Parameters
    ----------
    None

    Attributes
    ----------
    _adapters : list[DataLoaderPort]
        The list of registered data loader adapters.

    Notes
    -----
    The registry only decides which adapter should handle a file. It does not
    perform detailed file validation itself. Validation such as checking whether
    a CSV file exists, whether it is empty, or whether its column names are valid
    should be handled inside the selected adapter.

    Additional adapters can be added with the register method. Adapters are
    checked in registration order, so if multiple adapters support the same file,
    the first matching adapter will be used.

    Examples
    --------
    Load a file using the default adapters:
        >>> registry = DataLoaderRegistry()
        >>> df = registry.load("data.csv")

    Pass adapter-specific options through the registry:
        >>> df = registry.load(
        ...     "data.csv",
        ...     require_rows=True,
        ...     column_name_handling="clean_deduplicate",
        ...     blank_row_handling="drop"
        ... )

    Register a custom adapter:
        >>> registry = DataLoaderRegistry()
        >>> registry.register(MyCustomAdapter())
    """

    def __init__(self) -> None:
        """Initialize the registry with the default data loader adapters."""
        self._adapters: list[DataLoaderPort] = [
            CSVAdapter(),
            ExcelAdapter(),
            ParquetAdapter(),
            SQLiteAdapter(),
        ]

    def register(self, adapter: DataLoaderPort) -> None:
        """Register a new data loader adapter.

        Parameters
        ----------
        adapter : DataLoaderPort
            The adapter to add to the registry.

        Notes
        -----
        Newly registered adapters are appended to the end of the adapter list.
        This means the default adapters will be checked first unless a custom
        adapter is inserted manually into _adapters.
        """
        self._adapters.append(adapter)

    def get_adapter(self, data_path: str | Path) -> DataLoaderPort:
        """Return the first adapter that supports the given data path.

        Parameters
        ----------
        data_path : str or Path
            Path to the file that should be loaded.

        Returns
        -------
        DataLoaderPort
            The first registered adapter whose supports method returns True.

        Raises
        ------
        ValueError
            If no registered adapter supports the provided file path.
        """
        for adapter in self._adapters:
            if adapter.supports(data_path):
                return adapter

        raise ValueError(f"No data loader found for file: {data_path}")

    def load(self, data_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load data using the matching data loader adapter.

        Parameters
        ----------
        data_path : str or Path
            Path to the file that should be loaded.
        **kwargs : Any
            Additional keyword arguments passed directly to the selected
            adapter's load method.

        Returns
        -------
        pandas.DataFrame
            The loaded data.

        Raises
        ------
        ValueError
            If no registered adapter supports the provided file path.

        Notes
        -----
        Adapter-specific loading options can be passed through this method. For
        example, CSV-specific options such as require_rows,
        column_name_handling, blank_row_handling, encoding, and
        expected_delimiter will be forwarded to CSVAdapter.load.
        """
        adapter = self.get_adapter(data_path)
        return adapter.load(data_path, **kwargs)