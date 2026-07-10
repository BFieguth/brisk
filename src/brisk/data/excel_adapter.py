"""Excel loading adapter for the data loader system.

This module contains the ExcelAdapter class, which handles loading one sheet
from an Excel workbook into a pandas DataFrame. It performs basic file
validation before loading, provides clearer error messages for common Excel
loading failures, and includes optional validation and cleanup controls for
column names and blank rows.

The adapter is intended to be generic and reusable across downstream data
loading workflows. Dataset-specific validation, such as required biological,
clinical, or experimental columns, should usually be handled by a separate
validator after the Excel file has been loaded.

Exports:
    ExcelAdapter: A class for validating and loading Excel files into pandas
    DataFrames.
"""

from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile

import pandas as pd


ColumnNameHandling = Literal[
    "ignore",
    "validate",
    "clean",
    "clean_deduplicate",
]

BlankRowHandling = Literal[
    "allow",
    "reject",
    "drop",
]


class ExcelAdapter:
    """A class that loads Excel files into pandas DataFrames.

    This adapter checks whether a path points to a supported Excel file and
    safely loads one sheet using pandas. The load process includes basic
    validation for file existence, file type, empty files, read errors, and
    whether the loaded DataFrame has columns. It also supports optional controls
    for requiring data rows, validating or cleaning column names, handling
    completely blank rows, and selecting a single sheet.

    Notes
    -----
    The load method always performs the following checks:
    - The path has a supported Excel suffix
    - The path exists
    - The path points to a file rather than a directory
    - The file is not zero bytes
    - The requested sheet can be read by pandas
    - The loaded result is a pandas DataFrame
    - The loaded DataFrame contains at least one column

    Optional validation and cleanup controls include:
    - sheet_name: select one sheet by name or zero-based index
    - require_rows: reject sheets with headers but no data rows
    - column_name_handling: ignore, validate, clean, or clean and deduplicate
      column names
    - blank_row_handling: allow, reject, or drop completely blank rows

    Examples
    --------
    Load the first sheet with default validation:
        >>> adapter = ExcelAdapter()
        >>> df = adapter.load("data.xlsx")

    Load a specific sheet by name:
        >>> df = adapter.load("data.xlsx", sheet_name="Sheet1")

    Load a specific sheet by index:
        >>> df = adapter.load("data.xlsx", sheet_name=1)

    Load a sheet while cleaning column names and dropping blank rows:
        >>> df = adapter.load(
        ...     "data.xlsx",
        ...     column_name_handling="clean_deduplicate",
        ...     blank_row_handling="drop"
        ... )
    """

    _SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".xlsm"}

    _VALID_COLUMN_NAME_HANDLING = {
        "ignore",
        "validate",
        "clean",
        "clean_deduplicate",
    }

    _VALID_BLANK_ROW_HANDLING = {
        "allow",
        "reject",
        "drop",
    }

    def supports(self, data_path: str | Path) -> bool:
        """Return True if the file is a supported Excel file."""
        return Path(data_path).suffix.lower() in self._SUPPORTED_SUFFIXES

    def load(
        self,
        data_path: str | Path,
        *,
        sheet_name: str | int = 0,
        require_rows: bool = False,
        column_name_handling: ColumnNameHandling = "ignore",
        blank_row_handling: BlankRowHandling = "allow",
        **read_excel_kwargs: Any,
    ) -> pd.DataFrame:
        """Load one Excel sheet and return it as a pandas DataFrame."""

        path = Path(data_path)

        self._validate_options(
            sheet_name=sheet_name,
            column_name_handling=column_name_handling,
            blank_row_handling=blank_row_handling,
        )

        self._validate_path(path)

        read_excel_options = dict(read_excel_kwargs)
        read_excel_options["sheet_name"] = sheet_name

        df = self._read_excel(path, **read_excel_options)

        self._validate_is_dataframe(df, path, sheet_name)
        self._validate_has_columns(df, path, sheet_name)

        df = self._handle_column_names(
            df=df,
            path=path,
            sheet_name=sheet_name,
            column_name_handling=column_name_handling,
        )

        df = self._handle_blank_rows(
            df=df,
            path=path,
            sheet_name=sheet_name,
            blank_row_handling=blank_row_handling,
        )

        if require_rows and df.empty:
            raise ValueError(
                f"Excel sheet contains no data rows: {path}, sheet={sheet_name!r}"
            )

        return df

    def _validate_options(
        self,
        *,
        sheet_name: str | int,
        column_name_handling: str,
        blank_row_handling: str,
    ) -> None:
        """Validate adapter-level options."""

        if sheet_name is None:
            raise ValueError(
                "sheet_name=None is not supported because this adapter returns "
                "one pandas DataFrame at a time. Provide a sheet name or "
                "zero-based sheet index instead."
            )

        if isinstance(sheet_name, bool) or not isinstance(sheet_name, (str, int)):
            raise TypeError(
                "sheet_name must be a sheet name string or a zero-based integer "
                f"index, got {type(sheet_name).__name__}."
            )

        if isinstance(sheet_name, str) and sheet_name.strip() == "":
            raise ValueError("sheet_name cannot be an empty string.")

        if isinstance(sheet_name, int) and sheet_name < 0:
            raise ValueError("sheet_name index must be zero or greater.")

        if column_name_handling not in self._VALID_COLUMN_NAME_HANDLING:
            raise ValueError(
                "Invalid column_name_handling value: "
                f"{column_name_handling!r}. Expected one of "
                f"{sorted(self._VALID_COLUMN_NAME_HANDLING)}."
            )

        if blank_row_handling not in self._VALID_BLANK_ROW_HANDLING:
            raise ValueError(
                "Invalid blank_row_handling value: "
                f"{blank_row_handling!r}. Expected one of "
                f"{sorted(self._VALID_BLANK_ROW_HANDLING)}."
            )

    def _validate_path(self, path: Path) -> None:
        """Validate that the path points to a readable Excel file."""

        if not self.supports(path):
            raise ValueError(f"ExcelAdapter cannot load non-Excel file: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Excel file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Excel path is not a file: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"Excel file is empty: {path}")

    def _read_excel(self, path: Path, **read_excel_options: Any) -> pd.DataFrame:
        """Read the Excel file with pandas and provide clearer errors."""

        try:
            return pd.read_excel(path, **read_excel_options)

        except ImportError as exc:
            raise ImportError(
                "Excel file could not be loaded because a required Excel engine "
                "is not installed. You may need openpyxl for .xlsx/.xlsm files "
                "or xlrd for .xls files."
            ) from exc

        except BadZipFile as exc:
            raise ValueError(
                f"Excel file appears to be corrupted or is not a valid workbook: {path}"
            ) from exc

        except ValueError as exc:
            raise ValueError(
                "Excel file could not be loaded. Check that the requested sheet "
                f"exists and that the file format is supported. File: {path}"
            ) from exc

        except OSError as exc:
            raise OSError(f"Excel file could not be opened: {path}") from exc

    def _validate_is_dataframe(
        self,
        df: pd.DataFrame,
        path: Path,
        sheet_name: str | int,
    ) -> None:
        """Validate that pandas returned a single DataFrame."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "ExcelAdapter expected pandas.read_excel to return a DataFrame, "
                f"but got {type(df).__name__}. File: {path}, sheet={sheet_name!r}"
            )

    def _validate_has_columns(
        self,
        df: pd.DataFrame,
        path: Path,
        sheet_name: str | int,
    ) -> None:
        """Validate that the loaded DataFrame has at least one column."""

        if df.shape[1] == 0:
            raise ValueError(
                f"Excel sheet contains no columns: {path}, sheet={sheet_name!r}"
            )

    def _handle_column_names(
        self,
        *,
        df: pd.DataFrame,
        path: Path,
        sheet_name: str | int,
        column_name_handling: ColumnNameHandling,
    ) -> pd.DataFrame:
        """Validate or clean column names."""

        if column_name_handling == "ignore":
            return df

        current_columns = list(df.columns)

        if column_name_handling == "validate":
            self._validate_column_names(
                current_columns,
                path,
                sheet_name,
                check_whitespace=True,
                check_duplicates=True,
            )
            return df

        cleaned_columns = [
            self._clean_column_name(column) for column in current_columns
        ]

        self._validate_column_names(
            cleaned_columns,
            path,
            sheet_name,
            check_whitespace=False,
            check_duplicates=False,
        )

        duplicate_columns = self._find_duplicates(cleaned_columns)

        if duplicate_columns:
            if column_name_handling == "clean":
                raise ValueError(
                    "Cleaning column names would create duplicate columns in "
                    f"{path}, sheet={sheet_name!r}: {duplicate_columns}"
                )

            cleaned_columns = self._deduplicate_columns(cleaned_columns)

            self._validate_column_names(
                cleaned_columns,
                path,
                sheet_name,
                check_whitespace=False,
                check_duplicates=True,
            )

        cleaned_df = df.copy()
        cleaned_df.columns = cleaned_columns

        return cleaned_df

    def _validate_column_names(
        self,
        columns: list[Any],
        path: Path,
        sheet_name: str | int,
        *,
        check_whitespace: bool = True,
        check_duplicates: bool = True,
    ) -> None:
        """Validate column names for common problems."""

        issues: list[str] = []

        blank_columns = [
            column for column in columns if str(column).strip() == ""
        ]

        if blank_columns:
            issues.append("blank column names")

        unnamed_columns = [
            column for column in columns if str(column).startswith("Unnamed:")
        ]

        if unnamed_columns:
            issues.append(f"unnamed columns: {unnamed_columns}")

        if check_whitespace:
            whitespace_columns = [
                column
                for column in columns
                if isinstance(column, str) and column != column.strip()
            ]

            if whitespace_columns:
                issues.append(
                    f"columns with leading/trailing spaces: {whitespace_columns}"
                )

        if check_duplicates:
            duplicate_columns = self._find_duplicates(columns)

            if duplicate_columns:
                issues.append(f"duplicate columns: {duplicate_columns}")

        if issues:
            raise ValueError(
                f"Invalid column names in Excel file {path}, "
                f"sheet={sheet_name!r}: " + "; ".join(issues)
            )

    def _clean_column_name(self, column: Any) -> Any:
        """Clean a single column name without changing the data itself."""

        if isinstance(column, str):
            return column.strip()

        return column

    def _find_duplicates(self, values: list[Any]) -> list[Any]:
        """Find duplicate values while preserving order."""

        seen = set()
        duplicates = []

        for value in values:
            if value in seen:
                if value not in duplicates:
                    duplicates.append(value)
            else:
                seen.add(value)

        return duplicates

    def _deduplicate_columns(self, columns: list[Any]) -> list[Any]:
        """Rename duplicate columns using suffixes like _1, _2, _3."""

        used = set()
        counts: dict[str, int] = {}
        deduplicated_columns: list[Any] = []

        for column in columns:
            if column not in used:
                deduplicated_columns.append(column)
                used.add(column)
                counts[str(column)] = 1
                continue

            base_name = str(column)
            number = counts.get(base_name, 1)
            new_column = f"{base_name}_{number}"

            while new_column in used:
                number += 1
                new_column = f"{base_name}_{number}"

            deduplicated_columns.append(new_column)
            used.add(new_column)
            counts[base_name] = number + 1

        return deduplicated_columns

    def _handle_blank_rows(
        self,
        *,
        df: pd.DataFrame,
        path: Path,
        sheet_name: str | int,
        blank_row_handling: BlankRowHandling,
    ) -> pd.DataFrame:
        """Allow, reject, or drop completely blank rows."""

        if blank_row_handling == "allow":
            return df

        blank_rows = df.apply(self._is_blank_row, axis=1)

        if not blank_rows.any():
            return df

        blank_row_count = int(blank_rows.sum())

        if blank_row_handling == "reject":
            raise ValueError(
                f"Excel sheet contains {blank_row_count} completely blank row(s): "
                f"{path}, sheet={sheet_name!r}"
            )

        if blank_row_handling == "drop":
            return df.loc[~blank_rows].reset_index(drop=True)

        return df

    def _is_blank_row(self, row: pd.Series) -> bool:
        """Return True if every value in a row is blank or missing."""

        return all(
            pd.isna(value) or (isinstance(value, str) and value.strip() == "")
            for value in row
        )