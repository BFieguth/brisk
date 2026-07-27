"""CSV loading adapter for the data loader system.

This module contains the CSVAdapter class, which handles loading CSV files into
pandas DataFrames. It performs basic file validation before loading, provides
clearer error messages for common CSV loading failures, and includes optional
validation and cleanup controls for column names, blank rows, file encoding, and
delimiter handling.

The adapter is intended to be generic and reusable across downstream data
loading workflows. Dataset-specific validation is not done here.

Exports:
    CSVAdapter: A class for validating and loading CSV files into pandas
    DataFrames.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


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


class CSVAdapter:
    """A class that loads CSV files into pandas DataFrames.

    This adapter checks whether a path points to a CSV file and safely loads it
    using pandas. The load process includes basic validation for file existence,
    file type, empty files, parsing errors. It also supports optional controls for requiring data rows,
    validating or cleaning column names, handling completely blank rows, setting
    the expected text encoding, and setting the expected delimiter.

    Methods
    -------
    supports(data_path)
        Return True if the provided path has a ".csv" suffix.

    load(data_path, require_rows=False, column_name_handling="ignore",
         blank_row_handling="allow", encoding=None, expected_delimiter=None,
         **read_csv_kwargs)
        Validate and load a CSV file as a pandas DataFrame.

    Notes
    -----
    The load method always performs the following checks:
    - The path has a ".csv" suffix
    - The path exists
    - The path points to a file rather than a directory
    - The file is not zero bytes
    - The file can be parsed by pandas
    - The loaded DataFrame contains at least one column

    Optional validation and cleanup controls include:
    - require_rows: reject CSV files with headers but no data rows
    - column_name_handling: ignore, validate, clean, or clean and deduplicate
      column names
    - blank_row_handling: allow, reject, or drop completely blank rows
    - encoding: pass a specific text encoding to pandas
    - expected_delimiter: set the expected delimiter used by pandas

    Column name cleaning is conservative and only modifies headers, not the
    underlying data values. For example, leading and trailing spaces in column
    names can be removed, while the cell values in the DataFrame remain
    unchanged.

    Examples
    --------
    Load a CSV file with default validation:
        >>> adapter = CSVAdapter()
        >>> df = adapter.load("data.csv")

    Load a CSV file and require at least one data row:
        >>> df = adapter.load("data.csv", require_rows=True)

    Load a CSV file while cleaning column names and dropping blank rows:
        >>> df = adapter.load(
        ...     "data.csv",
        ...     column_name_handling="clean_deduplicate",
        ...     blank_row_handling="drop"
        ... )

    Load a CSV file with explicit encoding and delimiter settings:
        >>> df = adapter.load(
        ...     "data.csv",
        ...     encoding="utf-8",
        ...     expected_delimiter=","
        ... )
    """

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
        """Return True if the file is a CSV file."""
        return Path(data_path).suffix.lower() == ".csv"

    def load(
        self,
        data_path: str | Path,
        *,
        require_rows: bool = False,
        column_name_handling: ColumnNameHandling = "ignore",
        blank_row_handling: BlankRowHandling = "allow",
        encoding: str | None = None,
        expected_delimiter: str | None = None,
        **read_csv_kwargs: Any,
    ) -> pd.DataFrame:
        """Load a CSV file and return it as a pandas DataFrame."""

        path = Path(data_path)

        self._validate_options(
            column_name_handling=column_name_handling,
            blank_row_handling=blank_row_handling,
            expected_delimiter=expected_delimiter,
        )

        self._validate_path(path)

        read_csv_options = dict(read_csv_kwargs)

        if encoding is not None:
            read_csv_options["encoding"] = encoding

        if expected_delimiter is not None:
            self._apply_expected_delimiter(
                expected_delimiter=expected_delimiter,
                read_csv_options=read_csv_options,
            )

        if blank_row_handling in {"reject", "drop"}:
            read_csv_options.setdefault("skip_blank_lines", False)

        df = self._read_csv(path, **read_csv_options)

        self._validate_has_columns(df, path)

        df = self._handle_column_names(
            df=df,
            path=path,
            column_name_handling=column_name_handling,
        )

        df = self._handle_blank_rows(
            df=df,
            path=path,
            blank_row_handling=blank_row_handling,
        )

        if require_rows and df.empty:
            raise ValueError(f"CSV file contains no data rows: {path}")

        return df

    def _validate_options(
        self,
        *,
        column_name_handling: str,
        blank_row_handling: str,
        expected_delimiter: str | None,
    ) -> None:
        """Validate adapter-level options."""

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

        if expected_delimiter is not None and len(expected_delimiter) != 1:
            raise ValueError(
                "expected_delimiter must be a single character, "
                f"got {expected_delimiter!r}."
            )

    def _validate_path(self, path: Path) -> None:
        """Validate that the path points to a readable CSV file."""

        if not self.supports(path):
            raise ValueError(f"CSVAdapter cannot load non-CSV file: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"CSV path is not a file: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {path}")

    def _apply_expected_delimiter(
        self,
        *,
        expected_delimiter: str,
        read_csv_options: dict[str, Any],
    ) -> None:
        """Apply and validate the expected delimiter option."""

        existing_sep = read_csv_options.get("sep")
        existing_delimiter = read_csv_options.get("delimiter")

        if existing_sep is not None and existing_sep != expected_delimiter:
            raise ValueError(
                "Conflicting delimiter settings: "
                f"expected_delimiter={expected_delimiter!r}, "
                f"sep={existing_sep!r}."
            )

        if existing_delimiter is not None and existing_delimiter != expected_delimiter:
            raise ValueError(
                "Conflicting delimiter settings: "
                f"expected_delimiter={expected_delimiter!r}, "
                f"delimiter={existing_delimiter!r}."
            )

        # Pandas does not like having both sep and delimiter.
        # Normalize everything to sep.
        read_csv_options.pop("delimiter", None)
        read_csv_options["sep"] = expected_delimiter

    def _read_csv(self, path: Path, **read_csv_options: Any) -> pd.DataFrame:
        """Read the CSV file with pandas and provide clearer errors."""

        try:
            return pd.read_csv(path, **read_csv_options)

        except EmptyDataError as exc:
            raise ValueError(
                f"CSV file could not be loaded because it has no readable data: {path}"
            ) from exc

        except UnicodeDecodeError as exc:
            raise ValueError(
                "CSV file could not be decoded. Try passing a different encoding, "
                f"such as encoding='utf-8' or encoding='latin1'. File: {path}"
            ) from exc

        except ParserError as exc:
            raise ValueError(
                "CSV file could not be parsed. Check the delimiter, quoting, "
                f"or malformed rows. File: {path}"
            ) from exc

        except OSError as exc:
            raise OSError(f"CSV file could not be opened: {path}") from exc

    def _validate_has_columns(self, df: pd.DataFrame, path: Path) -> None:
        """Validate that the loaded DataFrame has at least one column."""

        if df.shape[1] == 0:
            raise ValueError(f"CSV file contains no columns: {path}")

    def _handle_column_names(
        self,
        *,
        df: pd.DataFrame,
        path: Path,
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
            check_whitespace=False,
            check_duplicates=False,
        )

        duplicate_columns = self._find_duplicates(cleaned_columns)

        if duplicate_columns:
            if column_name_handling == "clean":
                raise ValueError(
                    "Cleaning column names would create duplicate columns in "
                    f"{path}: {duplicate_columns}"
                )

            cleaned_columns = self._deduplicate_columns(cleaned_columns)

            self._validate_column_names(
                cleaned_columns,
                path,
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
                f"Invalid column names in CSV file {path}: "
                + "; ".join(issues)
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
                f"CSV file contains {blank_row_count} completely blank row(s): {path}"
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