"""Define Pydantic models to represent the results of model runs."""
from typing import List, Optional, Tuple, Dict, Any
import re

from pydantic import BaseModel, Field, model_validator

NUM_RE = re.compile(r"[+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?")
PURE_NUM_RE = re.compile(r"^[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?$")

def _round_to(n: float, decimals: int = 3) -> float:
    return round(float(n), decimals)


def _round_mean_std_string(s: str, decimals: int = 3) -> str:
    """
    Round mean and standard deviation values in a string of format: "1.23456
    (0.0000123)".

    Parameters
    ----------
    s : str
        The string to round. Must be of format: "1.2 (0.0)"
    decimals : int, optional
        The number of decimal places to round to.

    Returns
    -------
    str
        The rounded string.
    """
    pattern = re.compile(
        r"^\s*[+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?\s*\(\s*[+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?\s*\)\s*$" # pylint: disable=C0301
    )
    if not pattern.match(s):
        return s
    return NUM_RE.sub(lambda m: str(_round_to(float(m.group()), decimals)), s)


def _round_numbers_in_bracketed_list_string(s: str, decimals: int = 3) -> str:
    """
    Round numbers in a string of format: "[1.2, 0.0, 3.4]"

    Parameters
    ----------
    s : str
        The string to round. Must be of format: "[1.2, 0.0, 3.4]"
    decimals : int, optional
        The number of decimal places to round to.

    Returns
    -------
    str
        The rounded string.
    """
    trimmed = s.strip()
    if not (trimmed.startswith("[") and trimmed.endswith("]")):
        return s
    return NUM_RE.sub(lambda m: str(_round_to(float(m.group()), decimals)), s)


def _round_dictionary_string(s: str, decimals: int = 3) -> str:
    """Round numbers in a string of format: "{'a': 1.2, 'b': 0.0, 'c': 3.4}".

    Rounding is only performed on the values (numbers after a colon).

    Parameters
    ----------
    s : str
        The string to round. Must be of format: "{'a': 1.2, 'b': 0.0, 'c': 3.4}"
    decimals : int, optional
        The number of decimal places to round to.

    Returns
    -------
    str
        The rounded string.
    """
    trimmed = s.strip()
    if not (trimmed.startswith("{") and trimmed.endswith("}")):
        return s
    return re.sub(
        r"(:\s*)([+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?)",
        lambda m: m.group(1) + str(_round_to(float(m.group(2)), decimals)),
        s
    )


def _deep_round(value: Any, decimals: int = 3) -> Any:
    """Recursively round numbers in a nested data structure.

    Parameters
    ----------
    value : Any
        The value to round.
    decimals : int, optional
        The number of decimal places to round to.

    Returns
    -------
    Any
        The rounded value.
    """
    if value is None:
        return value
    if isinstance(value, float):
        return _round_to(value, decimals)
    if isinstance(value, (list, tuple)):
        items = [_deep_round(v, decimals) for v in value]
        return type(value)(items) if isinstance(value, tuple) else items
    if isinstance(value, dict):
        return {k: _deep_round(v, decimals) for k, v in value.items()}
    if isinstance(value, str):
        s = value.strip()
        # Avoid any probable HTML/SVG strings
        if "<" in s and ">" in s:
            return value
        if PURE_NUM_RE.fullmatch(s):
            return str(_round_to(float(s), decimals))

        ms = _round_mean_std_string(value, decimals)
        if ms != value:
            return ms

        bl = _round_numbers_in_bracketed_list_string(value, decimals)
        if bl != value:
            return bl

        dc = _round_dictionary_string(value, decimals)
        if dc != value:
            return dc

        return value
    return value


class RoundedModel(BaseModel):
    """Enforce rounding of all numbers in the model."""
    @model_validator(mode="before")
    @classmethod
    def _round_all_numbers(cls, values):
        return _deep_round(values, 3)


class TableData(RoundedModel):
    name: str
    description: Optional[str] = Field(
        None, description="Optional description text displayed below the table"
    )
    columns: List[str] = Field(
        ..., description="List of column headers"
    )
    rows: List[List[str]] = Field(
        ..., description="List of rows, each row is a list of cell values"
    )


class PlotData(RoundedModel):
    """Structure for all plots in the report."""
    name: str
    description: str
    image: str


class FeatureDistribution(RoundedModel):
    """Distribution of a feature across train and test splits."""
    ID: str
    tables: List[TableData]
    plot: PlotData


class DataManager(RoundedModel):
    """Represents a DataManager instance."""
    ID: str
    test_size: float
    n_splits: int
    split_method: str
    group_column: str
    stratified: str
    random_state: int | None
    scale_method: str


class Navbar(RoundedModel):
    """Data for the navbar."""
    brisk_version: str
    timestamp: str


class ExperimentGroup(RoundedModel):
    """Data for an ExperimentGroup card on the home page."""
    name: str
    description: str
    datasets: List[str] = Field(
        default_factory=list, description="List of dataset IDs"
    )
    experiments: List[str] = Field(
        default_factory=list, description="List of experiment IDs"
    )
    data_split_scores: Dict[str, List[Tuple[str, str, str, str]]] = Field(
        default_factory=dict,
        description="Best algorithm and score for each data split."
    )
    test_scores: Dict[str, TableData] = Field(
        default_factory=dict,
        description="Test data scores indexed on dataset name and split number."
    )


class Experiment(RoundedModel):
    """Results of a single experiment."""
    ID: str
    dataset: str
    algorithm: List[str] = Field(
        default_factory=list,
        description="Display names of algorithms in experiment"
    )
    tuned_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tuned hyperparameter names and values"
    )
    hyperparam_grid: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameter grid used for tuning"
    )
    tables: List[TableData] = Field(
        default_factory=list, description="List of tables for this experiment"
    )
    plots: List[PlotData] = Field(
        default_factory=list, description="List of plots for this experiment"
    )


class Dataset(RoundedModel):
    """Represents a dataset within an ExperimentGroup."""
    ID: str
    splits: List[str] = Field(
        default_factory=list, description="List of data split indexes"
    )
    split_sizes: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Size of dataset and train/test split"
    )
    split_target_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Target feature stats per split"
    )
    split_corr_matrices: Dict[str, PlotData] = Field(
        default_factory=dict, description="Correlation matrix per split"
    )
    data_manager_id: str
    features: List[str] = Field(
        default_factory=list, description="List of feature names"
    )
    split_feature_distributions: Dict[str, List[FeatureDistribution]] = Field(
        default_factory=dict, description="Feature distributions per split"
    )


class ReportData(RoundedModel):
    """Represents the entire report."""
    navbar: Navbar
    datasets: Dict[str, Dataset] = Field(
        default_factory=dict, description="Map IDs to Dataset instances"
    )
    experiments: Dict[str, Experiment] = Field(
        default_factory=dict, description="Map IDs to Experiment instances"
    )
    experiment_groups: List[ExperimentGroup] = Field(
        default_factory=list, description="List of experiment groups"
    )
    data_managers: Dict[str, DataManager] = Field(
        default_factory=dict, description="Map IDs to DataManager instances"
    )
