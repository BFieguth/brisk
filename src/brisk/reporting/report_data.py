"""Define Pydantic models to represent the results of model runs.

This module provides Pydantic models for structuring experiment results. It
includes utilities for rounding numerical values and models for representing
various components of a machine learning report including datasets, experiments,
plots, and tables.

The module uses a custom `RoundedModel` base class that automatically rounds all
numerical values to 3 decimal places for consistent display in reports.

Classes
-------
RoundedModel
    Base Pydantic model that enforces rounding of all numbers
TableData
    Represents tabular data with columns and rows
PlotData
    Represents plot data with name, description, and image
FeatureDistribution
    Distribution of a feature across train and test splits
DataManager
    Represents a DataManager instance configuration
Navbar
    Data for the navigation bar
ExperimentGroup
    Data for an ExperimentGroup card on the home page
Experiment
    Results of a single machine learning experiment
Dataset
    Represents a dataset within an ExperimentGroup
ReportData
    Root model representing the entire report

Functions
---------
_round_to
    Round a float to specified decimal places
_round_mean_std_string
    Round mean and standard deviation values in a string
_round_numbers_in_bracketed_list_string
    Round numbers in a bracketed list string
_round_dictionary_string
    Round numbers in a dictionary string
_deep_round
    Recursively round numbers in nested data structures

Examples
--------
>>> from brisk.reporting.report_data import TableData, PlotData
>>> table = TableData(
...     name="Accuracy Scores",
...     columns=["Algorithm", "Score"],
...     rows=[["Random Forest", "0.95"], ["SVM", "0.92"]]
... )
>>> plot = PlotData(
...     name="Feature Importance",
...     description="Shows feature importance scores",
...     image="<svg>...</svg>"
... )
"""
from typing import List, Optional, Tuple, Dict, Any, Union
import re

from pydantic import BaseModel, Field, model_validator

NUM_RE = re.compile(r"[+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?")
PURE_NUM_RE = re.compile(r"^[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?$")

def _round_to(n: float, decimals: int = 3) -> float:
    """Round a float to specified decimal places.
    
    Parameters
    ----------
    n : float
        The number to round
    decimals : int, default=3
        The number of decimal places to round to
        
    Returns
    -------
    float
        The rounded number
        
    Examples
    --------
    >>> _round_to(3.14159, 2)
    3.14
    >>> _round_to(1.234567, 3)
    1.235
    """
    return round(float(n), decimals)


def _round_mean_std_string(s: str, decimals: int = 3) -> str:
    """Round mean and standard deviation values in a string of format:
    "1.23456 (0.0000123)".

    Parameters
    ----------
    s : str
        The string to round. Must be of format: "1.2 (0.0)"
    decimals : int, default=3
        The number of decimal places to round to

    Returns
    -------
    str
        The rounded string
        
    Examples
    --------
    >>> _round_mean_std_string("1.23456 (0.0000123)", 2)
    "1.23 (0.00)"
    >>> _round_mean_std_string("0.95 (0.02)", 1)
    "0.9 (0.0)"
    """
    pattern = re.compile(
        r"^\s*[+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?\s*\(\s*[+\-]?(?:\d*\.?\d+)(?:[eE][+\-]?\d+)?\s*\)\s*$" # pylint: disable=C0301
    )
    if not pattern.match(s):
        return s
    return NUM_RE.sub(lambda m: str(_round_to(float(m.group()), decimals)), s)


def _round_numbers_in_bracketed_list_string(s: str, decimals: int = 3) -> str:
    """Round numbers in a string of format: "[1.2, 0.0, 3.4]".

    Parameters
    ----------
    s : str
        The string to round. Must be of format: "[1.2, 0.0, 3.4]"
    decimals : int, default=3
        The number of decimal places to round to

    Returns
    -------
    str
        The rounded string
        
    Examples
    --------
    >>> _round_numbers_in_bracketed_list_string("[1.23456, 0.00123, 3.4567]", 2)
    "[1.23, 0.0, 3.46]"
    >>> _round_numbers_in_bracketed_list_string("[0.95, 0.02, 0.88]", 1)
    "[0.9, 0.0, 0.9]"
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
    decimals : int, default=3
        The number of decimal places to round to

    Returns
    -------
    str
        The rounded string
        
    Examples
    --------
    >>> _round_dictionary_string("{'accuracy': 0.9543, 'precision': 0.0123}", 2)
    "{'accuracy': 0.95, 'precision': 0.0}"
    >>> _round_dictionary_string("{'score': 0.88, 'error': 0.12}", 1)
    "{'score': 0.9, 'error': 0.1}"
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

    This function traverses nested data structures (lists, tuples, dicts) and
    rounds all numerical values found within them. It also handles special
    string formats like mean ± std, bracketed lists, and dictionary strings.

    Parameters
    ----------
    value : Any
        The value to round. Can be a number, string, list, tuple, dict, or None
    decimals : int, default=3
        The number of decimal places to round to

    Returns
    -------
    Any
        The rounded value with the same structure as the input
        
    Examples
    --------
    >>> _deep_round([1.234, 2.567, {"score": 0.987}], 2)
    [1.23, 2.57, {"score": 0.99}]
    >>> _deep_round("1.23456 (0.0000123)", 2)
    "1.23 (0.00)"
    >>> _deep_round({"accuracy": 0.95432, "scores": [0.1, 0.2]}, 1)
    {"accuracy": 0.9, "scores": [0.1, 0.2]}
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
    """Base Pydantic model that enforces rounding of all numbers.
    
    This model automatically rounds all numerical values to 3 decimal places
    before validation. It uses the `_deep_round` function to handle nested
    data structures and special string formats.
    
    Attributes
    ----------
    All attributes are automatically rounded to 3 decimal places
    
    Notes
    -----
    This class should be used as a base class for all models that need
    consistent numerical rounding for display purposes.
    
    Examples
    --------
    >>> class MyModel(RoundedModel):
    ...     value: float
    ...     scores: List[float]
    >>> model = MyModel(value=1.234567, scores=[0.1, 0.234567])
    >>> model.value
    1.235
    >>> model.scores
    [0.1, 0.235]
    """
    @model_validator(mode="before")
    @classmethod
    def _round_all_numbers(cls, values: Any) -> Any:
        """Round all numbers in the model data.
        
        Parameters
        ----------
        values : Any
            The input values to be validated and rounded
            
        Returns
        -------
        Any
            The values with all numbers rounded to 3 decimal places
        """
        return _deep_round(values, 3)


class TableData(RoundedModel):
    """Represents tabular data with columns and rows.
    
    This model is used to structure tabular data for display in reports.
    It includes metadata like name and description along with the actual
    table structure.
    
    Attributes
    ----------
    name : str
        The name/title of the table
    description : Optional[str]
        Optional description text displayed below the table
    columns : List[str]
        List of column headers
    rows : List[List[str]]
        List of rows, each row is a list of cell values
        
    Examples
    --------
    >>> table = TableData(
    ...     name="Model Performance",
    ...     description="Cross-validation results",
    ...     columns=["Algorithm", "Accuracy", "Precision"],
    ...     rows=[["Random Forest", "0.95", "0.92"], ["SVM", "0.93", "0.89"]]
    ... )
    """
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
    """Structure for all plots in the report.
    
    This model represents plot data including metadata and the actual
    plot content (typically as SVG or base64 encoded image data).
    
    Attributes
    ----------
    name : str
        The name/title of the plot
    description : str
        Description of what the plot shows
    image : str
        The plot content, typically as SVG string or base64 encoded image
        
    Examples
    --------
    >>> plot = PlotData(
    ...     name="Feature Importance",
    ...     description="Shows the importance of each feature",
    ...     image="<svg>...</svg>"
    ... )
    """
    name: str
    description: str
    image: str


class FeatureDistribution(RoundedModel):
    """Distribution of a feature across train and test splits.
    
    This model represents the distribution analysis of a single feature
    across different data splits, including both tabular statistics
    and visual plots.
    
    Attributes
    ----------
    ID : str
        Unique identifier for the feature
    tables : List[TableData]
        List of tables containing distribution statistics
    plot : PlotData
        Plot showing the feature distribution
        
    Examples
    --------
    >>> feature_dist = FeatureDistribution(
    ...     ID="feature_1",
    ...     tables=[TableData(...)],
    ...     plot=PlotData(...)
    ... )
    """
    ID: str
    tables: List[TableData]
    plot: PlotData


class DataManager(RoundedModel):
    """Represents a DataManager instance configuration.
    
    This model stores the configuration parameters used for data splitting
    and management in machine learning experiments.
    
    Attributes
    ----------
    ID : str
        Unique identifier for the data manager
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
    n_splits : int
        Number of cross-validation splits
    split_method : str
        Method used for splitting data (e.g., 'random', 'stratified')
    group_column : str
        Column name used for group-based splitting
    stratified : str
        Whether stratification is used ('True' or 'False')
    random_state : int | None
        Random seed for reproducible splits, None if not set
        
    Examples
    --------
    >>> data_mgr = DataManager(
    ...     ID="dm_1",
    ...     test_size=0.2,
    ...     n_splits=5,
    ...     split_method="stratified",
    ...     group_column="group_id",
    ...     stratified="True",
    ...     random_state=42
    ... )
    """
    ID: str
    test_size: float
    n_splits: int
    split_method: str
    group_column: str
    stratified: str
    random_state: int | None


class Navbar(RoundedModel):
    """Data for the navigation bar.
    
    This model contains metadata displayed in the report's navigation bar,
    typically including version information and timestamps.
    
    Attributes
    ----------
    brisk_version : str
        Version of the Brisk library used to generate the report
    timestamp : str
        Timestamp when the report was generated
        
    Examples
    --------
    >>> navbar = Navbar(
    ...     brisk_version="1.0.0",
    ...     timestamp="2024-01-15 10:30:00"
    ... )
    """
    brisk_version: str
    timestamp: str


class ExperimentGroup(RoundedModel):
    """Data for an ExperimentGroup card on the home page.
    
    This model represents a group of related experiments that are displayed
    together on the report's home page. It includes metadata about the group
    and references to datasets and experiments within the group.
    
    Attributes
    ----------
    name : str
        Name of the experiment group
    description : str
        Description of what the experiment group contains
    datasets : List[str]
        List of dataset IDs included in this group
    experiments : List[str]
        List of experiment IDs included in this group
    data_split_scores : Dict[str, List[Tuple[str, str | None, str, str | None]]]
        Best algorithm and score for each data split, keyed by dataset name
    test_scores : Dict[str, TableData]
        Test data scores indexed on dataset name and split number
        
    Examples
    --------
    >>> group = ExperimentGroup(
    ...     name="Classification Experiments",
    ...     description="Binary classification on various datasets",
    ...     datasets=["dataset_1", "dataset_2"],
    ...     experiments=["exp_1", "exp_2"],
    ...     data_split_scores={"dataset_1": [("XTree", "0.95", "0.92", None)]},
    ...     test_scores={"dataset_1": TableData(...)}
    ... )
    """
    name: str
    description: str
    datasets: List[str] = Field(
        default_factory=list, description="List of dataset IDs"
    )
    experiments: List[str] = Field(
        default_factory=list, description="List of experiment IDs"
    )
    data_split_scores: Dict[
        str, List[Tuple[str, str | None, str, str | None]]
    ] = Field(
        default_factory=dict,
        description="Best algorithm and score for each data split."
    )
    test_scores: Dict[str, TableData] = Field(
        default_factory=dict,
        description="Test data scores indexed on dataset name and split number."
    )


class Experiment(RoundedModel):
    """Results of a single machine learning experiment.
    
    This model represents the complete results of a single experiment,
    including algorithm information, hyperparameters, and all associated
    tables and plots.
    
    Attributes
    ----------
    ID : str
        Unique identifier for the experiment
    dataset : str
        Name of the dataset used in this experiment
    algorithm : List[str]
        Display names of algorithms used in the experiment
    tuned_params : Dict[str, Any]
        Tuned hyperparameter names and values
    hyperparam_grid : Dict[str, Any]
        Hyperparameter grid used for tuning
    tables : List[TableData]
        List of tables containing experiment results
    plots : List[PlotData]
        List of plots visualizing experiment results
        
    Examples
    --------
    >>> experiment = Experiment(
    ...     ID="exp_1",
    ...     dataset="iris",
    ...     algorithm=["Random Forest", "SVM"],
    ...     tuned_params={"n_estimators": 100, "max_depth": 10},
    ...     hyperparam_grid={"n_estimators": [50, 100, 200]},
    ...     tables=[TableData(...)],
    ...     plots=[PlotData(...)]
    ... )
    """
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
    """Represents a dataset within an ExperimentGroup.
    
    This model contains comprehensive information about a dataset including
    its splits, feature information, and various statistical analyses.
    
    Attributes
    ----------
    ID : str
        Unique identifier for the dataset
    splits : List[str]
        List of data split indexes (e.g., ["0", "1", "2"])
    split_sizes : Dict[str, Dict[str, int]]
        Size of dataset and train/test split for each split
    split_target_stats : Dict[str, Dict[str, Union[float, dict]]]
        Target feature statistics per split
    split_corr_matrices : Dict[str, PlotData]
        Correlation matrix plots per split
    data_manager_id : str
        ID of the associated DataManager
    features : List[str]
        List of feature names in the dataset
    split_feature_distributions : Dict[str, List[FeatureDistribution]]
        Feature distribution analyses per split
        
    Examples
    --------
    >>> dataset = Dataset(
    ...     ID="dataset_1",
    ...     splits=["0", "1", "2"],
    ...     split_sizes={"0": {"total": 1000, "train": 800, "test": 200}},
    ...     split_target_stats={"0": {"mean": 0.5, "std": 0.1}},
    ...     split_corr_matrices={"0": PlotData(...)},
    ...     data_manager_id="dm_1",
    ...     features=["feature_1", "feature_2"],
    ...     split_feature_distributions={"0": [FeatureDistribution(...)]}
    ... )
    """
    ID: str
    splits: List[str] = Field(
        default_factory=list, description="List of data split indexes"
    )
    split_sizes: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Size of dataset and train/test split"
    )
    split_target_stats: Dict[str, Dict[str, Union[float, dict]]] = Field(
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
    """Represents the entire machine learning report.
    
    This is the root model that contains all data for a complete machine
    learning report, including navigation information, datasets, experiments,
    and data managers.
    
    Attributes
    ----------
    navbar : Navbar
        Navigation bar data with version and timestamp information
    datasets : Dict[str, Dataset]
        Map of dataset IDs to Dataset instances
    experiments : Dict[str, Experiment]
        Map of experiment IDs to Experiment instances
    experiment_groups : List[ExperimentGroup]
        List of experiment groups for organizing related experiments
    data_managers : Dict[str, DataManager]
        Map of data manager IDs to DataManager instances
        
    Examples
    --------
    >>> report = ReportData(
    ...     navbar=Navbar(brisk_version="1.0.0", timestamp="2024-01-15"),
    ...     datasets={"dataset_1": Dataset(...)},
    ...     experiments={"exp_1": Experiment(...)},
    ...     experiment_groups=[ExperimentGroup(...)],
    ...     data_managers={"dm_1": DataManager(...)}
    ... )
    """
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
