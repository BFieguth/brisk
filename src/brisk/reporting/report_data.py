"""Define Pydantic models to represent the results of model runs."""
from typing import List, Optional, Tuple, Dict

from pydantic import BaseModel, Field

class TableData(BaseModel):
    """Structure for all tables in the report."""
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


class PlotData(BaseModel):
    """Structure for all plots in the report."""
    name: str
    description: str
    image: str


class FeatureDistribution(BaseModel):
    """Distribution of a feature across train and test splits."""
    ID: str
    table: TableData
    plot: PlotData


class DataManager(BaseModel):
    """Represents a DataManager instance."""
    ID: str
    n_splits: str
    split_method: str


class Navbar(BaseModel):
    """Data for the navbar."""
    brisk_version: str
    timestamp: str


class ExperimentGroup(BaseModel):
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


class Experiment(BaseModel):
    """Results of a single experiment."""
    ID: str
    dataset: str
    algorithm: List[str] = Field(
        default_factory=list,
        description="Display names of algorithms in experiment"
    )
    tuned_params: Dict[str, str] = Field(
        default_factory=dict,
        description="Tuned hyperparameter names and values"
    )
    hyperparam_grid: Dict[str, str] = Field(
        default_factory=dict, description="Hyperparameter grid used for tuning"
    )
    tables: List[TableData] = Field(
        default_factory=list, description="List of tables for this experiment"
    )
    plots: List[PlotData] = Field(
        default_factory=list, description="List of plots for this experiment"
    )


class Dataset(BaseModel):
    """Represents a dataset within an ExperimentGroup."""
    ID: str
    splits: List[str] = Field(
        default_factory=list, description="List of data split indexes"
    )
    split_sizes: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Size of dataset and train/test split"
    )
    split_target_stats: Dict[str, Dict[str, str]] = Field(
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


class ReportData(BaseModel):
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
