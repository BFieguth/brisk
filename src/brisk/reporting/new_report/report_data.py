from typing import List, Optional, Tuple, Dict
from datetime import datetime

from pydantic import BaseModel, Field

from brisk.version import __version__

class NavigationLink(BaseModel):
    """
    Specify what data to use for the new page and the type of page to render.
    """
    page_type: str
    page_data: str


class TableData(BaseModel):
    """
    Structure for all tables in the report.
    """
    columns: List[str] = Field(
        ..., description="List of column headers"
    )
    rows: List[List[str]] = Field(
        ..., description="List of rows, each row is a list of cell values"
    )
    description: Optional[str] = Field(
        None, description="Optional description text displayed below the table"
    )


class Navbar(BaseModel):
    """
    Data for the navbar.
    """
    brisk_version: str
    timestamp: str


class ExperimentGroupCard(BaseModel):
    """
    Data for an experiment group card on the home page.
    """
    group_name: str
    dataset_names: List[str] = Field(
        default_factory=list, 
        description="List of dataset names used for group"
    )
    description: str
    experiments: List[Tuple[str, str]] = Field(
        default_factory=list, 
        description="List of tuples with experiment name and link to experiment page."
    )
    data_splits_link: Dict[str, str] = Field(
        default_factory=dict, 
        description="Dict mapping dataset name to dataset page links"
    )
    data_split_scores: Dict[str, List[Tuple[str, str, str]]] = Field(
        default_factory=dict,
        description="Dict of best algorithm and score for each datasplit indexed on dataset name."
    )


class ReportData(BaseModel):
    """
    Contains all data needed for the report.
    """
    navbar: Navbar
    tables: List[TableData] = Field(
        default_factory=list,
        description="Test set scores for the home page"
    )
    experiment_group_cards: Dict[str, ExperimentGroupCard] = Field(
        default_factory=dict,
        description="Data for experiment group cards indexed on experiment group name"
    )


navbar = Navbar(
    brisk_version=f"Version: {__version__}",
    timestamp=f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

home_summary_table = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Random Forest", "0.95", "0.94", "0.96"],
        ["XGBoost", "0.97", "0.96", "0.98"],
        ["Logistic Regression", "0.89", "0.88", "0.90"],
        ["SVM", "0.92", "0.91", "0.93"],
        ["Neural Network", "0.96", "0.95", "0.97"]
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)

experiment_group_1 = ExperimentGroupCard(
    group_name="regression_test",
    dataset_names=["dataset", "data2", "another", "yet_another"],
    description="Fake data provided to develop the ExperimentGroup card.",
    experiments=[
        ("regression_test_linear", "experiment_page/regression_test_linear"),
        ("regression_test_ridge", "experiment_page/regression_test_ridge"),
        ("regression_test_lasso", "experiment_page/regression_test_lasso"),
    ],
    data_splits_link={
        "dataset": "dataset_page/dataset",
        "data2": "dataset_page/data2"
    },
    data_split_scores={
        "dataset": [
            ("Split 0", "Extra Trees", "0.84"),
            ("Split 1", "Random Forest", "0.82"),
            ("Split 2", "Extra Trees", "0.83")
        ],
        "data2": [
            ("Split 0", "Linear", "0.56"),
            ("Split 1", "Ridge", "0.73"),
            ("Split 2", "LASSO", "0.30")
        ]
    }
)

experiment_group_2 = ExperimentGroupCard(
    group_name="regression2",
    dataset_names=["dataset", "moredata"],
    description="Animate a second card",
    experiments=[
        ("regression_test_xtree", "experiment_page/regression_test_xtree"),
        ("regression_test_rf", "experiment_page/regression_test_rf"),
    ],
    data_splits_link={
        "dataset": "dataset_page/dataset",
        "moredata": "dataset_page/moredata"
    },
    data_split_scores={
        "dataset": [
            ("Split 0", "Extra Trees", "0.50"),
            ("Split 1", "Random Forest", "0.22"),
        ],
        "moredata": [
            ("Split 0", "Linear", "0.46"),
            ("Split 1", "Ridge", "0.36"),
        ]
    }
)

experiment_group_3 = ExperimentGroupCard(
    group_name="regression3",
    dataset_names=["dataset", "data3.0"],
    description="Animate a third card",
    experiments=[
        ("regression_test_xtree", "experiment_page/regression_test_xtree"),
        ("regression_test_rf", "experiment_page/regression_test_rf"),
    ],
    data_splits_link={
        "dataset": "dataset_page/dataset",
        "moredata": "dataset_page/moredata"
    },
    data_split_scores={
        "dataset": [
            ("Split 0", "Extra Trees", "0.54"),
            ("Split 1", "Random Forest", "0.32"),
        ],
        "moredata": [
            ("Split 0", "Linear", "0.92"),
            ("Split 1", "Ridge", "0.31"),
        ]
    }
)

experiment_group_4 = ExperimentGroupCard(
    group_name="regression4",
    dataset_names=["dataset", "data4.0"],
    description="Animate a third card",
    experiments=[
        ("regression_test_xtree", "experiment_page/regression_test_xtree"),
        ("regression_test_rf", "experiment_page/regression_test_rf"),
    ],
    data_splits_link={
        "dataset": "dataset_page/dataset",
        "moredata": "dataset_page/moredata"
    },
    data_split_scores={
        "dataset": [
            ("Split 0", "Extra Trees", "0.54"),
            ("Split 1", "Random Forest", "0.32"),
        ],
        "moredata": [
            ("Split 0", "Linear", "0.92"),
            ("Split 1", "Ridge", "0.31"),
        ]
    }
)

report_data = ReportData(
    navbar=navbar,
    tables=[home_summary_table],
    experiment_group_cards={
        "regression_test": experiment_group_1,
        "regression2": experiment_group_2,
        "regression3": experiment_group_3,
        "regression4": experiment_group_4
    }
)
