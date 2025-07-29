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
    data_split_scores: Dict[str, List[Tuple[str, str, str, str]]] = Field(
        default_factory=dict,
        description="Dict of best algorithm and score for each datasplit indexed on dataset name."
    )


class ReportData(BaseModel):
    """
    Contains all data needed for the report.
    """
    navbar: Navbar
    tables: Dict[str, TableData] = Field(
        default_factory=dict,
        description="Test set scores for the home page, indexed on ExperimentGroup, dataset and data split"
    )
    experiment_group_cards: Dict[str, ExperimentGroupCard] = Field(
        default_factory=dict,
        description="Data for experiment group cards indexed on experiment group name"
    )


navbar = Navbar(
    brisk_version=f"Version: {__version__}",
    timestamp=f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

regression_test = ExperimentGroupCard(
    group_name="regression_test",
    dataset_names=["dataset", "data2"],
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
            ("Split 0", "Linear", "0.26", "CCC"),
            ("Split 1", "Linear", "0.03", "CCC"),
            ("Split 2", "Ridge", "0.10", "CCC")
        ],
        "data2": [
            ("Split 0", "Linear", "0.56", "CCC"),
            ("Split 1", "Ridge", "0.73", "CCC"),
            ("Split 2", "LASSO", "0.30", "CCC")
        ]
    }
)

regression_test_dataset_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Linear", "0.95", "0.94", "0.96"],
        ["Ridge", "0.97", "0.96", "0.98"],
        ["LASSO", "0.89", "0.88", "0.90"],
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)

regression_test_dataset_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Linear", "0.2", "0.2", "0.2"],
        ["Ridge", "0.97", "0.22", "0.22"],
        ["LASSO", "0.89", "0.88", "0.90"],
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)

regression_test_dataset_split2 = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Linear", "0.3", "0.3", "0.3"],
        ["Ridge", "0.33", "0.33", "0.33"],
        ["LASSO", "0.89", "0.88", "0.90"],
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)

regression_test_data2_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Linear", "0.10", "0.11", "0.111"],
        ["Ridge", "0.97", "0.96", "0.98"],
        ["LASSO", "0.89", "0.88", "0.90"],
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)

regression_test_data2_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Linear", "0.25", "0.02", "0.29"],
        ["Ridge", "0.97", "0.22", "0.22"],
        ["LASSO", "0.89", "0.88", "0.90"],
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)

regression_test_data2_split2 = TableData(
    columns=["Model", "MAE", "MSE", "CCC"],
    rows=[
        ["Linear", "0.33", "0.323", "0.313"],
        ["Ridge", "0.33", "0.33", "0.33"],
        ["LASSO", "0.89", "0.88", "0.90"],
    ],
    description="Test set scores after hyperparameter tuning for selected data split."
)


# ============= EXPERIMENT GROUP 2: Tree-Based Methods =============
experiment_group_2 = ExperimentGroupCard(
    group_name="regression2",
    dataset_names=["clinical_data", "sensor_data"],
    description="Tree-based regression models for clinical and sensor data analysis.",
    experiments=[
        ("tree_extra_trees", "experiment_page/tree_extra_trees"),
        ("tree_random_forest", "experiment_page/tree_random_forest"),
        ("tree_xgboost", "experiment_page/tree_xgboost"),
    ],
    data_splits_link={
        "clinical_data": "dataset_page/clinical_data",
        "sensor_data": "dataset_page/sensor_data"
    },
    data_split_scores={
        "clinical_data": [
            ("Split 0", "XGBoost", "0.78", "CCC"),
            ("Split 1", "Random Forest", "0.72", "CCC"),
        ],
        "sensor_data": [
            ("Split 0", "Extra Trees", "0.85", "CCC"),
            ("Split 1", "XGBoost", "0.81", "CCC"),
        ]
    }
)

# Tree-based model table data
regression2_clinical_data_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "R²"],
    rows=[
        ["Extra Trees", "0.45", "0.42", "0.76", "0.84"],
        ["Random Forest", "0.48", "0.46", "0.72", "0.82"],
        ["XGBoost", "0.41", "0.38", "0.78", "0.87"],
        ["SVM", "0.52", "0.51", "0.68", "0.79"],
    ],
    description="Clinical data performance metrics for tree-based regression models."
)

regression2_clinical_data_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "R²"],
    rows=[
        ["Extra Trees", "0.38", "0.35", "0.70", "0.81"],
        ["Random Forest", "0.42", "0.40", "0.72", "0.83"],
        ["XGBoost", "0.44", "0.42", "0.69", "0.80"],
        ["SVM", "0.49", "0.48", "0.65", "0.76"],
    ],
    description="Clinical data performance metrics for tree-based regression models."
)

regression2_sensor_data_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "R²"],
    rows=[
        ["Extra Trees", "0.32", "0.28", "0.85", "0.91"],
        ["Random Forest", "0.35", "0.31", "0.82", "0.89"],
        ["XGBoost", "0.34", "0.30", "0.81", "0.88"],
        ["SVM", "0.41", "0.38", "0.76", "0.84"],
    ],
    description="Sensor data performance metrics for tree-based regression models."
)

regression2_sensor_data_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "R²"],
    rows=[
        ["Extra Trees", "0.36", "0.33", "0.79", "0.86"],
        ["Random Forest", "0.39", "0.37", "0.78", "0.85"],
        ["XGBoost", "0.33", "0.29", "0.81", "0.88"],
        ["SVM", "0.44", "0.42", "0.73", "0.81"],
    ],
    description="Sensor data performance metrics for tree-based regression models."
)

# ============= EXPERIMENT GROUP 3: Deep Learning Methods =============
experiment_group_3 = ExperimentGroupCard(
    group_name="regression3",
    dataset_names=["imaging_data", "genomic_data", "multimodal_data"],
    description="Deep learning approaches for complex multi-modal regression tasks.",
    experiments=[
        ("dl_cnn_model", "experiment_page/dl_cnn_model"),
        ("dl_resnet_model", "experiment_page/dl_resnet_model"),
        ("dl_transformer", "experiment_page/dl_transformer"),
        ("dl_multimodal", "experiment_page/dl_multimodal"),
    ],
    data_splits_link={
        "imaging_data": "dataset_page/imaging_data",
        "genomic_data": "dataset_page/genomic_data",
        "multimodal_data": "dataset_page/multimodal_data"
    },
    data_split_scores={
        "imaging_data": [
            ("Split 0", "ResNet", "0.89", "CCC"),
            ("Split 1", "CNN", "0.84", "CCC"),
            ("Split 2", "Transformer", "0.87", "CCC"),
        ],
        "genomic_data": [
            ("Split 0", "Transformer", "0.76", "Pearson"),
            ("Split 1", "LSTM", "0.71", "Pearson"),
        ],
        "multimodal_data": [
            ("Split 0", "Multimodal-Net", "0.92", "F1"),
            ("Split 1", "CNN+LSTM", "0.88", "F1"),
        ]
    }
)

# Deep learning model table data
regression3_imaging_data_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "LPIPS"],
    rows=[
        ["CNN", "0.28", "0.24", "0.86", "0.12"],
        ["ResNet-50", "0.22", "0.18", "0.89", "0.09"],
        ["Vision Transformer", "0.25", "0.21", "0.87", "0.10"],
        ["EfficientNet", "0.24", "0.20", "0.88", "0.08"],
        ["Linear Baseline", "0.45", "0.42", "0.72", "0.25"],
    ],
    description="Image-based regression performance using deep learning architectures."
)

regression3_imaging_data_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "LPIPS"],
    rows=[
        ["CNN", "0.31", "0.28", "0.84", "0.14"],
        ["ResNet-50", "0.26", "0.23", "0.87", "0.11"],
        ["Vision Transformer", "0.28", "0.25", "0.85", "0.12"],
        ["EfficientNet", "0.27", "0.24", "0.86", "0.10"],
        ["Linear Baseline", "0.48", "0.46", "0.69", "0.28"],
    ],
    description="Image-based regression performance using deep learning architectures."
)

regression3_imaging_data_split2 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "LPIPS"],
    rows=[
        ["CNN", "0.33", "0.30", "0.82", "0.15"],
        ["ResNet-50", "0.24", "0.20", "0.87", "0.09"],
        ["Vision Transformer", "0.29", "0.26", "0.84", "0.13"],
        ["EfficientNet", "0.25", "0.22", "0.86", "0.11"],
        ["Linear Baseline", "0.51", "0.49", "0.66", "0.31"],
    ],
    description="Image-based regression performance using deep learning architectures."
)

regression3_genomic_data_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "Pearson"],
    rows=[
        ["LSTM", "0.42", "0.39", "0.74", "0.79"],
        ["Transformer", "0.38", "0.34", "0.76", "0.82"],
        ["1D-CNN", "0.45", "0.42", "0.71", "0.76"],
        ["Linear", "0.58", "0.56", "0.62", "0.68"],
    ],
    description="Genomic sequence regression using sequence-based deep learning models."
)

regression3_genomic_data_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "Pearson"],
    rows=[
        ["LSTM", "0.46", "0.43", "0.71", "0.75"],
        ["Transformer", "0.41", "0.37", "0.74", "0.79"],
        ["1D-CNN", "0.48", "0.46", "0.68", "0.73"],
        ["Linear", "0.61", "0.59", "0.59", "0.65"],
    ],
    description="Genomic sequence regression using sequence-based deep learning models."
)

regression3_multimodal_data_split0 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "F1"],
    rows=[
        ["Multimodal-Net", "0.19", "0.15", "0.92", "0.89"],
        ["CNN+LSTM", "0.23", "0.19", "0.88", "0.85"],
        ["Late Fusion", "0.27", "0.24", "0.84", "0.81"],
        ["Early Fusion", "0.25", "0.22", "0.86", "0.83"],
        ["Single Modal", "0.34", "0.31", "0.79", "0.76"],
    ],
    description="Multi-modal regression combining imaging, text, and sensor data."
)

regression3_multimodal_data_split1 = TableData(
    columns=["Model", "MAE", "MSE", "CCC", "F1"],
    rows=[
        ["Multimodal-Net", "0.21", "0.17", "0.90", "0.87"],
        ["CNN+LSTM", "0.25", "0.21", "0.86", "0.83"],
        ["Late Fusion", "0.29", "0.26", "0.82", "0.79"],
        ["Early Fusion", "0.27", "0.24", "0.84", "0.81"],
        ["Single Modal", "0.36", "0.33", "0.77", "0.74"],
    ],
    description="Multi-modal regression combining imaging, text, and sensor data."
)

# ============= EXPERIMENT GROUP 4: Time Series Methods =============
experiment_group_4 = ExperimentGroupCard(
    group_name="regression4",
    dataset_names=["timeseries_data"],
    description="Specialized time series regression models for temporal prediction tasks.",
    experiments=[
        ("ts_lstm_model", "experiment_page/ts_lstm_model"),
        ("ts_transformer", "experiment_page/ts_transformer"),
        ("ts_prophet", "experiment_page/ts_prophet"),
        ("ts_arima", "experiment_page/ts_arima"),
    ],
    data_splits_link={
        "timeseries_data": "dataset_page/timeseries_data"
    },
    data_split_scores={
        "timeseries_data": [
            ("Split 0", "LSTM", "0.084", "MAPE"),
            ("Split 1", "Transformer", "0.074", "MAPE"),
            ("Split 2", "Prophet", "0.089", "MAPE"),
            ("Split 3", "LSTM", "0.087", "MAPE"),
        ]
    }
)

# Time series model table data
regression4_timeseries_data_split0 = TableData(
    columns=["Model", "MAE", "MSE", "MAPE", "SMAPE"],
    rows=[
        ["LSTM", "2.45", "8.21", "0.084", "0.078"],
        ["Transformer", "2.68", "9.15", "0.091", "0.085"],
        ["Prophet", "3.12", "11.8", "0.105", "0.098"],
        ["ARIMA", "3.85", "16.2", "0.128", "0.121"],
        ["Linear Trend", "4.21", "19.7", "0.142", "0.136"],
    ],
    description="Time series forecasting performance on financial data."
)

regression4_timeseries_data_split1 = TableData(
    columns=["Model", "MAE", "MSE", "MAPE", "SMAPE"],
    rows=[
        ["LSTM", "2.31", "7.89", "0.079", "0.072"],
        ["Transformer", "2.18", "7.24", "0.074", "0.068"],
        ["Prophet", "2.89", "10.4", "0.098", "0.091"],
        ["ARIMA", "3.67", "15.1", "0.124", "0.117"],
        ["Linear Trend", "4.05", "18.9", "0.138", "0.131"],
    ],
    description="Time series forecasting performance on weather data."
)

regression4_timeseries_data_split2 = TableData(
    columns=["Model", "MAE", "MSE", "MAPE", "SMAPE"],
    rows=[
        ["LSTM", "2.78", "9.42", "0.095", "0.089"],
        ["Transformer", "2.92", "10.1", "0.099", "0.093"],
        ["Prophet", "2.65", "8.78", "0.089", "0.083"],
        ["ARIMA", "3.21", "12.4", "0.109", "0.102"],
        ["Linear Trend", "3.98", "17.8", "0.135", "0.128"],
    ],
    description="Time series forecasting performance on energy consumption data."
)

regression4_timeseries_data_split3 = TableData(
    columns=["Model", "MAE", "MSE", "MAPE", "SMAPE"],
    rows=[
        ["LSTM", "2.52", "8.67", "0.087", "0.081"],
        ["Transformer", "2.74", "9.34", "0.093", "0.087"],
        ["Prophet", "3.05", "11.2", "0.103", "0.096"],
        ["ARIMA", "3.58", "14.7", "0.121", "0.114"],
        ["Linear Trend", "4.15", "19.3", "0.140", "0.133"],
    ],
    description="Time series forecasting performance on IoT sensor data."
)

report_data = ReportData(
    navbar=navbar,
    tables={
        "regression_test_dataset_split0": regression_test_dataset_split0,
        "regression_test_dataset_split1": regression_test_dataset_split1,
        "regression_test_dataset_split2": regression_test_dataset_split2,
        "regression_test_data2_split0": regression_test_data2_split0,
        "regression_test_data2_split1": regression_test_data2_split1,
        "regression_test_data2_split2": regression_test_data2_split2,
        
        "regression2_clinical_data_split0": regression2_clinical_data_split0,
        "regression2_clinical_data_split1": regression2_clinical_data_split1,
        "regression2_sensor_data_split0": regression2_sensor_data_split0,
        "regression2_sensor_data_split1": regression2_sensor_data_split1,
        
        "regression3_imaging_data_split0": regression3_imaging_data_split0,
        "regression3_imaging_data_split1": regression3_imaging_data_split1,
        "regression3_imaging_data_split2": regression3_imaging_data_split2,
        "regression3_genomic_data_split0": regression3_genomic_data_split0,
        "regression3_genomic_data_split1": regression3_genomic_data_split1,
        "regression3_multimodal_data_split0": regression3_multimodal_data_split0,
        "regression3_multimodal_data_split1": regression3_multimodal_data_split1,
        
        "regression4_timeseries_data_split0": regression4_timeseries_data_split0,
        "regression4_timeseries_data_split1": regression4_timeseries_data_split1,
        "regression4_timeseries_data_split2": regression4_timeseries_data_split2,
        "regression4_timeseries_data_split3": regression4_timeseries_data_split3,
    },
    experiment_group_cards={
        "regression_test": regression_test,
        "regression2": experiment_group_2,
        "regression3": experiment_group_3,
        "regression4": experiment_group_4
    }
)
