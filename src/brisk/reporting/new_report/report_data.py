from typing import List, Optional, Tuple, Dict
from datetime import datetime

from pydantic import BaseModel, Field

from brisk.version import __version__

from pathlib import Path

def load_svg_file(file_path: str) -> str:
    """Load SVG content directly from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: SVG file not found: {file_path}")
        return '<svg><text>Image not found</text></svg>'

# Load SVG files directly (much better than Base64!)
base_path = Path(__file__).parent.parent.parent.parent.parent / "dev" / "test_images"

matplotlib_simple_line_svg = load_svg_file(base_path / "simple" / "matplotlib_simple_line.svg")
plotnine_simple_scatter_svg = load_svg_file(base_path / "simple" / "plotnine_simple_scatter.svg")
comparison_scatter_plotnine_svg = load_svg_file(base_path / "plotnine" / "comparison_scatter_plotnine.svg")
comparison_scatter_matplotlib_svg = load_svg_file(base_path / "matplotlib" / "comparison_scatter_matplotlib.svg")


class TableData(BaseModel):
    """
    Structure for all tables in the report.
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


class PlotData(BaseModel):
    name: str
    description: str
    image: str


class FeatureDistribution(BaseModel):
    ID: str # dataset, split #, feature name
    table: TableData
    plot: PlotData


class DataManager(BaseModel):
    ID: str
    n_splits: str
    split_method: str


class Navbar(BaseModel):
    """
    Data for the navbar.
    """
    brisk_version: str
    timestamp: str


class ExperimentGroup(BaseModel):
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
        description="Dict of best algorithm and score for each datasplit indexed on dataset name."
    )
    test_scores: Dict[str, TableData] = Field(
        default_factory=dict, 
        description="Use dataset name and split number to index test data scores."
    )


class Experiment(BaseModel):
    ID: str
    # group: str # Is this needed if ID is unique?
    dataset: str # Dataset ID
    algorithm: List[str] = Field(
        default_factory=list, description="Display names of algorithms in experiment"
    )
    tuned_params: Dict[str, str] = Field(
        default_factory=dict, description="Tuned hyperparameter names and values"
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
    ID: str
    splits: List[str] = Field(
        default_factory=list, description="List of data split indexs"
    )
    size: Tuple[str, str] = Field(
        default_factory=tuple, description="Number of rows, number of columns"
    )
    target_stats: Dict[str, str] = Field(
        default_factory=dict, description="Target feature stats, stat name: value"
    )
    data_manager_id: str
    corr_matrix: PlotData
    features: List[str] = Field(
        default_factory=list, description="List of feature names"
    )
    feature_distributions: List[FeatureDistribution] = Field(
        default_factory=list, description="List of feature distributions"
    )


class ReportData(BaseModel):
    navbar: Navbar
    datasets: Dict[str, Dataset] = Field(
        default_factory=dict, description="List of datasets"
    )
    experiments: Dict[str, Experiment] = Field(
        default_factory=dict, description="List of experiments"
    )
    experiment_groups: List[ExperimentGroup] = Field(
        default_factory=list, description="List of experiment groups"
    )
    data_managers: Dict[str, DataManager] = Field(
        default_factory=dict, description="Map IDs to DataManager instances"
    )



# Define available plots for cycling through
available_plots = [
    PlotData(name="Feature Distribution", description="Distribution of feature values", image=matplotlib_simple_line_svg),
    PlotData(name="Feature Scatter", description="Feature scatter plot", image=plotnine_simple_scatter_svg),
    PlotData(name="Feature Comparison", description="Feature comparison plot", image=comparison_scatter_plotnine_svg),
    PlotData(name="Feature Analysis", description="Feature analysis visualization", image=comparison_scatter_matplotlib_svg)
]

def create_feature_distribution(feature_name: str, plot_index: int) -> FeatureDistribution:
    """Create a FeatureDistribution instance for a given feature."""
    # Create sample statistics table for the feature
    table = TableData(
        name=f"{feature_name} Statistics",
        description=f"Statistical summary for feature {feature_name}",
        columns=["Statistic", "Value"],
        rows=[
            ["Mean", f"{(hash(feature_name) % 100 + 50):.2f}"],
            ["Std Dev", f"{(hash(feature_name) % 20 + 5):.2f}"],
            ["Min", f"{(hash(feature_name) % 10):.2f}"],
            ["Max", f"{(hash(feature_name) % 200 + 100):.2f}"],
            ["25th Percentile", f"{(hash(feature_name) % 40 + 20):.2f}"],
            ["75th Percentile", f"{(hash(feature_name) % 60 + 80):.2f}"]
        ]
    )
    
    # Cycle through available plots
    plot = available_plots[plot_index % len(available_plots)]
    
    return FeatureDistribution(
        ID=feature_name,
        table=table,
        plot=plot
    )

def create_data_manager(dm_id: str, n_splits: int):
    """Create a DataManager instance for a given ID."""
    return DataManager(
        ID=dm_id,
        n_splits=str(n_splits),
        split_method="shuffle"
    )

# Create DataManager instances
data_managers_dict = {
    "dm_housing": create_data_manager("dm_housing", 3),
    "dm_medical": create_data_manager("dm_medical", 3),
    "dm_clinical": create_data_manager("dm_clinical", 2),
    "dm_sensor": create_data_manager("dm_sensor", 2),
    "dm_imaging": create_data_manager("dm_imaging", 3),
    "dm_genomic": create_data_manager("dm_genomic", 2),
    "dm_multimodal": create_data_manager("dm_multimodal", 2),
    "dm_timeseries": create_data_manager("dm_timeseries", 4)
}



# Create navbar
navbar = Navbar(
    brisk_version=f"Version: {__version__}",
    timestamp=f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# ============= EXPERIMENT GROUP 1: Linear Methods =============
# Create datasets for Linear Methods group
dataset_1 = Dataset(
    ID="Linear Methods_housing_data",
    splits=["split_0", "split_1", "split_2"],
    size=("506", "13"),
    target_stats={"mean": "22.53", "std": "9.20", "min": "5.00", "max": "50.00"},
    data_manager_id="dm_housing",
    corr_matrix=PlotData(
        name="Correlation Matrix - Housing Data",
        description="Feature correlation matrix for housing dataset",
        image=matplotlib_simple_line_svg
    ),
    features=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])]
)

dataset_2 = Dataset(
    ID="Linear Methods_medical_data",
    splits=["split_0", "split_1", "split_2"],
    size=("442", "10"),
    target_stats={"mean": "152.13", "std": "77.09", "min": "25.00", "max": "346.00"},
    data_manager_id="dm_medical",
    corr_matrix=PlotData(
        name="Correlation Matrix - Medical Data",
        description="Feature correlation matrix for medical dataset",
        image=plotnine_simple_scatter_svg
    ),
    features=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"])]
)

# Create experiments for Linear Methods group
linear_experiment = Experiment(
    ID="linear_regression",
    dataset="Linear Methods_housing_data",
    algorithm=["Linear Regression", "Second Algo"],
    tuned_params={"fit_intercept": "True", "normalize": "False"},
    hyperparam_grid={"fit_intercept": "[True, False]", "normalize": "[True, False]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for linear regression on housing data",
            columns=["Split", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Split 0", "3.45", "24.8", "0.67", "0.82"],
                ["Split 1", "3.12", "21.3", "0.72", "0.85"], 
                ["Split 2", "3.78", "28.1", "0.63", "0.79"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during training for linear regression",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_R²", "CV_CCC"],
            rows=[
                ["Split 0", "3.21", "22.1", "0.71", "0.84"],
                ["Split 1", "2.95", "19.8", "0.75", "0.87"], 
                ["Split 2", "3.52", "25.9", "0.67", "0.81"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Feature importance coefficients for linear regression",
            image=comparison_scatter_matplotlib_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation performance vs training set size",
            image=matplotlib_simple_line_svg
        )
    ]
)

ridge_experiment = Experiment(
    ID="ridge_regression",
    dataset="Linear Methods_housing_data",
    algorithm=["Ridge Regression"],
    tuned_params={"alpha": "1.0", "fit_intercept": "True"},
    hyperparam_grid={"alpha": "[0.1, 1.0, 10.0]", "fit_intercept": "[True, False]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for ridge regression on housing data",
            columns=["Split", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Split 0", "3.32", "23.1", "0.69", "0.83"],
                ["Split 1", "3.05", "20.8", "0.73", "0.86"],
                ["Split 2", "3.65", "26.9", "0.65", "0.81"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during training for ridge regression",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_R²", "CV_CCC"],
            rows=[
                ["Split 0", "3.08", "20.8", "0.72", "0.85"],
                ["Split 1", "2.89", "19.1", "0.76", "0.88"],
                ["Split 2", "3.42", "24.5", "0.69", "0.83"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Regularization path showing coefficient values vs alpha",
            image=matplotlib_simple_line_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation performance vs training set size",
            image=plotnine_simple_scatter_svg
        )
    ]
)

lasso_experiment = Experiment(
    ID="lasso_regression",
    dataset="Linear Methods_housing_data",
    algorithm=["LASSO Regression"],
    tuned_params={"alpha": "0.1", "fit_intercept": "True"},
    hyperparam_grid={"alpha": "[0.01, 0.1, 1.0]", "fit_intercept": "[True, False]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for LASSO regression on housing data",
            columns=["Split", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Split 0", "3.58", "26.2", "0.65", "0.80"],
                ["Split 1", "3.21", "22.5", "0.70", "0.84"],
                ["Split 2", "3.89", "29.8", "0.61", "0.78"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during training for LASSO regression",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_R²", "CV_CCC"],
            rows=[
                ["Split 0", "3.35", "24.1", "0.68", "0.82"],
                ["Split 1", "3.02", "20.9", "0.73", "0.86"],
                ["Split 2", "3.67", "27.8", "0.64", "0.80"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Features selected by LASSO regression and their coefficients",
            image=plotnine_simple_scatter_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation performance vs training set size",
            image=comparison_scatter_plotnine_svg
        )
    ]
)

experiment_group_1 = ExperimentGroup(
    name="Linear Methods",
    description="Comparison of linear regression methods including Ridge and LASSO regularization",
    datasets=["Linear Methods_housing_data", "Linear Methods_medical_data"],
    experiments=["linear_regression", "ridge_regression", "lasso_regression"],
    data_split_scores={
        "Linear Methods_housing_data": [
            ("Split 0", "Ridge", "0.83", "CCC"),
            ("Split 1", "Ridge", "0.86", "CCC"),
            ("Split 2", "Ridge", "0.81", "CCC")
        ],
        "Linear Methods_medical_data": [
            ("Split 0", "Linear", "0.78", "CCC"),
            ("Split 1", "LASSO", "0.82", "CCC"),
            ("Split 2", "Ridge", "0.80", "CCC")
        ]
    },
    test_scores={
        "Linear Methods_housing_data_split_0": TableData(
            name="Test Scores - Housing Data Split 0",
            description="Test set performance for all linear methods on housing data split 0",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Linear Regression", "3.45", "24.8", "0.67", "0.82"],
                ["Ridge Regression", "3.32", "23.1", "0.69", "0.83"],
                ["LASSO Regression", "3.58", "26.2", "0.65", "0.80"]
            ]
        ),
        "Linear Methods_housing_data_split_1": TableData(
            name="Test Scores - Housing Data Split 1",
            description="Test set performance for all linear methods on housing data split 1",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Linear Regression", "3.12", "21.3", "0.72", "0.85"],
                ["Ridge Regression", "3.05", "20.8", "0.73", "0.86"],
                ["LASSO Regression", "3.21", "22.5", "0.70", "0.84"]
            ]
        ),
        "Linear Methods_housing_data_split_2": TableData(
            name="Test Scores - Housing Data Split 2",
            description="Test set performance for all linear methods on housing data split 2",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Linear Regression", "3.78", "28.1", "0.63", "0.79"],
                ["Ridge Regression", "3.65", "26.9", "0.65", "0.81"],
                ["LASSO Regression", "3.89", "29.8", "0.61", "0.78"]
            ]
        ),
        "Linear Methods_medical_data_split_0": TableData(
            name="Test Scores - Medical Data Split 0",
            description="Test set performance for all linear methods on medical data split 0",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Linear Regression", "12.5", "189.2", "0.65", "0.78"],
                ["Ridge Regression", "12.1", "182.4", "0.67", "0.80"],
                ["LASSO Regression", "12.8", "195.6", "0.63", "0.76"]
            ]
        ),
        "Linear Methods_medical_data_split_1": TableData(
            name="Test Scores - Medical Data Split 1",
            description="Test set performance for all linear methods on medical data split 1",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Linear Regression", "11.8", "175.3", "0.69", "0.81"],
                ["Ridge Regression", "11.4", "168.9", "0.71", "0.83"],
                ["LASSO Regression", "12.2", "182.7", "0.67", "0.82"]
            ]
        ),
        "Linear Methods_medical_data_split_2": TableData(
            name="Test Scores - Medical Data Split 2",
            description="Test set performance for all linear methods on medical data split 2",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Linear Regression", "13.1", "201.8", "0.62", "0.77"],
                ["Ridge Regression", "12.7", "194.5", "0.64", "0.80"],
                ["LASSO Regression", "13.5", "208.2", "0.60", "0.75"]
            ]
        )
    }
)

# ============= EXPERIMENT GROUP 2: Tree-Based Methods =============
# Create datasets for Tree-Based Methods group
clinical_dataset = Dataset(
    ID="Tree-Based Methods_clinical_data",
    splits=["split_0", "split_1"],
    size=("1000", "25"),
    target_stats={"mean": "68.5", "std": "12.8", "min": "45.0", "max": "95.0"},
    data_manager_id="dm_clinical",
    corr_matrix=PlotData(
        name="Clinical Features Correlation",
        description="Correlation matrix for clinical features",
        image=comparison_scatter_plotnine_svg
    ),
    features=["age", "gender", "bmi", "blood_pressure", "cholesterol", "glucose", "smoking", "exercise"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["age", "gender", "bmi", "blood_pressure", "cholesterol", "glucose", "smoking", "exercise"])]
)

sensor_dataset = Dataset(
    ID="Tree-Based Methods_sensor_data", 
    splits=["split_0", "split_1"],
    size=("2000", "15"),
    target_stats={"mean": "35.2", "std": "8.9", "min": "18.0", "max": "65.0"},
    data_manager_id="dm_sensor",
    corr_matrix=PlotData(
        name="Sensor Data Correlation",
        description="Correlation matrix for sensor measurements",
        image=matplotlib_simple_line_svg
    ),
    features=["temp", "humidity", "pressure", "acceleration_x", "acceleration_y", "acceleration_z"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["temp", "humidity", "pressure", "acceleration_x", "acceleration_y", "acceleration_z"])]
)

# Create experiments for Tree-Based Methods
rf_experiment = Experiment(
    ID="random_forest",
    dataset="Tree-Based Methods_clinical_data",
    algorithm=["Random Forest"],
    tuned_params={"n_estimators": "100", "max_depth": "10", "min_samples_split": "5"},
    hyperparam_grid={"n_estimators": "[50, 100, 200]", "max_depth": "[5, 10, 15]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for Random Forest on clinical data",
            columns=["Split", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Split 0", "2.85", "15.2", "0.82", "0.90"],
                ["Split 1", "2.93", "16.8", "0.80", "0.89"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during training for Random Forest",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_R²", "CV_CCC"],
    rows=[
                ["Split 0", "2.68", "13.9", "0.85", "0.92"],
                ["Split 1", "2.76", "15.2", "0.83", "0.91"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Feature importance ranking from Random Forest",
            image=comparison_scatter_matplotlib_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation performance vs number of trees",
            image=matplotlib_simple_line_svg
        )
    ]
)

xgb_experiment = Experiment(
    ID="xgboost",
    dataset="Tree-Based Methods_clinical_data",
    algorithm=["XGBoost"],
    tuned_params={"n_estimators": "150", "learning_rate": "0.1", "max_depth": "8"},
    hyperparam_grid={"n_estimators": "[100, 150, 200]", "learning_rate": "[0.05, 0.1, 0.2]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for XGBoost on clinical data",
            columns=["Split", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Split 0", "2.78", "14.8", "0.84", "0.91"],
                ["Split 1", "2.89", "16.2", "0.81", "0.90"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during training for XGBoost",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_R²", "CV_CCC"],
    rows=[
                ["Split 0", "2.61", "13.5", "0.86", "0.93"],
                ["Split 1", "2.72", "14.9", "0.84", "0.92"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Feature importance scores from XGBoost model",
            image=plotnine_simple_scatter_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation loss curves during boosting",
            image=comparison_scatter_plotnine_svg
        )
    ]
)

et_experiment = Experiment(
    ID="extra_trees",
    dataset="Tree-Based Methods_sensor_data",
    algorithm=["Extra Trees"],
    tuned_params={"n_estimators": "100", "max_features": "sqrt", "bootstrap": "False"},
    hyperparam_grid={"n_estimators": "[50, 100, 200]", "max_features": "['sqrt', 'log2']"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for Extra Trees on sensor data",
            columns=["Split", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Split 0", "3.12", "18.5", "0.85", "0.92"],
                ["Split 1", "3.28", "19.8", "0.83", "0.91"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during training for Extra Trees",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_R²", "CV_CCC"],
    rows=[
                ["Split 0", "2.94", "16.8", "0.87", "0.94"],
                ["Split 1", "3.09", "18.1", "0.86", "0.93"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Feature importance from Extra Trees ensemble",
            image=comparison_scatter_plotnine_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Performance vs tree depth for Extra Trees",
            image=matplotlib_simple_line_svg
        )
    ]
)

experiment_group_2 = ExperimentGroup(
    name="Tree-Based Methods",
    description="Ensemble methods including Random Forest, XGBoost, and Extra Trees for clinical and sensor data",
    datasets=["Tree-Based Methods_clinical_data", "Tree-Based Methods_sensor_data"],
    experiments=["random_forest", "xgboost", "extra_trees"],
    data_split_scores={
        "Tree-Based Methods_clinical_data": [
            ("Split 0", "XGBoost", "0.91", "CCC"),
            ("Split 1", "XGBoost", "0.90", "CCC")
        ],
        "Tree-Based Methods_sensor_data": [
            ("Split 0", "Extra Trees", "0.92", "CCC"),
            ("Split 1", "Extra Trees", "0.91", "CCC")
        ]
    },
    test_scores={
        "Tree-Based Methods_clinical_data_split_0": TableData(
            name="Test Scores - Clinical Data Split 0",
            description="Test set performance for all tree-based methods on clinical data split 0",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Random Forest", "2.85", "15.2", "0.82", "0.90"],
                ["XGBoost", "2.78", "14.8", "0.84", "0.91"],
                ["Extra Trees", "2.92", "16.1", "0.81", "0.89"]
            ]
        ),
        "Tree-Based Methods_clinical_data_split_1": TableData(
            name="Test Scores - Clinical Data Split 1",
            description="Test set performance for all tree-based methods on clinical data split 1",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Random Forest", "2.93", "16.8", "0.80", "0.89"],
                ["XGBoost", "2.89", "16.2", "0.81", "0.90"],
                ["Extra Trees", "3.01", "17.5", "0.79", "0.88"]
            ]
        ),
        "Tree-Based Methods_sensor_data_split_0": TableData(
            name="Test Scores - Sensor Data Split 0",
            description="Test set performance for all tree-based methods on sensor data split 0",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Random Forest", "3.25", "19.8", "0.84", "0.91"],
                ["XGBoost", "3.18", "19.2", "0.85", "0.92"],
                ["Extra Trees", "3.12", "18.5", "0.85", "0.92"]
            ]
        ),
        "Tree-Based Methods_sensor_data_split_1": TableData(
            name="Test Scores - Sensor Data Split 1",
            description="Test set performance for all tree-based methods on sensor data split 1",
            columns=["Algorithm", "MAE", "MSE", "R²", "CCC"],
            rows=[
                ["Random Forest", "3.38", "21.2", "0.82", "0.90"],
                ["XGBoost", "3.31", "20.6", "0.83", "0.91"],
                ["Extra Trees", "3.28", "19.8", "0.83", "0.91"]
            ]
        )
    }
)

# ============= EXPERIMENT GROUP 3: Deep Learning Methods =============
# Create datasets for Deep Learning Methods
imaging_dataset = Dataset(
    ID="Deep Learning Methods_imaging_data",
    splits=["split_0", "split_1", "split_2"],
    size=("5000", "2048"),
    target_stats={"mean": "0.73", "std": "0.12", "min": "0.35", "max": "0.98"},
    data_manager_id="dm_imaging",
    corr_matrix=PlotData(
        name="Image Feature Correlation",
        description="Principal component correlation for image features",
        image=comparison_scatter_matplotlib_svg
    ),
    features=["PC1", "PC2", "PC3", "texture_contrast", "texture_energy", "edge_density"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["PC1", "PC2", "PC3", "texture_contrast", "texture_energy", "edge_density"])]
)

genomic_dataset = Dataset(
    ID="Deep Learning Methods_genomic_data",
    splits=["split_0", "split_1"],
    size=("3000", "10000"),
    target_stats={"mean": "2.85", "std": "1.45", "min": "0.12", "max": "8.90"},
    data_manager_id="dm_genomic",
    corr_matrix=PlotData(
        name="Gene Expression Correlation",
        description="Top genes correlation matrix",
        image=matplotlib_simple_line_svg
    ),
    features=["GENE_001", "GENE_002", "GENE_003", "pathway_score_1", "pathway_score_2"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["GENE_001", "GENE_002", "GENE_003", "pathway_score_1", "pathway_score_2"])]
)

multimodal_dataset = Dataset(
    ID="Deep Learning Methods_multimodal_data",
    splits=["split_0", "split_1"],
    size=("1500", "5000"),
    target_stats={"mean": "0.68", "std": "0.18", "min": "0.15", "max": "0.95"},
    data_manager_id="dm_multimodal",
    corr_matrix=PlotData(
        name="Multimodal Feature Correlation",
        description="Cross-modal feature correlation analysis",
        image=plotnine_simple_scatter_svg
    ),
    features=["image_features", "text_embeddings", "numerical_features", "categorical_encoded"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["image_features", "text_embeddings", "numerical_features", "categorical_encoded"])]
)

# Create experiments for Deep Learning Methods
cnn_experiment = Experiment(
    ID="cnn_model",
    dataset="Deep Learning Methods_imaging_data",
    algorithm=["CNN"],
    tuned_params={"learning_rate": "0.001", "batch_size": "32", "epochs": "100"},
    hyperparam_grid={"learning_rate": "[0.0001, 0.001, 0.01]", "batch_size": "[16, 32, 64]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for CNN on imaging data",
            columns=["Split", "MAE", "MSE", "CCC", "LPIPS"],
            rows=[
                ["Split 0", "0.082", "0.015", "0.86", "0.09"],
                ["Split 1", "0.089", "0.018", "0.84", "0.11"],
                ["Split 2", "0.095", "0.021", "0.82", "0.13"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during CNN training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_CCC", "CV_LPIPS"],
    rows=[
                ["Split 0", "0.075", "0.013", "0.88", "0.08"],
                ["Split 1", "0.081", "0.016", "0.86", "0.10"],
                ["Split 2", "0.087", "0.019", "0.84", "0.12"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Gradient-based feature attribution maps for CNN",
            image=comparison_scatter_plotnine_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation loss curves during CNN training",
            image=matplotlib_simple_line_svg
        )
    ]
)

resnet_experiment = Experiment(
    ID="resnet_model",
    dataset="Deep Learning Methods_imaging_data", 
    algorithm=["ResNet"],
    tuned_params={"learning_rate": "0.0005", "weight_decay": "0.001", "epochs": "150"},
    hyperparam_grid={"learning_rate": "[0.0001, 0.0005, 0.001]", "weight_decay": "[0.0001, 0.001]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for ResNet on imaging data",
            columns=["Split", "MAE", "MSE", "CCC", "LPIPS"],
            rows=[
                ["Split 0", "0.075", "0.012", "0.89", "0.07"],
                ["Split 1", "0.081", "0.015", "0.87", "0.08"],
                ["Split 2", "0.088", "0.018", "0.85", "0.10"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during ResNet training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_CCC", "CV_LPIPS"],
    rows=[
                ["Split 0", "0.068", "0.010", "0.91", "0.06"],
                ["Split 1", "0.074", "0.013", "0.89", "0.07"],
                ["Split 2", "0.081", "0.016", "0.87", "0.09"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Visual attention maps and gradient-based importance from ResNet",
            image=comparison_scatter_matplotlib_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation metrics over epochs for ResNet",
            image=plotnine_simple_scatter_svg
        )
    ]
)

transformer_experiment = Experiment(
    ID="transformer_model",
    dataset="Deep Learning Methods_genomic_data",
    algorithm=["Transformer"],
    tuned_params={"num_heads": "8", "num_layers": "6", "learning_rate": "0.0001"},
    hyperparam_grid={"num_heads": "[4, 8, 12]", "num_layers": "[4, 6, 8]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for Transformer on genomic data",
            columns=["Split", "MAE", "MSE", "CCC", "Pearson"],
            rows=[
                ["Split 0", "0.68", "0.58", "0.76", "0.82"],
                ["Split 1", "0.72", "0.63", "0.74", "0.79"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during Transformer training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_CCC", "CV_Pearson"],
    rows=[
                ["Split 0", "0.62", "0.51", "0.79", "0.85"],
                ["Split 1", "0.66", "0.57", "0.77", "0.82"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Attention weights and token importance from Transformer",
            image=matplotlib_simple_line_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation performance over epochs",
            image=comparison_scatter_plotnine_svg
        )
    ]
)

multimodal_experiment = Experiment(
    ID="multimodal_net",
    dataset="Deep Learning Methods_multimodal_data",
    algorithm=["Multimodal-Net"],
    tuned_params={"fusion_type": "late", "dropout": "0.2", "learning_rate": "0.001"},
    hyperparam_grid={"fusion_type": "['early', 'late']", "dropout": "[0.1, 0.2, 0.3]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for multimodal fusion network",
            columns=["Split", "MAE", "MSE", "CCC", "F1"],
            rows=[
                ["Split 0", "0.098", "0.024", "0.92", "0.89"],
                ["Split 1", "0.105", "0.028", "0.90", "0.87"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during multimodal training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_CCC", "CV_F1"],
    rows=[
                ["Split 0", "0.089", "0.021", "0.94", "0.91"],
                ["Split 1", "0.096", "0.025", "0.92", "0.89"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Modality contributions and cross-modal attention weights",
            image=plotnine_simple_scatter_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training curves for each modality and fusion performance",
            image=comparison_scatter_matplotlib_svg
        )
    ]
)

experiment_group_3 = ExperimentGroup(
    name="Deep Learning Methods",
    description="Neural network approaches including CNN, ResNet, Transformers for imaging, genomic, and multimodal data",
    datasets=["Deep Learning Methods_imaging_data", "Deep Learning Methods_genomic_data", "Deep Learning Methods_multimodal_data"],
    experiments=["cnn_model", "resnet_model", "transformer_model", "multimodal_net"],
    data_split_scores={
        "Deep Learning Methods_imaging_data": [
            ("Split 0", "ResNet", "0.89", "CCC"),
            ("Split 1", "ResNet", "0.87", "CCC"),
            ("Split 2", "ResNet", "0.85", "CCC")
        ],
        "Deep Learning Methods_genomic_data": [
            ("Split 0", "Transformer", "0.76", "CCC"),
            ("Split 1", "Transformer", "0.74", "CCC")
        ],
        "Deep Learning Methods_multimodal_data": [
            ("Split 0", "Multimodal-Net", "0.92", "CCC"),
            ("Split 1", "Multimodal-Net", "0.90", "CCC")
        ]
    },
    test_scores={
        "Deep Learning Methods_imaging_data_split_0": TableData(
            name="Test Scores - Imaging Data Split 0",
            description="Test set performance for all deep learning methods on imaging data split 0",
            columns=["Algorithm", "MAE", "MSE", "CCC", "LPIPS"],
            rows=[
                ["CNN", "0.082", "0.015", "0.86", "0.09"],
                ["ResNet", "0.075", "0.012", "0.89", "0.07"],
                ["Transformer", "0.089", "0.018", "0.84", "0.11"],
                ["Multimodal-Net", "0.088", "0.017", "0.85", "0.10"]
            ]
        ),
        "Deep Learning Methods_imaging_data_split_1": TableData(
            name="Test Scores - Imaging Data Split 1",
            description="Test set performance for all deep learning methods on imaging data split 1",
            columns=["Algorithm", "MAE", "MSE", "CCC", "LPIPS"],
            rows=[
                ["CNN", "0.089", "0.018", "0.84", "0.11"],
                ["ResNet", "0.081", "0.015", "0.87", "0.08"],
                ["Transformer", "0.095", "0.021", "0.82", "0.13"],
                ["Multimodal-Net", "0.092", "0.019", "0.83", "0.12"]
            ]
        ),
        "Deep Learning Methods_imaging_data_split_2": TableData(
            name="Test Scores - Imaging Data Split 2",
            description="Test set performance for all deep learning methods on imaging data split 2",
            columns=["Algorithm", "MAE", "MSE", "CCC", "LPIPS"],
            rows=[
                ["CNN", "0.095", "0.021", "0.82", "0.13"],
                ["ResNet", "0.088", "0.018", "0.85", "0.10"],
                ["Transformer", "0.102", "0.024", "0.80", "0.15"],
                ["Multimodal-Net", "0.098", "0.022", "0.81", "0.14"]
            ]
        ),
        "Deep Learning Methods_genomic_data_split_0": TableData(
            name="Test Scores - Genomic Data Split 0",
            description="Test set performance for all deep learning methods on genomic data split 0",
            columns=["Algorithm", "MAE", "MSE", "CCC", "Pearson"],
            rows=[
                ["CNN", "0.72", "0.65", "0.73", "0.79"],
                ["ResNet", "0.70", "0.62", "0.75", "0.81"],
                ["Transformer", "0.68", "0.58", "0.76", "0.82"],
                ["Multimodal-Net", "0.69", "0.60", "0.75", "0.80"]
            ]
        ),
        "Deep Learning Methods_genomic_data_split_1": TableData(
            name="Test Scores - Genomic Data Split 1",
            description="Test set performance for all deep learning methods on genomic data split 1",
            columns=["Algorithm", "MAE", "MSE", "CCC", "Pearson"],
            rows=[
                ["CNN", "0.76", "0.70", "0.71", "0.76"],
                ["ResNet", "0.74", "0.67", "0.73", "0.78"],
                ["Transformer", "0.72", "0.63", "0.74", "0.79"],
                ["Multimodal-Net", "0.73", "0.65", "0.72", "0.77"]
            ]
        ),
        "Deep Learning Methods_multimodal_data_split_0": TableData(
            name="Test Scores - Multimodal Data Split 0",
            description="Test set performance for all deep learning methods on multimodal data split 0",
            columns=["Algorithm", "MAE", "MSE", "CCC", "F1"],
            rows=[
                ["CNN", "0.105", "0.028", "0.89", "0.86"],
                ["ResNet", "0.102", "0.026", "0.91", "0.88"],
                ["Transformer", "0.108", "0.030", "0.88", "0.85"],
                ["Multimodal-Net", "0.098", "0.024", "0.92", "0.89"]
            ]
        ),
        "Deep Learning Methods_multimodal_data_split_1": TableData(
            name="Test Scores - Multimodal Data Split 1",
            description="Test set performance for all deep learning methods on multimodal data split 1",
            columns=["Algorithm", "MAE", "MSE", "CCC", "F1"],
            rows=[
                ["CNN", "0.112", "0.032", "0.87", "0.84"],
                ["ResNet", "0.108", "0.030", "0.89", "0.86"],
                ["Transformer", "0.115", "0.034", "0.86", "0.83"],
                ["Multimodal-Net", "0.105", "0.028", "0.90", "0.87"]
            ]
        )
    }
)

# ============= EXPERIMENT GROUP 4: Time Series Methods =============
# Create dataset for Time Series Methods
timeseries_dataset = Dataset(
    ID="Time Series Methods_timeseries_data",
    splits=["split_0", "split_1", "split_2", "split_3"],
    size=("10000", "50"),
    target_stats={"mean": "125.8", "std": "45.2", "min": "45.0", "max": "280.0"},
    data_manager_id="dm_timeseries",
    corr_matrix=PlotData(
        name="Time Series Features Correlation",
        description="Lag correlation and seasonality patterns",
        image=comparison_scatter_matplotlib_svg
    ),
    features=["lag_1", "lag_7", "lag_30", "trend", "seasonal", "moving_avg_7", "moving_avg_30"],
    feature_distributions=[create_feature_distribution(feature, i) for i, feature in enumerate(["lag_1", "lag_7", "lag_30", "trend", "seasonal", "moving_avg_7", "moving_avg_30"])]
)

# Create experiments for Time Series Methods
lstm_experiment = Experiment(
    ID="lstm_model",
    dataset="Time Series Methods_timeseries_data",
    algorithm=["LSTM"],
    tuned_params={"hidden_size": "128", "num_layers": "2", "sequence_length": "30"},
    hyperparam_grid={"hidden_size": "[64, 128, 256]", "num_layers": "[1, 2, 3]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for LSTM on time series data",
            columns=["Split", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["Split 0", "8.45", "89.2", "0.084", "0.078"],
                ["Split 1", "8.12", "85.8", "0.079", "0.072"],
                ["Split 2", "9.23", "102.1", "0.095", "0.089"],
                ["Split 3", "8.78", "94.5", "0.087", "0.081"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during LSTM training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_MAPE", "CV_SMAPE"],
    rows=[
                ["Split 0", "7.82", "81.5", "0.078", "0.072"],
                ["Split 1", "7.59", "79.1", "0.074", "0.068"],
                ["Split 2", "8.67", "94.8", "0.088", "0.082"],
                ["Split 3", "8.21", "87.2", "0.081", "0.075"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Temporal feature importance and attention weights from LSTM",
            image=comparison_scatter_plotnine_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training and validation loss over epochs for LSTM",
            image=matplotlib_simple_line_svg
        )
    ]
)

ts_transformer_experiment = Experiment(
    ID="transformer_ts",
    dataset="Time Series Methods_timeseries_data",
    algorithm=["Transformer"],
    tuned_params={"d_model": "256", "nhead": "8", "sequence_length": "60"},
    hyperparam_grid={"d_model": "[128, 256, 512]", "nhead": "[4, 8, 16]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for Transformer on time series data",
            columns=["Split", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["Split 0", "8.92", "95.1", "0.091", "0.085"],
                ["Split 1", "7.84", "79.2", "0.074", "0.068"],
                ["Split 2", "9.68", "108.3", "0.099", "0.093"],
                ["Split 3", "9.15", "99.8", "0.093", "0.087"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during Transformer training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_MAPE", "CV_SMAPE"],
    rows=[
                ["Split 0", "8.25", "87.8", "0.084", "0.078"],
                ["Split 1", "7.31", "73.5", "0.069", "0.063"],
                ["Split 2", "8.95", "99.7", "0.092", "0.086"],
                ["Split 3", "8.52", "92.1", "0.086", "0.080"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Temporal attention patterns and feature importance",
            image=matplotlib_simple_line_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Training curves showing attention over time",
            image=comparison_scatter_matplotlib_svg
        )
    ]
)

prophet_experiment = Experiment(
    ID="prophet_model",
    dataset="Time Series Methods_timeseries_data",
    algorithm=["Prophet"],
    tuned_params={"seasonality_mode": "multiplicative", "changepoint_prior_scale": "0.05"},
    hyperparam_grid={"seasonality_mode": "['additive', 'multiplicative']", "changepoint_prior_scale": "[0.01, 0.05, 0.1]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for Prophet on time series data",
            columns=["Split", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["Split 0", "10.2", "125.8", "0.105", "0.098"],
                ["Split 1", "9.84", "118.5", "0.098", "0.091"],
                ["Split 2", "11.1", "142.3", "0.115", "0.108"],
                ["Split 3", "10.6", "135.2", "0.103", "0.096"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during Prophet training",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_MAPE", "CV_SMAPE"],
    rows=[
                ["Split 0", "9.58", "116.2", "0.098", "0.091"],
                ["Split 1", "9.21", "109.8", "0.091", "0.084"],
                ["Split 2", "10.45", "131.7", "0.108", "0.101"],
                ["Split 3", "9.95", "125.6", "0.096", "0.089"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="Component importance: trend, seasonality, and holiday effects",
            image=plotnine_simple_scatter_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Model performance vs training period length",
            image=comparison_scatter_plotnine_svg
        )
    ]
)

arima_experiment = Experiment(
    ID="arima_model",
    dataset="Time Series Methods_timeseries_data",
    algorithm=["ARIMA"],
    tuned_params={"p": "2", "d": "1", "q": "2"},
    hyperparam_grid={"p": "[1, 2, 3]", "d": "[0, 1, 2]", "q": "[1, 2, 3]"},
    tables=[
        TableData(
            name="Test Scores",
            description="Test set performance metrics for ARIMA on time series data",
            columns=["Split", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["Split 0", "12.8", "185.4", "0.128", "0.121"],
                ["Split 1", "11.9", "168.2", "0.124", "0.117"],
                ["Split 2", "13.5", "201.8", "0.135", "0.128"],
                ["Split 3", "12.3", "178.9", "0.121", "0.114"]
            ]
        ),
        TableData(
            name="Train CV Scores",
            description="Cross-validation performance during ARIMA model selection",
            columns=["Split", "CV_MAE", "CV_MSE", "CV_MAPE", "CV_SMAPE"],
    rows=[
                ["Split 0", "11.95", "171.8", "0.119", "0.112"],
                ["Split 1", "11.12", "155.6", "0.115", "0.108"],
                ["Split 2", "12.68", "186.9", "0.126", "0.119"],
                ["Split 3", "11.54", "165.2", "0.113", "0.106"]
            ]
        )
    ],
    plots=[
        PlotData(
            name="Feature Importance",
            description="AR, I, and MA component contributions to ARIMA model",
            image=comparison_scatter_matplotlib_svg
        ),
        PlotData(
            name="Learning Curve",
            description="Model selection criteria (AIC, BIC) vs model complexity",
            image=plotnine_simple_scatter_svg
        )
    ]
)

experiment_group_4 = ExperimentGroup(
    name="Time Series Methods",
    description="Specialized time series forecasting models including LSTM, Transformers, Prophet, and ARIMA",
    datasets=["Time Series Methods_timeseries_data"],
    experiments=["lstm_model", "transformer_ts", "prophet_model", "arima_model"],
    data_split_scores={
        "Time Series Methods_timeseries_data": [
            ("Split 0", "LSTM", "0.084", "MAPE"),
            ("Split 1", "Transformer", "0.074", "MAPE"),
            ("Split 2", "Prophet", "0.089", "MAPE"),
            ("Split 3", "LSTM", "0.087", "MAPE")
        ]
    },
    test_scores={
        "Time Series Methods_timeseries_data_split_0": TableData(
            name="Test Scores - Time Series Data Split 0",
            description="Test set performance for all time series methods on time series data split 0",
            columns=["Algorithm", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["LSTM", "8.45", "89.2", "0.084", "0.078"],
                ["Transformer", "8.92", "95.1", "0.091", "0.085"],
                ["Prophet", "10.2", "125.8", "0.105", "0.098"],
                ["ARIMA", "12.8", "185.4", "0.128", "0.121"]
            ]
        ),
        "Time Series Methods_timeseries_data_split_1": TableData(
            name="Test Scores - Time Series Data Split 1",
            description="Test set performance for all time series methods on time series data split 1",
            columns=["Algorithm", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["LSTM", "8.12", "85.8", "0.079", "0.072"],
                ["Transformer", "7.84", "79.2", "0.074", "0.068"],
                ["Prophet", "9.84", "118.5", "0.098", "0.091"],
                ["ARIMA", "11.9", "168.2", "0.124", "0.117"]
            ]
        ),
        "Time Series Methods_timeseries_data_split_2": TableData(
            name="Test Scores - Time Series Data Split 2",
            description="Test set performance for all time series methods on time series data split 2",
            columns=["Algorithm", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["LSTM", "9.23", "102.1", "0.095", "0.089"],
                ["Transformer", "9.68", "108.3", "0.099", "0.093"],
                ["Prophet", "11.1", "142.3", "0.115", "0.108"],
                ["ARIMA", "13.5", "201.8", "0.135", "0.128"]
            ]
        ),
        "Time Series Methods_timeseries_data_split_3": TableData(
            name="Test Scores - Time Series Data Split 3",
            description="Test set performance for all time series methods on time series data split 3",
            columns=["Algorithm", "MAE", "MSE", "MAPE", "SMAPE"],
            rows=[
                ["LSTM", "8.78", "94.5", "0.087", "0.081"],
                ["Transformer", "9.15", "99.8", "0.093", "0.087"],
                ["Prophet", "10.6", "135.2", "0.103", "0.096"],
                ["ARIMA", "12.3", "178.9", "0.121", "0.114"]
            ]
        )
    }
)

# Create the final report data structure
report_data = ReportData(
    navbar=navbar,
    datasets={
        "Linear Methods_housing_data": dataset_1,
        "Linear Methods_medical_data": dataset_2,
        "Tree-Based Methods_clinical_data": clinical_dataset,
        "Tree-Based Methods_sensor_data": sensor_dataset,
        "Deep Learning Methods_imaging_data": imaging_dataset,
        "Deep Learning Methods_genomic_data": genomic_dataset,
        "Deep Learning Methods_multimodal_data": multimodal_dataset,
        "Time Series Methods_timeseries_data": timeseries_dataset
    },
    experiments={
        "linear_regression": linear_experiment,
        "ridge_regression": ridge_experiment,
        "lasso_regression": lasso_experiment,
        "random_forest": rf_experiment,
        "xgboost": xgb_experiment,
        "extra_trees": et_experiment,
        "cnn_model": cnn_experiment,
        "resnet_model": resnet_experiment,
        "transformer_model": transformer_experiment,
        "multimodal_net": multimodal_experiment,
        "lstm_model": lstm_experiment,
        "transformer_ts": ts_transformer_experiment,
        "prophet_model": prophet_experiment,
        "arima_model": arima_experiment
    },
    experiment_groups=[
        experiment_group_1, experiment_group_2, experiment_group_3, experiment_group_4
    ],
    data_managers=data_managers_dict
)
