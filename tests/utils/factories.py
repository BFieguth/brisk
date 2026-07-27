"""Factories to create objects for testing."""
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import linear_model, metrics

from brisk.adapters.sklearn.model_adapter import (
    SklearnAlgorithmWrapper as AlgorithmWrapper,
)
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.configuration.algorithm_collection import AlgorithmCollection
from brisk.adapters.sklearn.metric_adapter import (
    SklearnMetricManager as MetricManager,
    SklearnMetricWrapper as MetricWrapper,
)
from brisk.data import data_split_info, data_splits, splitkey

# pylint: disable=W0212

class AlgorithmFactory:
    """Factory to create AlgorithmWrapper instances for use in tests."""
    @classmethod
    def simple(cls):
        return AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            default_params={"alpha": 0.5},
            hyperparam_grid={"alpha": [0.1, 0.5, 1.0]}
        )

    @classmethod
    def full(
        cls,
        name: str = "ridge",
        display_name: str = "Ridge Regression",
        algorithm_class=linear_model.Ridge,
        default_params: Optional[Dict[str, Any]] = None,
        hyperparam_grid: Optional[Dict[str, Any]] = None
    ):
        if default_params is None:
            default_params = {"alpha": 0.5}
        if hyperparam_grid is None:
            hyperparam_grid = {"alpha": [0.1, 0.5, 1.0]}
        return AlgorithmWrapper(
            name=name,
            display_name=display_name,
            algorithm_class=algorithm_class,
            default_params=default_params,
            hyperparam_grid=hyperparam_grid
        )

    @classmethod
    def collection(cls):
        return AlgorithmCollection(
            cls.simple(),
            cls.full(
                "linear", "Linear Regression", linear_model.LinearRegression,
                {}, {}
            ),
            cls.full("lasso", "LASSO Regression", linear_model.Lasso)
        )


class ExperimentGroupFactory:
    """Factory to create ExperimentGroup instances for use in tests."""

    DEFAULT_NAME = "test_group"
    DEFAULT_DATASETS = ["data.csv"]
    DEFAULT_WORKFLOW = "test_workflow"
    DEFAULT_ALGORITHMS = ["ridge"]

    @classmethod
    def create_dataset_files(cls, tmp_path: Path, datasets: list[str]):
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            if isinstance(dataset, str):
                (datasets_dir / dataset).write_text("")
            elif isinstance(dataset, tuple):
                (datasets_dir / dataset[0]).write_text("")

    @classmethod
    def simple(
        cls,
        tmp_path: Path,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls.create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            name=name or cls.DEFAULT_NAME,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )

    @classmethod
    def with_data_config(
        cls,
        tmp_path: Path,
        data_config: dict,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls.create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            name=name or cls.DEFAULT_NAME,
            data_config=data_config,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )

    @classmethod
    def with_hyperparam_grid(
        cls,
        tmp_path: Path,
        hyperparam_grid: dict,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls.create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            name=name or cls.DEFAULT_NAME,
            algorithm_config=hyperparam_grid,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )

    @classmethod
    def with_description(
        cls,
        tmp_path: Path,
        description: str,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls.create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            description=description,
            name=name or cls.DEFAULT_NAME,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )


class DataFrameFactory:
    """Factory for creating deterministic test datasets.

    All methods are idempotent - calling with the same parameters will produce
    identical datasets. Maximum constraints: 10 features, 100 samples.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate (max 100)
    n_features : int
        Number of features to generate (max 10)
    problem_type : {'regression', 'binary', 'multiclass'}
        Type of target variable to generate
    feature_types : List[str], optional
        List specifying type for each feature: 'continuous' or 'categorical'
        If None, all features are continuous. Length must equal n_features.
    n_categorical_levels : int, optional
        Number of unique levels for categorical features (default 5)
    add_group : bool, optional
        Whether to add a 'group' column (default False)
    n_groups : int, optional
        Number of unique groups if add_group=True (default 3)
    test_size : float, optional
        Proportion of data for test set in train_test_split (default 0.2)
    random_state : int, optional
        Seed for reproducibility (default 42)

    Examples
    --------
    >>> # Simple regression with continuous features
    >>> df = DataFrameFactory.dataframe(
    ...     n_samples=50, n_features=3, problem_type='regression'
    ... )

    >>> # Mixed feature types with groups
    >>> df = DataFrameFactory.dataframe(
    ...     n_samples=80,
    ...     n_features=4,
    ...     problem_type='binary',
    ...     feature_types=['continuous', 'continuous',
    ...                    'categorical', 'categorical'],
    ...     add_group=True
    ... )

    >>> # Train-test split
    >>> data = DataFrameFactory.train_test_split(
    ...     n_samples=100, n_features=5, problem_type='multiclass'
    ... )
    """
    MAX_FEATURES = 10
    MAX_SAMPLES = 100

    @classmethod
    def dataframe(
        cls,
        n_samples: int,
        n_features: int,
        problem_type: Literal["regression", "binary", "multiclass"],
        feature_types: Optional[
            List[Literal["continuous", "categorical"]]
        ] = None,
        n_categorical_levels: int = 5,
        add_group: bool = False,
        n_groups: int = 3,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Create a single DataFrame with features and target."""
        cls._validate_params(n_samples, n_features, feature_types)

        np.random.seed(random_state)

        if feature_types is None:
            feature_types = ["continuous"] * n_features

        # Create features
        data = {}
        for i, feat_type in enumerate(feature_types):
            col_name = f"feature_{i}"
            if feat_type == "continuous":
                data[col_name] = np.random.randn(n_samples)
            elif feat_type == "categorical":
                categories = [f"cat_{j}" for j in range(n_categorical_levels)]
                data[col_name] = np.random.choice(categories, size=n_samples)
            else:
                raise ValueError(f"Invalid feature type: {feat_type}")

        # Add group column
        if add_group:
            groups = [f"group_{j}" for j in range(n_groups)]
            data["group"] = np.random.choice(groups, size=n_samples)

        # Create target
        if problem_type == "regression":
            data["target"] = np.random.randn(n_samples)
        elif problem_type == "binary":
            data["target"] = np.random.choice([0, 1], size=n_samples)
        elif problem_type == "multiclass":
            data["target"] = np.random.choice([0, 1, 2], size=n_samples)
        else:
            raise ValueError(f"Invalid problem_type: {problem_type}")

        return pd.DataFrame(data)

    @classmethod
    def train_test_split(
        cls,
        n_samples: int,
        n_features: int,
        problem_type: Literal["regression", "binary", "multiclass"],
        feature_types: Optional[
            List[Literal["continuous", "categorical"]]
        ] = None,
        n_categorical_levels: int = 5,
        add_group: bool = False,
        n_groups: int = 3,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Creates train-test split DataFrame."""
        df = cls.dataframe(
            n_samples=n_samples,
            n_features=n_features,
            problem_type=problem_type,
            feature_types=feature_types,
            n_categorical_levels=n_categorical_levels,
            add_group=add_group,
            n_groups=n_groups,
            random_state=random_state
        )

        np.random.seed(random_state)
        n_test = int(n_samples * test_size)
        test_indices = np.random.choice(n_samples, size=n_test, replace=False)
        train_indices = np.array([
            i for i in range(n_samples) if i not in test_indices
        ])

        target_col = "target"
        feature_cols = [col for col in df.columns if col != target_col]

        result = {
            "X_train": df.loc[train_indices, feature_cols].reset_index(
                drop=True
            ),
            "X_test": df.loc[test_indices, feature_cols].reset_index(drop=True),
            "y_train": df.loc[train_indices, target_col].reset_index(drop=True),
            "y_test": df.loc[test_indices, target_col].reset_index(drop=True),
        }

        if add_group:
            groups = df["group"]

            result["group_index_train"] = {
                "values": groups.iloc[train_indices].values.copy(),
                "indices": train_indices.copy(),
                "series": groups.iloc[train_indices].copy()
            }
            result["group_index_test"] = {
                "values": groups.iloc[test_indices].values.copy(),
                "indices": test_indices.copy(),
                "series": groups.iloc[test_indices].copy()
            }
        else:
            result["group_index_train"] = None
            result["group_index_test"] = None

        return result

    @classmethod
    def _validate_params(
        cls,
        n_samples: int,
        n_features: int,
        feature_types: Optional[List[str]]
    ) -> None:
        """Validate input parameters against constraints"""
        if n_samples > cls.MAX_SAMPLES:
            raise ValueError(
                f"n_samples ({n_samples}) exceeds maximum ({cls.MAX_SAMPLES})"
            )
        if n_features > cls.MAX_FEATURES:
            raise ValueError(
                f"n_features ({n_features}) exceeds maximum "
                f"({cls.MAX_FEATURES})"
            )
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if feature_types is not None and len(feature_types) != n_features:
            raise ValueError(
                f"Length of feature_types ({len(feature_types)} must match)"
                f"n_features ({n_features})"
            )


class MetricManagerFactory:
    """Factory to create MetricManager instances for use in tests."""

    @classmethod
    def regression(cls):
        return MetricManager(
            MetricWrapper(
                name="mean_absolute_error",
                func=metrics._regression.mean_absolute_error,
                display_name="Mean Absolute Error",
                abbr="MAE",
                greater_is_better=False
            ),
            MetricWrapper(
                name="mean_squared_error",
                func=metrics._regression.mean_squared_error,
                display_name="Mean Squared Error",
                abbr="MSE",
                greater_is_better=False
            ),
        )

    @classmethod
    def classification(cls):
        return MetricManager(
            MetricWrapper(
                name="accuracy",
                func=metrics.accuracy_score,
                display_name="Accuracy",
                greater_is_better=True
            ),
            MetricWrapper(
                name="precision",
                func=metrics.precision_score,
                display_name="Precision",
                greater_is_better=True
            ),
        )


class DataSplitInfoFactory:
    """Factory for creating DataSplitInfo instances for testing."""
    @classmethod
    def simple(
        cls,
        split_index: int = 0,
        group_name: str = "test_group",
        dataset_name: str = "test_data.csv",
        table_name: str = None,
        problem_type: str = "regression",
        n_samples: int = 100,
        n_features: int = 3
    ):
        """Create a simple DataSplitInfo instance."""
        data = DataFrameFactory.train_test_split(
            n_samples=n_samples,
            n_features=n_features,
            problem_type=problem_type
        )

        split_key_obj = splitkey.SplitKey(
            group_name=group_name,
            dataset_name=dataset_name,
            table_name=table_name
        )

        return data_split_info.DataSplitInfo(
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=data["y_train"],
            y_test=data["y_test"],
            group_index_train=data["group_index_train"],
            group_index_test=data["group_index_test"],
            split_key=split_key_obj,
            split_index=split_index
        )


class DataSplitsFactory:
    """Factory for creating DataSplits instances for testing."""
    @classmethod
    def simple(
        cls,
        n_splits: int = 3,
        group_name: str = "test_group",
        dataset_name: str = "test_data.csv",
        problem_type: str = "regression"
    ):
        """Create a DataSplits instance with multiple splits."""
        splits = data_splits.DataSplits(n_splits=n_splits)

        for i in range(n_splits):
            split_info = DataSplitInfoFactory.simple(
                split_index=i,
                group_name=group_name,
                dataset_name=dataset_name,
                problem_type=problem_type
            )
            splits.add(split_info)

        return splits

