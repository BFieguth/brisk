"""Provides preprocessing classes for data transformation.

This module contains a base preprocessor class and specific implementations
for various preprocessing tasks such as missing data handling, scaling,
categorical encoding, and feature selection.

Exports:
    BasePreprocessor: Abstract base class for all preprocessors
    MissingDataPreprocessor: Handles missing value imputation and removal
    ScalingPreprocessor: Handles numerical feature scaling
    CategoricalEncodingPreprocessor: Handles categorical feature encoding
    FeatureSelectionPreprocessor: Handles feature selection methods
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import (
    SelectKBest,
    RFECV,
    SequentialFeatureSelector,
    f_classif,
    f_regression,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
)


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors.

    All preprocessors must implement the fit and transform methods to follow
    the scikit-learn estimator interface pattern.
    """

    def __init__(self, **kwargs):
        self.is_fitted = False
        self._validate_params(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def _validate_params(self, **kwargs) -> None:
        """Validate the parameters passed to the preprocessor."""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasePreprocessor":  # pylint: disable=C0103
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target values

        Returns
        -------
        self : BasePreprocessor
            Fitted preprocessor instance
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """Transform the data using the fitted preprocessor.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform

        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:  # pylint: disable=C0103
        """Fit the preprocessor and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target values

        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """Get the feature names after preprocessing.

        Parameters
        ----------
        feature_names : List[str]
            Original feature names

        Returns
        -------
        List[str]
            Feature names after preprocessing
        """
        return feature_names


class MissingDataPreprocessor(BasePreprocessor):
    """Preprocessor for handling missing values.

    Parameters
    ----------
    strategy : str
        Strategy for handling missing values: "drop_rows" or "impute"
    impute_method : str, optional
        Imputation method when strategy="impute": "mean", "median", "mode", "constant"
    constant_value : Any, optional
        Constant value to use when impute_method="constant"
    """

    def __init__(
        self,
        strategy: str = "drop_rows",
        impute_method: str = "mean",
        constant_value: Any = 0,
        **kwargs
    ):
        super().__init__(
            strategy=strategy,
            impute_method=impute_method,
            constant_value=constant_value,
            **kwargs
        )
        self.constant_values = {}

    def _validate_params(self, **kwargs) -> None:
        """Validate missing data handling parameters."""
        strategy = kwargs.get("strategy", "drop_rows")
        valid_strategies = ["drop_rows", "impute"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {strategy}. Choose from {valid_strategies}"
            )

        impute_method = kwargs.get("impute_method", "mean")
        valid_impute_methods = ["mean", "median", "mode", "constant"]
        if impute_method not in valid_impute_methods:
            raise ValueError(
                f"Invalid impute_method: {impute_method}. Choose from {valid_impute_methods}"
            )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MissingDataPreprocessor":
        """Fit the missing data preprocessor.

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target values (not used for missing data handling)

        Returns
        -------
        self : MissingDataPreprocessor
            Fitted preprocessor
        """
        # For imputation methods, fit imputers
        if self.strategy == "impute":
            for column in X.columns:
                if X[column].isnull().any():
                    if self.impute_method == "constant":
                        self.constant_values[column] = self.constant_value
                    elif self.impute_method == "mean":
                        self.constant_values[column] = X[column].mean()
                    elif self.impute_method == "median":
                        self.constant_values[column] = X[column].median()
                    elif self.impute_method == "mode":
                        mode_values = X[column].mode()
                        self.constant_values[column] = mode_values[0] if len(mode_values) > 0 else 0

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """Transform the data by handling missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform

        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_transformed = X.copy()  # pylint: disable=C0103

        # Apply the chosen strategy
        if self.strategy == "drop_rows":
            # Drop rows with any missing values
            X_transformed = X_transformed.dropna()
        elif self.strategy == "impute":
            # Fill missing values with fitted values for columns that were fitted
            for column, value in self.constant_values.items():
                if column in X_transformed.columns:
                    X_transformed[column] = X_transformed[column].fillna(value)

            # For any remaining missing values in other columns, use the default impute method
            remaining_missing = X_transformed.columns[X_transformed.isnull().any()].tolist()
            for column in remaining_missing:
                if column not in self.constant_values:
                    if self.impute_method == "constant":
                        X_transformed[column] = X_transformed[column].fillna(self.constant_value)
                    elif self.impute_method == "mean":
                        X_transformed[column] = X_transformed[column].fillna(X_transformed[column].mean())
                    elif self.impute_method == "median":
                        X_transformed[column] = X_transformed[column].fillna(X_transformed[column].median())
                    elif self.impute_method == "mode":
                        mode_values = X_transformed[column].mode()
                        mode_value = mode_values[0] if len(mode_values) > 0 else 0
                        X_transformed[column] = X_transformed[column].fillna(mode_value)

        return X_transformed

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """Get the feature names after missing data handling.

        Parameters
        ----------
        feature_names : List[str]
            Original feature names

        Returns
        -------
        List[str]
            Feature names (no columns are dropped in this simplified version)
        """
        return feature_names


class ScalingPreprocessor(BasePreprocessor):
    """Preprocessor for scaling numerical features.

    Parameters
    ----------
    method : str or dict
        Scaling method: "standard", "minmax", "robust", "maxabs", "normalizer"
        Or dict mapping column names to methods: {"col1": "standard", "col2": "minmax"}
    categorical_features : List[str], optional
        List of categorical feature names used for finding continous features to scale
    """

    def __init__(self, method: str = "standard", categorical_features: Optional[List[str]] = None, **kwargs):
        super().__init__(
            method=method,
            categorical_features=categorical_features,
            **kwargs
        )
        self.scaler = None

    def _validate_params(self, **kwargs) -> None:
        """Validate the scaling method."""
        method = kwargs.get("method", "standard")
        valid_methods = [
            "standard", "minmax", "robust", "maxabs", "normalizer"
        ]

        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method}"
            )

    def _create_scaler(self, method: str):
        """Create the scaler based on method."""
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "maxabs":
            return MaxAbsScaler()
        elif method == "normalizer":
            return Normalizer()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, X: pd.DataFrame,
            y: Optional[pd.Series] = None) -> "ScalingPreprocessor":
        """Fit the scaler to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target values (not used for scaling)

        Returns
        -------
        self : ScalingPreprocessor
            Fitted preprocessor
        """
        # Get features to scale (exclude categorical features)
        categorical_features = self.categorical_features or []
        features_to_scale = [
            col for col in X.columns
            if col not in categorical_features
        ]

        if features_to_scale:
            # Create single scaler for all features
            self.scaler = self._create_scaler(self.method)
            self.scaler.fit(X[features_to_scale])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """Transform the data using the fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform

        Returns
        -------
        pd.DataFrame
            Transformed data with scaled continous features
        """
        if not self.scaler:
            return X.copy()

        X_transformed = X.copy()  # pylint: disable=C0103

        # Get features to scale
        categorical_features = self.categorical_features or []
        features_to_scale = [col for col in X.columns if col not in categorical_features]

        if features_to_scale:
            # Scale all features with one single scaler
            scaled_features = self.scaler.transform(X[features_to_scale])
            for i, feature in enumerate(features_to_scale):
                X_transformed[feature] = scaled_features[:, i]

        return X_transformed

    def get_feature_names(self, feature_names: Optional[List[str]] = None) -> List[str]:  # pylint: disable=W0237
        """Get the feature names after transformation.

        Parameters
        ----------
        feature_names : List[str], optional
            Original feature names

        Returns
        -------
        List[str]
            Feature names after transformation (same as input)
        """
        if feature_names is None:
            return []
        return feature_names.copy()

class CategoricalEncodingPreprocessor(BasePreprocessor):
    """Preprocessor for categorical feature encoding.

    Supports ordinal, one-hot, label, and cyclic encoding.

    Parameters
    ----------
    method : str or dict
        Encoding method: "ordinal", "onehot", "label", "cyclic"
        Or dict mapping column names to methods: {"col1": "ordinal", "col2": "onehot"}
    categorical_features : List[str], optional
        List of categorical feature names to encode
    """

    def __init__(
        self,
        method: str = "label",
        categorical_features: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            method=method,
            categorical_features=categorical_features,
            **kwargs
        )
        self.encoders = {}

    def _validate_params(self, **kwargs) -> None:
        """Validate encoding parameters."""
        method = kwargs.get("method", "label")
        valid_methods = ["ordinal", "onehot", "label", "cyclic"]

        if isinstance(method, str):
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method: {method}. Choose from {valid_methods}"
                )
        elif isinstance(method, dict):
            for column, encoding_method in method.items():
                if encoding_method not in valid_methods:
                    raise ValueError(
                        f"Invalid method '{encoding_method}' for column '{column}'. Choose from {valid_methods}"
                    )
        else:
            raise ValueError("method must be a string or dict")

    def _create_encoder(self, method: str) -> Any:
        """Create the appropriate encoder for the method."""
        if method == "ordinal":
            return preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        elif method == "onehot":
            return preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        elif method == "label":
            return preprocessing.LabelEncoder()
        elif method == "cyclic":
            return None
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalEncodingPreprocessor":
        """Fit the encoders to the data."""

        # Check if categorical_features is provided
        if self.categorical_features is None:
            self.is_fitted = True
            return self

        # Determine which features to encode and their methods
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue

            # Get the method for this specific feature
            method = self._get_method_for_feature(feature)

            if method == "ordinal":
                encoder = self._create_encoder(method)
                encoder.fit(X[[feature]])
                self.encoders[feature] = encoder

            elif method == "onehot":
                encoder = self._create_encoder(method)
                encoder.fit(X[[feature]])
                self.encoders[feature] = encoder
            elif method == "label":
                encoder = self._create_encoder(method)
                encoder.fit(X[feature])
                self.encoders[feature] = encoder
            elif method == "cyclic":
                unique_values = X[feature].unique()
                sorted_values = sorted(unique_values)
                self.encoders[feature] = sorted_values

        self.is_fitted = True
        return self

    def _get_method_for_feature(self, feature: str) -> str:
        """Get the encoding method for a specific feature."""
        if isinstance(self.method, str):
            return self.method
        else:
            return self.method.get(feature, "label")  # Default to label if not specified

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """Transform the data using the fitted encoders."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # If no categorical features, return original data
        if self.categorical_features is None:
            return X

        X_transformed = X.copy()  # pylint: disable=C0103

        for feature in self.categorical_features:
            if feature not in X.columns or feature not in self.encoders:
                continue

            method = self._get_method_for_feature(feature)

            if method == "ordinal":
                # Apply ordinal encoding
                encoder = self.encoders[feature]
                X_transformed[feature] = encoder.transform(X[[feature]]).flatten()

            elif method == "onehot":
                # Apply one-hot encoding
                encoder = self.encoders[feature]
                encoded = encoder.transform(X[[feature]])
                feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded,
                    index=X.index,
                    columns=feature_names
                )
                X_transformed = pd.concat([X_transformed.drop(columns=[feature]), encoded_df], axis=1)
            elif method == "label":
                # Apply label encoding
                encoder = self.encoders[feature]
                X_transformed[feature] = encoder.transform(X[feature])
            elif method == "cyclic":
                # Apply cyclic encoding
                sorted_values = self.encoders[feature]
                n_categories = len(sorted_values)

                # Create sin and cos features for cyclic encoding
                sin_feature = f"{feature}_sin"
                cos_feature = f"{feature}_cos"

                # Map categories to indices and apply cyclic transformation
                category_to_index = {val: idx for idx, val in enumerate(sorted_values)}
                indices = X[feature].map(category_to_index)

                # Apply cyclic transformation: sin(2π * index / n_categories) and cos(2π * index / n_categories)
                X_transformed[sin_feature] = np.sin(2 * np.pi * indices / n_categories)
                X_transformed[cos_feature] = np.cos(2 * np.pi * indices / n_categories)

                # Remove original feature
                X_transformed = X_transformed.drop(columns=[feature])

        return X_transformed

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """Get the feature names after encoding."""
        if not self.is_fitted:
            return feature_names

        # If no categorical features, return original feature names
        if self.categorical_features is None:
            return feature_names

        new_feature_names = []
        for feature in feature_names:
            if feature in self.categorical_features and feature in self.encoders:
                method = self._get_method_for_feature(feature)
                if method == "onehot":
                    encoder = self.encoders[feature]
                    feature_names_encoded = [f"{feature}_{val}" for val in encoder.categories_[0]]
                    new_feature_names.extend(feature_names_encoded)

                elif method == "cyclic":
                    new_feature_names.extend([f"{feature}_sin", f"{feature}_cos"])
                else:
                    new_feature_names.append(feature)
            else:
                new_feature_names.append(feature)

        return new_feature_names


class FeatureSelectionPreprocessor(BasePreprocessor):
    """Preprocessor for feature selection methods.

    Supports SelectKBest, RFECV, and SequentialFeatureSelector methods.

    Parameters
    ----------
    method : str
        Feature selection method ("selectkbest", "rfecv", "sequential")
    n_features_to_select : int
        Number of features to select
    feature_selection_cv : int
        Number of CV folds for RFECV and SequentialFeatureSelector
    estimator : Any, optional
        Direct estimator to use for RFECV and SequentialFeatureSelector
    algorithm_config : list or AlgorithmCollection of AlgorithmWrapper, optional
        User-provided collection of AlgorithmWrapper objects to use for feature selection
    feature_selection_estimator : str, optional
        The name of the estimator to use for feature selection.
        If not specified, defaults to the first algorithm in the relevant wrapper list
    problem_type : str, optional
        The type of problem ("classification" or "regression").
        Used to determine appropriate scoring function for SelectKBest.
    """

    def __init__(
        self,
        method: str = "selectkbest",
        n_features_to_select: int = 5,
        feature_selection_cv: int = 3,
        estimator: Optional[Any] = None,
        algorithm_config=None,
        feature_selection_estimator: Optional[str] = None,
        problem_type: str = "classification",
        **kwargs
    ):
        super().__init__(
            method=method,
            n_features_to_select=n_features_to_select,
            feature_selection_cv=feature_selection_cv,
            estimator=estimator,
            algorithm_config=algorithm_config,
            feature_selection_estimator=feature_selection_estimator,
            problem_type=problem_type,
            **kwargs
        )
        self.selector = None
        self.scaler = None  # Store fitted scaler for internal use

    def _validate_params(self, **kwargs) -> None:
        """Validate feature selection parameters."""
        method = kwargs.get("method", "selectkbest")
        valid_methods = ["selectkbest", "rfecv", "sequential"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. Choose from {valid_methods}"
            )

        n_features = kwargs.get("n_features_to_select", 5)
        if n_features < 1:
            raise ValueError("n_features_to_select must be >= 1")

        cv = kwargs.get("feature_selection_cv", 3)
        if cv < 2:
            raise ValueError("feature_selection_cv must be >= 2")

        problem_type = kwargs.get("problem_type", "classification")
        valid_problem_types = ["classification", "regression"]
        if problem_type not in valid_problem_types:
            raise ValueError(
                f"Invalid problem_type: {problem_type}. Choose from {valid_problem_types}"
            )

    def _get_feature_selection_estimator(self):
        """Get the estimator for feature selection using the original DataManager pattern."""
        if self.method in ("rfecv", "sequential"):
            if self.algorithm_config is None:
                raise ValueError("algorithm_config must be provided.")
            wrapper_list = self.algorithm_config
            if self.feature_selection_estimator:
                for wrapper in wrapper_list:
                    if wrapper.name == self.feature_selection_estimator:
                        return wrapper.instantiate()
            return wrapper_list[0].instantiate()
        return None

    def _create_selector(self) -> Any:
        """Create the feature selector based on the method."""
        if self.method == "selectkbest":
            # Use appropriate scoring function based on problem type
            if self.problem_type == "classification":
                return SelectKBest(score_func=f_classif, k=self.n_features_to_select)
            else:  # regression
                return SelectKBest(score_func=f_regression, k=self.n_features_to_select)

        elif self.method in ["rfecv", "sequential"]:
            # Use the original pattern to get the estimator
            estimator = self.estimator if self.estimator is not None else self._get_feature_selection_estimator()
            if estimator is None:
                raise ValueError(
                    f"estimator must be provided for {self.method} method"
                )

            if self.method == "rfecv":
                return RFECV(
                    estimator=estimator,
                    min_features_to_select=self.n_features_to_select,
                    step=1,
                    cv=self.feature_selection_cv,
                )
            else:  # sequential
                return SequentialFeatureSelector(
                    estimator,
                    n_features_to_select=self.n_features_to_select,
                    direction="forward",
                    cv=self.feature_selection_cv,
                )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelectionPreprocessor":
        """Fit the feature selector to the data."""
        if self.method in ["rfecv", "sequential"] and y is None:
            raise ValueError(f"y must be provided for {self.method} method")

        # If we have a scaler, use scaled data for feature selection (better performance)
        # But only scale the continuous features that the scaler was fitted on
        if self.scaler is not None:
            # Get the features that the scaler was fitted on (continuous features only)
            scaler_features = self.scaler.feature_names_in_

            # Create a copy of X and scale only the continuous features
            X_scaled = X.copy()  # pylint: disable=C0103
            if scaler_features is not None:
                # Scale only the features that the scaler was fitted on
                X_scaled[scaler_features] = self.scaler.transform(X[scaler_features])
        else:
            X_scaled = X  # pylint: disable=C0103

        self.selector = self._create_selector()
        if y is not None:
            self.selector.fit(X_scaled, y)
        else:
            self.selector.fit(X_scaled)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """Transform the data using the fitted selector."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if self.selector is None:
            return X

        # Get selected feature names
        selected_features = self.get_feature_names(list(X.columns))

        # Return selected features from original unscaled data
        return X[selected_features]

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """Get the selected feature names."""
        if not self.is_fitted or self.selector is None:
            return feature_names

        if hasattr(self.selector, "get_support"):
            selected_mask = self.selector.get_support()
            return [name for name, keep in zip(feature_names, selected_mask) if keep]

        return feature_names
