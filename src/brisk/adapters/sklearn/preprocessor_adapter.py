"""Sklearn adapter for the preprocessing system.

Wraps scikit-learn preprocessing, encoding, and feature-selection utilities
behind the ``PreprocessorPort`` interface. All direct ``sklearn`` usage for
data preprocessing lives here so the domain core depends only on the port.

Classes
-------
BasePreprocessor
    Abstract base providing the common preprocessor interface.
MissingDataPreprocessor
    Handles missing value imputation and removal strategies.
ScalingPreprocessor
    Scales numerical features using scikit-learn scalers.
CategoricalEncodingPreprocessor
    Encodes categorical features with multiple strategies.
FeatureSelectionPreprocessor
    Selects features using scikit-learn feature-selection algorithms.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import pandas as pd
from sklearn import feature_selection, preprocessing


class BasePreprocessor(abc.ABC):
    """Abstract base class for all preprocessors.

    All preprocessors must implement the fit and transform methods to follow
    the scikit-learn estimator interface pattern. This ensures consistency
    across all preprocessing operations and conformance to ``PreprocessorPort``.

    Parameters
    ----------
    **kwargs
        Additional parameters specific to each preprocessor implementation.

    Attributes
    ----------
    is_fitted : bool
        Whether the preprocessor has been fitted to data.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.is_fitted = False
        self._validate_params(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abc.abstractmethod
    def _validate_params(self, **kwargs: Any) -> None:
        """Validate the parameters passed to the preprocessor.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """

    @abc.abstractmethod
    def fit(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> "BasePreprocessor":
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values.
        categorical_features : list[str], optional
            List of categorical feature names.

        Returns
        -------
        BasePreprocessor
            Fitted preprocessor instance.
        """

    @abc.abstractmethod
    def transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform the data using the fitted preprocessor.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target values to transform (if applicable).

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(transformed_X, transformed_y)``.

        Raises
        ------
        ValueError
            If the preprocessor has not been fitted.
        """

    @abc.abstractmethod
    def export_params(self) -> dict[str, Any]:
        """Export parameters for serialization and rerun functionality.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters in JSON-serializable format.
        """

    def fit_transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Fit the preprocessor and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values.
        categorical_features : list[str], optional
            List of categorical feature names.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(transformed_X, transformed_y)``.
        """
        return self.fit(X, y, categorical_features).transform(X, y)

    def get_feature_names(self, feature_names: list[str]) -> list[str]:
        """Get the feature names after preprocessing.

        Parameters
        ----------
        feature_names : list[str]
            Original feature names.

        Returns
        -------
        list[str]
            Feature names after preprocessing (unchanged by default).
        """
        return feature_names


class MissingDataPreprocessor(BasePreprocessor):
    """Preprocessor for handling missing values in datasets.

    Provides strategies for dealing with missing data including dropping
    rows with missing values or imputing missing values using various
    statistical methods.

    Parameters
    ----------
    strategy : str, default="drop_rows"
        Strategy for handling missing values: "drop_rows" or "impute".
    impute_method : str, default="mean"
        Imputation method when strategy="impute": "mean", "median", "mode",
        or "constant".
    constant_value : Any, default=0
        Constant value to use when impute_method="constant".

    Attributes
    ----------
    constant_values : dict
        Dictionary mapping column names to their fitted imputation values.
    is_fitted : bool
        Whether the preprocessor has been fitted.

    Examples
    --------
    Drop rows with missing values:
        >>> preprocessor = MissingDataPreprocessor(strategy="drop_rows")

    Impute with mean values:
        >>> preprocessor = MissingDataPreprocessor(
        ...     strategy="impute", impute_method="mean"
        ... )
    """

    strategy: str
    impute_method: str
    constant_value: Any

    def __init__(
        self,
        strategy: str = "drop_rows",
        impute_method: str = "mean",
        constant_value: Any = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            strategy=strategy,
            impute_method=impute_method,
            constant_value=constant_value,
            **kwargs,
        )
        self.constant_values: dict[str, Any] = {}

    def _validate_params(self, **kwargs: Any) -> None:
        """Validate missing data handling parameters.

        Raises
        ------
        ValueError
            If strategy or impute_method is invalid.
        """
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
                f"Invalid impute_method: {impute_method}. Choose from "
                f"{valid_impute_methods}"
            )

    def fit(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> "MissingDataPreprocessor":
        """Fit the missing data preprocessor.

        Learns imputation values from the training data for each column
        that contains missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values (not used for missing data handling).
        categorical_features : list[str], optional
            Unused; present for interface consistency.

        Returns
        -------
        MissingDataPreprocessor
            Fitted preprocessor.
        """
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
                        self.constant_values[column] = (
                            mode_values[0] if len(mode_values) > 0 else 0
                        )

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform the data by handling missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target values (passed through unchanged).

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(transformed_X, y)``.

        Raises
        ------
        ValueError
            If the preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_transformed = X.copy()  # pylint: disable=C0103

        if self.strategy == "drop_rows":
            X_transformed = X_transformed.dropna()  # pylint: disable=C0103
            if y is not None:
                y = y.loc[X_transformed.index]
        elif self.strategy == "impute":
            for column, value in self.constant_values.items():
                if column in X_transformed.columns:
                    X_transformed[column] = X_transformed[column].fillna(value)

            remaining_missing = X_transformed.columns[
                X_transformed.isnull().any()
            ].tolist()
            for column in remaining_missing:
                if column not in self.constant_values:
                    if self.impute_method == "constant":
                        X_transformed[column] = X_transformed[column].fillna(
                            self.constant_value
                        )
                    elif self.impute_method == "mean":
                        X_transformed[column] = X_transformed[column].fillna(
                            X_transformed[column].mean()
                        )
                    elif self.impute_method == "median":
                        X_transformed[column] = X_transformed[column].fillna(
                            X_transformed[column].median()
                        )
                    elif self.impute_method == "mode":
                        mode_values = X_transformed[column].mode()
                        mode_value = (
                            mode_values[0] if len(mode_values) > 0 else 0
                        )
                        X_transformed[column] = X_transformed[column].fillna(
                            mode_value
                        )

        return X_transformed, y

    def get_feature_names(self, feature_names: list[str]) -> list[str]:
        """Get the feature names after missing data handling.

        Parameters
        ----------
        feature_names : list[str]
            Original feature names.

        Returns
        -------
        list[str]
            Feature names (unchanged).
        """
        return feature_names

    def export_params(self) -> dict[str, Any]:
        """Export parameters for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters.
        """
        return {
            "strategy": self.strategy,
            "impute_method": self.impute_method,
            "constant_value": self.constant_value,
        }


class ScalingPreprocessor(BasePreprocessor):
    """Preprocessor for scaling numerical features.

    Provides various scaling methods for numerical features while preserving
    categorical features in their original form. Supports standard, min-max,
    robust, max-abs, and normalizer scaling methods.

    Parameters
    ----------
    method : str, default="standard"
        Scaling method: "standard", "minmax", "robust", "maxabs", or
        "normalizer".

    Attributes
    ----------
    scaler : sklearn.preprocessing scaler
        The fitted scaler object.
    _scaled_features : list
        List of feature names that were scaled during fit.
    is_fitted : bool
        Whether the preprocessor has been fitted.
    """

    method: str
    _scaled_features: list[str]

    def __init__(self, method: str = "standard", **kwargs: Any) -> None:
        super().__init__(method=method, **kwargs)
        self.scaler: Any = None

    def _validate_params(self, **kwargs: Any) -> None:
        """Validate the scaling method.

        Raises
        ------
        ValueError
            If method is not a valid scaling method.
        """
        method = kwargs.get("method", "standard")
        valid_methods = [
            "standard", "minmax", "robust", "maxabs", "normalizer"
        ]

        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method}"
            )

    def _create_scaler(self, method: str) -> Any:
        """Create the scaler based on method.

        Parameters
        ----------
        method : str
            Scaling method name.

        Returns
        -------
        sklearn.preprocessing scaler
            The appropriate scaler object.

        Raises
        ------
        ValueError
            If method is unknown.
        """
        if method == "standard":
            return preprocessing.StandardScaler()
        elif method == "minmax":
            return preprocessing.MinMaxScaler()
        elif method == "robust":
            return preprocessing.RobustScaler()
        elif method == "maxabs":
            return preprocessing.MaxAbsScaler()
        elif method == "normalizer":
            return preprocessing.Normalizer()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> "ScalingPreprocessor":
        """Fit the scaler to the data.

        Learns scaling parameters from the training data, excluding
        categorical features from scaling.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values (not used for scaling).
        categorical_features : list[str], optional
            List of categorical feature names to exclude from scaling.

        Returns
        -------
        ScalingPreprocessor
            Fitted preprocessor.
        """
        categorical_features = categorical_features or []
        features_to_scale = [
            col for col in X.columns
            if col not in categorical_features
        ]

        if features_to_scale:
            self.scaler = self._create_scaler(self.method)
            self.scaler.fit(X[features_to_scale])
            self._scaled_features = features_to_scale

        self.is_fitted = True
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Fit the scaler and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit and transform.
        y : pd.Series, optional
            Target values (not used for scaling).
        categorical_features : list[str], optional
            List of categorical feature names to exclude from scaling.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(scaled_X, y)``.
        """
        return self.fit(X, y, categorical_features).transform(X, y)

    def transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform the data using the fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target values (passed through unchanged).

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(scaled_X, y)``.
        """
        if not self.scaler:
            return X.copy(), y

        X_transformed = X.copy()  # pylint: disable=C0103

        features_to_scale = getattr(self, "_scaled_features", [])

        if features_to_scale:
            scaled_features = self.scaler.transform(X[features_to_scale])
            for i, feature in enumerate(features_to_scale):
                X_transformed[feature] = scaled_features[:, i]

        return X_transformed, y

    def get_feature_names(
        self,
        feature_names: list[str] | None = None,
    ) -> list[str]:
        """Get the feature names after transformation.

        Parameters
        ----------
        feature_names : list[str], optional
            Original feature names.

        Returns
        -------
        list[str]
            Feature names after transformation (same as input).
        """
        if feature_names is None:
            return []
        return feature_names.copy()

    def export_params(self) -> dict[str, Any]:
        """Export parameters for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters.
        """
        return {
            "method": self.method,
        }


class CategoricalEncodingPreprocessor(BasePreprocessor):
    """Preprocessor for categorical feature encoding.

    Supports multiple encoding strategies including ordinal, one-hot, label,
    cyclic, and threshold encoding. Can encode both features and target
    variables based on configuration.

    Parameters
    ----------
    method : str or dict, default="label"
        Encoding method: "ordinal", "onehot", "label", "cyclic", or "threshold",
        or a dict mapping column names to methods. If a target feature name
        matches a key in the dict, it will be encoded.
    cutoffs : list, optional
        For threshold encoding: list of cutoff values to create bins.

    Attributes
    ----------
    encoders : dict
        Dictionary mapping feature names to their fitted encoder objects.
    target_encoder : object or None
        Encoder for target variable if target name matches method dict.
    is_fitted : bool
        Whether the preprocessor has been fitted.
    """

    method: str | dict[str, str]

    def __init__(
        self,
        method: str = "label",
        cutoffs: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(method=method, cutoffs=cutoffs, **kwargs)
        self.encoders: dict[str, Any] = {}
        self.target_encoder: dict[str, Any] | None = None
        self.cutoffs = cutoffs or []

    def _validate_params(self, **kwargs: Any) -> None:
        """Validate encoding parameters.

        Raises
        ------
        ValueError
            If method is invalid or cutoffs are missing for threshold encoding.
        """
        method = kwargs.get("method", "label")
        cutoffs = kwargs.get("cutoffs", [])
        valid_methods = ["ordinal", "onehot", "label", "cyclic", "threshold"]

        if isinstance(method, str):
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method: {method}. Choose from {valid_methods}"
                )
        elif isinstance(method, dict):
            for column, encoding_method in method.items():
                if encoding_method not in valid_methods:
                    raise ValueError(
                        f"Invalid method '{encoding_method}' for column "
                        f"'{column}'. Choose from {valid_methods}"
                    )
        else:
            raise ValueError("method must be a string or dict")

        if isinstance(method, str) and method == "threshold":
            if not cutoffs:
                raise ValueError(
                    "cutoffs must be provided for threshold encoding"
                )
        elif isinstance(method, dict):
            for column, encoding_method in method.items():
                if encoding_method == "threshold" and not cutoffs:
                    raise ValueError(
                        "cutoffs must be provided for threshold encoding of "
                        f"column '{column}'"
                    )

    def _create_encoder(self, method: str) -> Any:
        """Create the appropriate encoder for the method.

        Parameters
        ----------
        method : str
            Encoding method name.

        Returns
        -------
        sklearn.preprocessing encoder or None
            The appropriate encoder object, or None for custom methods.

        Raises
        ------
        ValueError
            If method is unknown.
        """
        if method == "ordinal":
            return preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        elif method == "onehot":
            return preprocessing.OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
        elif method == "label":
            return preprocessing.LabelEncoder()
        elif method == "cyclic":
            return None
        elif method == "threshold":
            return None
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _apply_threshold_encoding(
        self,
        data: pd.Series,
        cutoffs: list[float],
    ) -> pd.Series:
        """Apply threshold encoding to convert continuous values to bins.

        Parameters
        ----------
        data : pd.Series
            Data to encode.
        cutoffs : list[float]
            Cutoff values for binning.

        Returns
        -------
        pd.Series
            Encoded data with integer bin labels.
        """
        bins = [-np.inf] + cutoffs + [np.inf]
        labels = list(range(len(bins) - 1))

        binned = pd.cut(data, bins=bins, labels=labels, include_lowest=True)
        return binned.astype(int)

    def _should_encode_target(self, y: pd.Series | None) -> bool:
        """Check if target should be encoded.

        Parameters
        ----------
        y : pd.Series or None
            Target variable.

        Returns
        -------
        bool
            True if target should be encoded, False otherwise.
        """
        if y is None or not hasattr(y, "name") or y.name is None:
            return False

        return isinstance(self.method, dict) and y.name in self.method

    def fit(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> "CategoricalEncodingPreprocessor":
        """Fit the encoders to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values.
        categorical_features : list[str], optional
            List of categorical feature names to encode.

        Returns
        -------
        CategoricalEncodingPreprocessor
            Fitted preprocessor.
        """
        categorical_features = categorical_features or []

        if y is not None and self._should_encode_target(y):
            target_method = self._get_method_for_feature(str(y.name))

            if target_method == "threshold":
                self.target_encoder = {
                    "method": "threshold",
                    "cutoffs": self.cutoffs,
                }
            else:
                encoder = self._create_encoder(target_method)
                if encoder is not None:
                    if target_method == "label":
                        encoder.fit(y)
                    else:
                        encoder.fit(np.asarray(y).reshape(-1, 1))
                    self.target_encoder = {
                        "method": target_method,
                        "encoder": encoder,
                    }

        if not categorical_features:
            self.is_fitted = True
            return self

        for feature in categorical_features:
            if feature not in X.columns:
                continue

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
            elif method == "threshold":
                self.encoders[feature] = {
                    "method": "threshold",
                    "cutoffs": self.cutoffs,
                }

        self.is_fitted = True
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Fit the encoders and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit and transform.
        y : pd.Series, optional
            Target values.
        categorical_features : list[str], optional
            List of categorical feature names to encode.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(encoded_X, encoded_y)``.
        """
        self.fit(X, y, categorical_features)
        return self.transform(X, y)

    def _get_method_for_feature(self, feature: str) -> str:
        """Get the encoding method for a specific feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        str
            Encoding method for the feature.
        """
        if isinstance(self.method, str):
            return self.method
        else:
            return self.method.get(feature, "label")

    def transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform features using the fitted encoders.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target values to transform (if applicable).

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(encoded_X, encoded_y)``.

        Raises
        ------
        ValueError
            If the preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_transformed = self._transform_features(X)  # pylint: disable=C0103

        y_transformed = y
        if (
            y is not None
            and self._should_encode_target(y)
            and self.target_encoder
        ):
            y_transformed = self._transform_target(y)

        return X_transformed, y_transformed

    def _transform_features(
        self, X: pd.DataFrame  # pylint: disable=C0103
    ) -> pd.DataFrame:
        """Transform features using fitted encoders.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        if not hasattr(self, "encoders") or not self.encoders:
            return X

        X_transformed = X.copy()  # pylint: disable=C0103

        for feature in self.encoders:
            if feature not in X.columns or feature not in self.encoders:
                continue

            method = self._get_method_for_feature(feature)

            if method == "ordinal":
                encoder = self.encoders[feature]
                X_transformed[feature] = encoder.transform(
                    X[[feature]]
                ).flatten()
            elif method == "onehot":
                encoder = self.encoders[feature]
                encoded = encoder.transform(X[[feature]])
                feature_names = [
                    f"{feature}_{val}" for val in encoder.categories_[0]
                ]
                encoded_df = pd.DataFrame(
                    encoded,
                    index=X.index,
                    columns=feature_names,
                )
                X_transformed = pd.concat(  # pylint: disable=C0103
                    [X_transformed.drop(columns=[feature]), encoded_df], axis=1
                )
            elif method == "label":
                encoder = self.encoders[feature]
                X_transformed[feature] = encoder.transform(X[feature])
            elif method == "cyclic":
                sorted_values = self.encoders[feature]
                n_categories = len(sorted_values)

                sin_feature = f"{feature}_sin"
                cos_feature = f"{feature}_cos"

                category_to_index = {
                    val: idx for idx, val in enumerate(sorted_values)
                }
                indices = X[feature].map(category_to_index)

                X_transformed[sin_feature] = np.sin(
                    2 * np.pi * indices / n_categories
                )
                X_transformed[cos_feature] = np.cos(
                    2 * np.pi * indices / n_categories
                )

                X_transformed = X_transformed.drop(  # pylint: disable=C0103
                    columns=[feature]
                )
            elif method == "threshold":
                encoder_info = self.encoders[feature]
                cutoffs = encoder_info["cutoffs"]
                X_transformed[feature] = self._apply_threshold_encoding(
                    X[feature], cutoffs
                )

        return X_transformed

    def _transform_target(self, y: pd.Series) -> pd.Series:
        """Transform target variable using fitted encoder.

        Parameters
        ----------
        y : pd.Series
            Target variable to transform.

        Returns
        -------
        pd.Series
            Transformed target variable.
        """
        if not self.target_encoder:
            return y

        method = self.target_encoder["method"]

        if method == "threshold":
            cutoffs = self.target_encoder["cutoffs"]
            return self._apply_threshold_encoding(y, cutoffs)
        else:
            encoder = self.target_encoder["encoder"]
            if method == "label":
                return pd.Series(
                    encoder.transform(y), index=y.index, name=y.name
                )
            else:
                transformed = encoder.transform(
                    np.asarray(y).reshape(-1, 1)
                ).flatten()
                return pd.Series(transformed, index=y.index, name=y.name)

    def get_feature_names(self, feature_names: list[str]) -> list[str]:
        """Get the feature names after encoding.

        Parameters
        ----------
        feature_names : list[str]
            Original feature names.

        Returns
        -------
        list[str]
            Updated feature names after encoding.
        """
        if not self.is_fitted:
            return feature_names

        if not hasattr(self, "encoders") or not self.encoders:
            return feature_names

        new_feature_names = []
        for feature in feature_names:
            if feature in self.encoders:
                method = self._get_method_for_feature(feature)
                if method == "onehot":
                    encoder = self.encoders[feature]
                    feature_names_encoded = [
                        f"{feature}_{val}" for val in encoder.categories_[0]
                    ]
                    new_feature_names.extend(feature_names_encoded)
                elif method == "cyclic":
                    new_feature_names.extend(
                        [f"{feature}_sin", f"{feature}_cos"]
                    )
                else:
                    new_feature_names.append(feature)
            else:
                new_feature_names.append(feature)

        return new_feature_names

    def export_params(self) -> dict[str, Any]:
        """Export parameters for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters.
        """
        return {
            "method": self.method,
            "cutoffs": self.cutoffs,
        }


class FeatureSelectionPreprocessor(BasePreprocessor):
    """Preprocessor for feature selection methods.

    Supports various feature selection algorithms including SelectKBest,
    RFECV, and SequentialFeatureSelector. Can use different estimators
    for wrapper methods.

    Parameters
    ----------
    method : str, default="selectkbest"
        Feature selection method ("selectkbest", "rfecv", "sequential").
    n_features_to_select : int, default=5
        Number of features to select.
    feature_selection_cv : int, default=3
        Number of CV folds for RFECV and SequentialFeatureSelector.
    estimator : Any, optional
        Direct estimator to use for RFECV and SequentialFeatureSelector.
    algorithm_config : AlgorithmCollection, optional
        User-provided collection of algorithm wrappers to use for feature
        selection.
    feature_selection_estimator : str, optional
        The name of the estimator to use for feature selection.
    problem_type : str, default="classification"
        The type of problem ("classification" or "regression").

    Attributes
    ----------
    selector : sklearn.feature_selection selector
        The fitted feature selector object.
    scaler : sklearn.preprocessing scaler, optional
        Fitted scaler for internal use (if provided).
    is_fitted : bool
        Whether the preprocessor has been fitted.
    """

    method: str
    n_features_to_select: int
    feature_selection_cv: int
    estimator: Any
    algorithm_config: Any
    feature_selection_estimator: str | None
    problem_type: str

    def __init__(
        self,
        method: str = "selectkbest",
        n_features_to_select: int = 5,
        feature_selection_cv: int = 3,
        estimator: Any | None = None,
        algorithm_config: Any = None,
        feature_selection_estimator: str | None = None,
        problem_type: str = "classification",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            method=method,
            n_features_to_select=n_features_to_select,
            feature_selection_cv=feature_selection_cv,
            estimator=estimator,
            algorithm_config=algorithm_config,
            feature_selection_estimator=feature_selection_estimator,
            problem_type=problem_type,
            **kwargs,
        )
        self.selector: Any = None
        self.scaler: Any = None

    def _validate_params(self, **kwargs: Any) -> None:
        """Validate feature selection parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
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
                f"Invalid problem_type: {problem_type}. Choose from "
                f"{valid_problem_types}"
            )

    def _get_feature_selection_estimator(self) -> Any:
        """Get the estimator for feature selection.

        Returns
        -------
        sklearn estimator
            The estimator to use for feature selection.

        Raises
        ------
        ValueError
            If algorithm_config is not provided for wrapper methods.
        """
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
        """Create the feature selector based on the method.

        Returns
        -------
        sklearn.feature_selection selector
            The appropriate feature selector.

        Raises
        ------
        ValueError
            If estimator is required but not provided.
        """
        if self.method == "selectkbest":
            if self.problem_type == "classification":
                return feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif,
                    k=self.n_features_to_select,
                )
            else:
                return feature_selection.SelectKBest(
                    score_func=feature_selection.f_regression,
                    k=self.n_features_to_select,
                )

        elif self.method in ["rfecv", "sequential"]:
            estimator = (
                self.estimator
                if self.estimator is not None
                else self._get_feature_selection_estimator()
            )
            if estimator is None:
                raise ValueError(
                    f"estimator must be provided for {self.method} method"
                )

            if self.method == "rfecv":
                return feature_selection.RFECV(
                    estimator=estimator,
                    min_features_to_select=self.n_features_to_select,
                    step=1,
                    cv=self.feature_selection_cv,
                )
            else:
                return feature_selection.SequentialFeatureSelector(
                    estimator,
                    n_features_to_select=self.n_features_to_select,
                    direction="forward",
                    cv=self.feature_selection_cv,
                )

    def fit(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> "FeatureSelectionPreprocessor":
        """Fit the feature selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data features.
        y : pd.Series, optional
            Target values (required for RFECV and SequentialFeatureSelector).
        categorical_features : list[str], optional
            Unused; present for interface consistency.

        Returns
        -------
        FeatureSelectionPreprocessor
            Fitted preprocessor.

        Raises
        ------
        ValueError
            If y is required but not provided for wrapper methods.
        """
        if self.method in ["rfecv", "sequential"] and y is None:
            raise ValueError(f"y must be provided for {self.method} method")

        if self.scaler is not None:
            scaler_features = self.scaler.feature_names_in_

            X_scaled = X.copy()  # pylint: disable=C0103
            if scaler_features is not None:
                X_scaled[scaler_features] = self.scaler.transform(
                    X[scaler_features]
                )
        else:
            X_scaled = X  # pylint: disable=C0103

        self.selector = self._create_selector()
        if y is not None:
            self.selector.fit(X_scaled, y)
        else:
            self.selector.fit(X_scaled)

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        y: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform the data using the fitted selector.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target values (passed through unchanged).

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Tuple containing ``(selected_X, y)``.

        Raises
        ------
        ValueError
            If the preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if self.selector is None:
            return X, y

        selected_features = self.get_feature_names(list(X.columns))

        X_transformed = X[selected_features]  # pylint: disable=C0103

        return X_transformed, y

    def get_feature_names(self, feature_names: list[str]) -> list[str]:
        """Get the selected feature names after feature selection.

        Parameters
        ----------
        feature_names : list[str]
            Original feature names.

        Returns
        -------
        list[str]
            Names of selected features.
        """
        if not self.is_fitted or self.selector is None:
            return feature_names

        if hasattr(self.selector, "get_support"):
            selected_mask = self.selector.get_support()
            return [
                name for name, keep in zip(feature_names, selected_mask) if keep
            ]

        return feature_names

    def export_params(self) -> dict[str, Any]:
        """Export parameters for serialization and rerun functionality.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters in JSON-serializable format.
        """
        estimator_str = (
            type(self.estimator).__name__ if self.estimator else None
        )
        fs_estimator_str = (
            type(self.feature_selection_estimator).__name__
            if self.feature_selection_estimator else None
        )
        algo_config_str = (
            str(self.algorithm_config) if self.algorithm_config else None
        )

        return {
            "method": self.method,
            "n_features_to_select": self.n_features_to_select,
            "feature_selection_cv": self.feature_selection_cv,
            "estimator": estimator_str,
            "algorithm_config": algo_config_str,
            "feature_selection_estimator": fs_estimator_str,
            "problem_type": self.problem_type,
        }
