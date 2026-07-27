"""Unit tests for the sklearn preprocessor adapter.

Adapted from ``tests/integration/data/test_preprocessing_integration.py``.
The integration suite exercises the original preprocessors through
``DataManager.split()``; ``DataManager`` selects preprocessors via
``isinstance`` against the *domain* classes, so adapter instances must be
tested directly. These tests drive the adapter classes through their
fit/transform interface and assert the same behaviours, plus
``PreprocessorPort`` conformance.
"""

import numpy as np
import pandas as pd
import pytest

from brisk.adapters.sklearn.preprocessor_adapter import (
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor,
    MissingDataPreprocessor,
    ScalingPreprocessor,
)
from brisk.ports.preprocessor import PreprocessorPort

# pylint: disable=W0621, C0103


@pytest.fixture()
def numeric_df():
    """Numeric features with no missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        "feat_a": np.random.randn(50) * 10 + 50,
        "feat_b": np.random.randn(50) * 5 + 20,
        "feat_c": np.random.randn(50) * 2 + 100,
    })


@pytest.fixture()
def missing_df():
    """Numeric features with scattered NaN values."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_a": np.random.randn(50) * 10 + 50,
        "feat_b": np.random.randn(50) * 5 + 20,
        "feat_c": np.random.randn(50) * 2 + 100,
    })
    for idx in [0, 3, 5, 10, 15, 20, 25, 30]:
        col = ["feat_a", "feat_b", "feat_c"][idx % 3]
        df.loc[idx, col] = np.nan
    return df


@pytest.fixture()
def categorical_df():
    """One categorical and one numeric feature."""
    np.random.seed(42)
    return pd.DataFrame({
        "num_feat": np.random.randn(50) * 10 + 50,
        "cat_feat": np.random.choice(["A", "B", "C"], size=50),
    })


@pytest.fixture()
def multi_feature_df():
    """Several numeric features for feature selection."""
    np.random.seed(42)
    return pd.DataFrame({f"feat_{i}": np.random.randn(80) for i in range(6)})


# ---------------------------------------------------------------------------
# PreprocessorPort conformance
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPreprocessorPortConformance:
    @pytest.mark.parametrize(
        "factory",
        [
            MissingDataPreprocessor,
            ScalingPreprocessor,
            CategoricalEncodingPreprocessor,
            FeatureSelectionPreprocessor,
        ],
    )
    def test_satisfies_preprocessor_port(self, factory):
        assert isinstance(factory(), PreprocessorPort)


# ---------------------------------------------------------------------------
# MissingDataPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestMissingDataPreprocessor:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            MissingDataPreprocessor(strategy="not_a_strategy")

    def test_invalid_impute_method_raises(self):
        with pytest.raises(ValueError):
            MissingDataPreprocessor(strategy="impute", impute_method="bad")

    def test_transform_before_fit_raises(self, missing_df):
        pre = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        with pytest.raises(ValueError, match="must be fitted"):
            pre.transform(missing_df, None)

    def test_impute_mean_removes_all_nan(self, missing_df):
        pre = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        Xt, _ = pre.fit_transform(missing_df, None)
        assert not Xt.isnull().any().any()
        assert pre.is_fitted

    def test_impute_median_removes_all_nan(self, missing_df):
        pre = MissingDataPreprocessor(strategy="impute", impute_method="median")
        Xt, _ = pre.fit_transform(missing_df, None)
        assert not Xt.isnull().any().any()

    def test_drop_rows_aligns_x_and_y(self, missing_df):
        y = pd.Series(np.random.choice([0, 1], size=len(missing_df)))
        pre = MissingDataPreprocessor(strategy="drop_rows")
        Xt, yt = pre.fit_transform(missing_df, y)
        assert not Xt.isnull().any().any()
        assert len(Xt) == len(yt)

    def test_impute_preserves_row_count(self, missing_df):
        y = pd.Series(np.random.choice([0, 1], size=len(missing_df)))
        pre = MissingDataPreprocessor(strategy="impute", impute_method="mean")
        Xt, yt = pre.fit_transform(missing_df, y)
        assert len(Xt) == len(missing_df)
        assert len(yt) == len(y)

    def test_export_params(self):
        pre = MissingDataPreprocessor(
            strategy="impute", impute_method="constant", constant_value=-1
        )
        assert pre.export_params() == {
            "strategy": "impute",
            "impute_method": "constant",
            "constant_value": -1,
        }


# ---------------------------------------------------------------------------
# ScalingPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestScalingPreprocessor:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ScalingPreprocessor(method="not_a_method")

    def test_standard_scaling_transforms_values(self, numeric_df):
        pre = ScalingPreprocessor(method="standard")
        Xt, _ = pre.fit_transform(numeric_df, None)
        for col in Xt.columns:
            assert abs(Xt[col].mean()) < 0.5
            assert 0.5 < Xt[col].std() < 1.5

    def test_minmax_scaling(self, numeric_df):
        pre = ScalingPreprocessor(method="minmax")
        Xt, _ = pre.fit_transform(numeric_df, None)
        for col in Xt.columns:
            assert Xt[col].min() >= -0.01
            assert Xt[col].max() <= 1.01

    def test_scaling_preserves_shape(self, numeric_df):
        pre = ScalingPreprocessor(method="standard")
        Xt, _ = pre.fit_transform(numeric_df, None)
        assert Xt.shape == numeric_df.shape

    def test_scaler_object_created(self, numeric_df):
        pre = ScalingPreprocessor(method="standard")
        pre.fit(numeric_df, None)
        assert pre.scaler is not None

    def test_scaling_excludes_categorical_features(self, categorical_df):
        """Categorical features passed via categorical_features stay raw."""
        df = categorical_df.copy()
        df["cat_feat"] = df["cat_feat"].map({"A": 0, "B": 1, "C": 2})
        pre = ScalingPreprocessor(method="standard")
        Xt, _ = pre.fit_transform(
            df, None, categorical_features=["cat_feat"]
        )
        assert set(Xt["cat_feat"].unique()).issubset({0, 1, 2})

    def test_get_feature_names_unchanged(self, numeric_df):
        pre = ScalingPreprocessor(method="standard")
        pre.fit(numeric_df, None)
        names = list(numeric_df.columns)
        assert pre.get_feature_names(names) == names

    def test_export_params(self):
        assert ScalingPreprocessor(method="robust").export_params() == {
            "method": "robust"
        }


# ---------------------------------------------------------------------------
# CategoricalEncodingPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCategoricalEncodingPreprocessor:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            CategoricalEncodingPreprocessor(method="not_a_method")

    def test_threshold_without_cutoffs_raises(self):
        with pytest.raises(ValueError, match="cutoffs"):
            CategoricalEncodingPreprocessor(method="threshold")

    def test_onehot_creates_binary_columns(self, categorical_df):
        pre = CategoricalEncodingPreprocessor(method="onehot")
        Xt, _ = pre.fit_transform(
            categorical_df, None, categorical_features=["cat_feat"]
        )
        assert "cat_feat" not in Xt.columns
        onehot_cols = [c for c in Xt.columns if c.startswith("cat_feat_")]
        assert len(onehot_cols) >= 2
        for col in onehot_cols:
            assert set(Xt[col].unique()).issubset({0.0, 1.0})

    def test_label_encoding_produces_integers(self, categorical_df):
        pre = CategoricalEncodingPreprocessor(method="label")
        Xt, _ = pre.fit_transform(
            categorical_df, None, categorical_features=["cat_feat"]
        )
        assert "cat_feat" in Xt.columns
        assert set(Xt["cat_feat"].unique()).issubset({0, 1, 2})

    def test_ordinal_encoding(self, categorical_df):
        pre = CategoricalEncodingPreprocessor(method="ordinal")
        Xt, _ = pre.fit_transform(
            categorical_df, None, categorical_features=["cat_feat"]
        )
        assert Xt["cat_feat"].dtype in [np.float64, np.int64, np.int32]

    def test_encoding_preserves_numeric_feature(self, categorical_df):
        pre = CategoricalEncodingPreprocessor(method="label")
        Xt, _ = pre.fit_transform(
            categorical_df, None, categorical_features=["cat_feat"]
        )
        pd.testing.assert_series_equal(
            categorical_df["num_feat"], Xt["num_feat"], check_names=False
        )

    def test_get_feature_names_onehot(self, categorical_df):
        pre = CategoricalEncodingPreprocessor(method="onehot")
        pre.fit(categorical_df, None, categorical_features=["cat_feat"])
        names = pre.get_feature_names(["num_feat", "cat_feat"])
        assert "num_feat" in names
        assert "cat_feat" not in names
        assert any(n.startswith("cat_feat_") for n in names)

    def test_target_encoding_via_method_dict(self, categorical_df):
        """A dict method that names the target should encode y too."""
        y = pd.Series(["lo", "hi", "hi", "lo"] * 12 + ["lo", "hi"], name="tgt")
        pre = CategoricalEncodingPreprocessor(
            method={"cat_feat": "label", "tgt": "ordinal"}
        )
        _, yt = pre.fit_transform(
            categorical_df, y, categorical_features=["cat_feat"]
        )
        assert set(np.unique(yt)).issubset({0.0, 1.0})

    def test_target_passthrough_when_method_is_str(self, categorical_df):
        y = pd.Series(np.random.choice([0, 1], size=len(categorical_df)),
                      name="tgt")
        pre = CategoricalEncodingPreprocessor(method="label")
        _, yt = pre.fit_transform(
            categorical_df, y, categorical_features=["cat_feat"]
        )
        assert list(yt) == list(y)


# ---------------------------------------------------------------------------
# FeatureSelectionPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestFeatureSelectionPreprocessor:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FeatureSelectionPreprocessor(method="not_a_method")

    def test_n_features_below_one_raises(self):
        with pytest.raises(ValueError):
            FeatureSelectionPreprocessor(n_features_to_select=0)

    def test_selectkbest_reduces_feature_count(self, multi_feature_df):
        y = pd.Series(np.random.choice([0, 1], size=len(multi_feature_df)))
        pre = FeatureSelectionPreprocessor(
            method="selectkbest", n_features_to_select=3,
            problem_type="classification",
        )
        Xt, _ = pre.fit_transform(multi_feature_df, y)
        assert Xt.shape[1] == 3

    def test_selected_features_are_subset(self, multi_feature_df):
        original = set(multi_feature_df.columns)
        y = pd.Series(np.random.choice([0, 1], size=len(multi_feature_df)))
        pre = FeatureSelectionPreprocessor(
            method="selectkbest", n_features_to_select=3,
            problem_type="classification",
        )
        Xt, _ = pre.fit_transform(multi_feature_df, y)
        assert set(Xt.columns).issubset(original)

    def test_get_feature_names_returns_selected(self, multi_feature_df):
        y = pd.Series(np.random.choice([0, 1], size=len(multi_feature_df)))
        pre = FeatureSelectionPreprocessor(
            method="selectkbest", n_features_to_select=2,
            problem_type="classification",
        )
        pre.fit(multi_feature_df, y)
        selected = pre.get_feature_names(list(multi_feature_df.columns))
        assert len(selected) == 2

    def test_transform_before_fit_raises(self, multi_feature_df):
        pre = FeatureSelectionPreprocessor(method="selectkbest")
        with pytest.raises(ValueError, match="must be fitted"):
            pre.transform(multi_feature_df, None)

    def test_export_params(self):
        pre = FeatureSelectionPreprocessor(
            method="selectkbest", n_features_to_select=4,
            problem_type="regression",
        )
        params = pre.export_params()
        assert params["method"] == "selectkbest"
        assert params["n_features_to_select"] == 4
        assert params["problem_type"] == "regression"
