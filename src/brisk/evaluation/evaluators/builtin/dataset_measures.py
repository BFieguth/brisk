"""Evaluators that calculate measures on datasets.

This module provides built-in evaluators for calculating various statistical
measures and metrics on datasets. These evaluators help understand the
characteristics and distribution of data in both training and test sets.
"""
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from brisk.evaluation.evaluators import dataset_measure_evaluator

class ContinuousStatistics(dataset_measure_evaluator.DatasetMeasureEvaluator):
    """Calculate continuous statistics for a dataset.
    
    This evaluator calculates comprehensive descriptive statistics for
    continuous features in both training and test datasets, including
    measures of central tendency, dispersion, and distribution shape.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to 'continuous_statistics'
    """

    def __init__(self, method_name: str, description: str):
        """Initialize the ContinuousStatistics evaluator."""
        super().__init__(method_name, description)
        self.name = "continuous_statistics"

    def _calculate_measures(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate continuous statistics for a dataset.

        Calculates comprehensive descriptive statistics for continuous features
        including mean, median, standard deviation, variance, min/max values,
        percentiles, skewness, kurtosis, and coefficient of variation.

        Parameters
        ----------
        train_data : pd.DataFrame or pd.Series
            The training data containing continuous features
        test_data : pd.DataFrame or pd.Series
            The test data containing continuous features
        feature_names : List[str]
            The names of the continuous features to calculate statistics for

        Returns
        -------
        Dict[str, Dict[str, float]]
            A nested dictionary containing statistics for each feature.
            Structure: {feature_name: {split: {statistic: value}}}
            where split is 'train' or 'test' and statistic includes:
            - mean, median, std_dev, variance
            - min, max, range
            - 25_percentile, 75_percentile
            - skewness, kurtosis, coefficient_of_variation
        """
        stats = {}
        for feature in feature_names:
            feature_train = train_data[feature]
            feature_test = test_data[feature]
            feature_stats = {
                "train": {
                    "mean": feature_train.mean(),
                    "median": feature_train.median(),
                    "std_dev": feature_train.std(),
                    "variance": feature_train.var(),
                    "min": feature_train.min(),
                    "max": feature_train.max(),
                    "range": feature_train.max() - feature_train.min(),
                    "25_percentile": feature_train.quantile(0.25),
                    "75_percentile": feature_train.quantile(0.75),
                    "skewness": feature_train.skew(),
                    "kurtosis": feature_train.kurt(),
                    "coefficient_of_variation": (
                        feature_train.std() / feature_train.mean()
                        if feature_train.mean() != 0
                        else None
                    )
                },
                "test": {
                    "mean": feature_test.mean(),
                    "median": feature_test.median(),
                    "std_dev": feature_test.std(),
                    "variance": feature_test.var(),
                    "min": feature_test.min(),
                    "max": feature_test.max(),
                    "range": feature_test.max() - feature_test.min(),
                    "25_percentile": feature_test.quantile(0.25),
                    "75_percentile": feature_test.quantile(0.75),
                    "skewness": feature_test.skew(),
                    "kurtosis": feature_test.kurt(),
                    "coefficient_of_variation": (
                        feature_test.std() / feature_test.mean()
                        if feature_test.mean() != 0
                        else None
                    )
                }
            }
            stats[feature] = feature_stats
        return stats


class CategoricalStatistics(dataset_measure_evaluator.DatasetMeasureEvaluator):
    """Calculate categorical statistics for a dataset.
    
    This evaluator calculates descriptive statistics for categorical features
    in both training and test datasets, including frequency distributions,
    proportions, entropy, and chi-square tests for distribution differences.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to 'categorical_statistics'
    """

    def __init__(self, method_name: str, description: str):
        """Initialize the CategoricalStatistics evaluator."""
        super().__init__(method_name, description)
        self.name = "categorical_statistics"

    def _calculate_measures(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate categorical statistics for a dataset.

        Calculates comprehensive descriptive statistics for categorical features
        including frequency distributions, proportions, entropy, and chi-square
        tests to assess distribution differences between train and test sets.

        Parameters
        ----------
        train_data : pd.DataFrame or pd.Series
            The training data containing categorical features
        test_data : pd.DataFrame or pd.Series
            The test data containing categorical features
        feature_names : List[str]
            The names of the categorical features to calculate statistics for

        Returns
        -------
        Dict[str, Dict[str, float]]
            A nested dictionary containing statistics for each feature.
            Structure: {feature_name: {split: {statistic: value}}}
            where split is 'train' or 'test' and statistic includes:
            - frequency: dict of value counts
            - proportion: dict of normalized value counts
            - num_unique: number of unique values
            - entropy: Shannon entropy of the distribution
            - chi_square: chi-square test results (chi2_stat, p_value,
            degrees_of_freedom)
        """
        stats = {}
        for feature in feature_names:
            feature_train = train_data[feature]
            feature_test = test_data[feature]
            feature_stats = {
                "train": {
                    "frequency": feature_train.value_counts().to_dict(),
                    "proportion": feature_train.value_counts(
                        normalize=True
                    ).to_dict(),
                    "num_unique": feature_train.nunique(),
                    "entropy": -np.sum(np.fromiter(
                        (p * np.log2(p)
                        for p in feature_train.value_counts(normalize=True)
                        if p > 0),
                        dtype=float
                    ))
                },
                "test": {
                    "frequency": feature_test.value_counts().to_dict(),
                    "proportion": feature_test.value_counts(
                        normalize=True
                    ).to_dict(),
                    "num_unique": feature_test.nunique(),
                    "entropy": -np.sum(np.fromiter(
                        (p * np.log2(p)
                        for p in feature_test.value_counts(normalize=True)
                        if p > 0),
                        dtype=float
                    ))
                }
            }
            train_counts = feature_train.value_counts()
            test_counts = feature_test.value_counts()
            contingency_table = pd.concat(
                [train_counts, test_counts], axis=1
                ).fillna(0)
            contingency_table.columns = ["train", "test"]
            chi2, p_value, dof, _ = scipy_stats.chi2_contingency(
                contingency_table
            )
            feature_stats["chi_square"] = {
                "chi2_stat": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof
            }
            stats[feature] = feature_stats
        return stats
