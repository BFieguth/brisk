"""Implement evaluators that calculate measures on a dataset."""
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from brisk.evaluation.evaluators.dataset_measure_evaluator import DatasetMeasureEvaluator

class ContinuousStatistics(DatasetMeasureEvaluator):
    def _calculate_measures(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
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
    

class CategoricalStatistics(DatasetMeasureEvaluator):
    def _calculate_measures(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        stats = {}
        for feature in feature_names:
            feature_train = train_data[feature]
            feature_test = test_data[feature]
            feature_stats = {
                "train": {
                    "frequency": feature_train.value_counts().to_dict(),
                    "proportion": feature_train.value_counts(normalize=True).to_dict(),
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
                    "proportion": feature_test.value_counts(normalize=True).to_dict(),
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
