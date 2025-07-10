"""Store and analyze data splits created by DataManager.

This module defines the DataSplitInfo class, which is responsible for storing 
and analyzing data related to the training and testing splits of datasets within
the Brisk framework. The DataSplitInfo class provides methods for calculating 
descriptive statistics for both continuous and categorical features, as well as 
visualizing the distributions of these features through various plots.

Examples
--------
>>> from brisk.data.data_split_info import DataSplitInfo
>>> data_info = DataSplitInfo(X_train, X_test, y_train, y_test, 
...                          filename="dataset.csv", scaler=my_scaler)
>>> data_info.save_distribution("output_directory")
"""

import json
import logging
import os
from typing import Any, List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import seaborn as sns

class DataSplitInfo:
    """Store and analyze features and labels of training and testing splits.

    This class provides methods for calculating descriptive statistics for both 
    continuous and categorical features, as well as visualizing the 
    distributions of these features through various plots.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training features
    X_test : pd.DataFrame
        The testing features
    y_train : pd.Series
        The training labels
    y_test : pd.Series
        The testing labels
    filename : str
        The filename or table name of the dataset
    scaler : object, optional
        The scaler used for this split
    features : list of str, optional
        The order of input features
    categorical_features : list of str, optional
        List of categorical feature names

    Attributes
    ----------
    X_train : pd.DataFrame
        The training features
    X_test : pd.DataFrame
        The testing features
    y_train : pd.Series
        The training labels
    y_test : pd.Series
        The testing labels
    filename : str
        The filename or table name of the dataset
    scaler : object or None
        The scaler used for this split
    features : list of str or None
        The order of input features
    categorical_features : list of str
        List of categorical features present in the training dataset
    continuous_features : list of str
        List of continuous features derived from the training dataset
    continuous_stats : dict
        Descriptive statistics for continuous features
    categorical_stats : dict
        Statistics for categorical features

    Notes
    -----
    The class automatically detects categorical features if not provided. 
    Statistics are calculated for both continuous and categorical features 
    during initialization.
    """
    def __init__(
        self,
        X_train: pd.DataFrame, # pylint: disable=C0103
        X_test: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        y_test: pd.Series,
        filename: str,
        scaler: Optional[Any] = None,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
                )
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.X_train = X_train.copy(deep=True) # pylint: disable=C0103
        self.X_test = X_test.copy(deep=True) # pylint: disable=C0103
        self.y_train = y_train.copy(deep=True)
        self.y_test = y_test.copy(deep=True)

        self.filename = filename
        self.features = features

        if categorical_features is None:
            categorical_features = self._detect_categorical_features()

        self.categorical_features = [
            feature for feature in categorical_features
            if feature in X_train.columns
        ]

        self.continuous_features = [
            col for col in X_train.columns
            if col not in self.categorical_features
        ]

        self.scaler = scaler
        if self.continuous_features and scaler:
            self.scaler = scaler.fit(X_train[self.continuous_features])

        self.logger.info(
            "Calculating stats for continuous features in %s split.", 
            self.filename
        )

        self.continuous_stats = {}
        for feature in self.continuous_features:
            self.continuous_stats[feature] = {
                "train": self._calculate_continuous_stats(
                    self.X_train[feature]
                    ),
                "test": self._calculate_continuous_stats(
                    self.X_test[feature]
                    )
            }

        self.logger.info(
            "Calculating stats for categorical features in %s split.", 
            self.filename
            )

        self.categorical_stats = {}
        for feature in self.categorical_features:
            self.categorical_stats[feature] = {
                "train": self._calculate_categorical_stats(
                    self.X_train[feature], feature
                    ),
                "test": self._calculate_categorical_stats(
                    self.X_test[feature], feature
                    )
            }

    def _detect_categorical_features(self) -> List[str]:
        """Detect possible categorical features in the dataset.

        Checks datatype and if less than 5% of the columns have unique values.

        Returns
        -------
        list of str
            Names of detected categorical features

        Notes
        -----
        Features are considered categorical if they are:
        - Object dtype
        - Category dtype
        - Boolean dtype
        - Have less than 5% unique values
        """
        combined_data = pd.concat([self.X_train, self.X_test], axis=0)
        categorical_features = []

        for column in combined_data.columns:
            series = combined_data[column]
            n_unique = series.nunique()
            n_samples = len(series)

            is_categorical = any([
                series.dtype == "object",
                series.dtype == "category",
                series.dtype == "bool",
                (n_unique / n_samples < 0.05)
            ])

            if is_categorical:
                categorical_features.append(column)

        self.logger.info(
            "Detected %d categorical features: %s",
            len(categorical_features),
            categorical_features
        )
        return categorical_features

    def _calculate_continuous_stats(self, feature_series: pd.Series) -> dict:
        """Calculate descriptive statistics for a continuous feature.

        Args:
            feature_series (pd.Series): The series of continuous feature values.

        Returns:
            dict: A dictionary containing descriptive statistics such as mean, 
            median, standard deviation, variance, min, max, range, percentiles, 
            skewness, kurtosis, and coefficient of variation.
        """
        stats = {
            "mean": feature_series.mean(),
            "median": feature_series.median(),
            "std_dev": feature_series.std(),
            "variance": feature_series.var(),
            "min": feature_series.min(),
            "max": feature_series.max(),
            "range": feature_series.max() - feature_series.min(),
            "25_percentile": feature_series.quantile(0.25),
            "75_percentile": feature_series.quantile(0.75),
            "skewness": feature_series.skew(),
            "kurtosis": feature_series.kurt(),
            "coefficient_of_variation": (
                feature_series.std() / feature_series.mean()
                if feature_series.mean() != 0
                else None
                )
        }
        return stats

    def _calculate_categorical_stats(
        self,
        feature_series: pd.Series,
        feature_name: str
    ) -> dict:
        """Calculate statistics for a categorical feature.

        Args:
            feature_series (pd.Series): Series of categorical feature values.
            feature_name (str): The name of the categorical feature.

        Returns:
            dict: A dictionary containing frequency counts, proportions, number 
            of unique values, entropy, and Chi-Square test results.
        """
        stats = {
            "frequency": feature_series.value_counts().to_dict(),
            "proportion": feature_series.value_counts(normalize=True).to_dict(),
            "num_unique": feature_series.nunique(),
            "entropy": -np.sum(np.fromiter(
                (p * np.log2(p)
                    for p in feature_series.value_counts(normalize=True)
                    if p > 0),
                dtype=float
            ))
        }

        # Check if test data exists for Chi-Square test
        if feature_name in self.X_test.columns:
            train_counts = self.X_train[feature_name].value_counts()
            test_counts = self.X_test[feature_name].value_counts()

            # Create a contingency table for Chi-Square test
            contingency_table = pd.concat(
                [train_counts, test_counts], axis=1
                ).fillna(0)
            contingency_table.columns = ["train", "test"]

            # Perform the Chi-Square test for independence
            chi2, p_value, dof, _ = scipy_stats.chi2_contingency(
                contingency_table
                )
            stats["chi_square"] = {
                "chi2_stat": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof
            }
        else:
            stats["chi_square"] = None

        return stats

    def _get_bin_number(self, feature_series: pd.Series) -> int:
        """Get the number of bins for a given feature.
        
        Args:
            feature_series (pd.Series): The series of feature values.
        """
        # Sturges' rule
        return int(np.ceil(np.log2(len(feature_series)) + 1))

    def _plot_histogram_boxplot(
        self,
        feature_name: str,
        output_dir: str
    ) -> None:
        """Create and save histograms and boxplots for a given feature.

        Args:
            feature_name (str): The name of the feature to plot.
            output_dir (str): The directory where the plots will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        train_series = self.X_train[feature_name].dropna()
        test_series = self.X_test[feature_name].dropna()

        _, axs = plt.subplots(
            nrows=2, ncols=2, sharex="col",
            gridspec_kw={"height_ratios": (3, 1)}, figsize=(12, 6)
        )

        # Histograms
        bins_train = self._get_bin_number(train_series)
        bins_test = self._get_bin_number(test_series)

        axs[0, 0].hist(
            train_series, bins=bins_train, edgecolor="black", alpha=0.7
            )
        axs[0, 0].set_title(
            f"Train Distribution of {feature_name}", fontsize=14
            )
        axs[0, 0].set_ylabel("Frequency", fontsize=12)

        axs[0, 1].hist(
            test_series, bins=bins_test, edgecolor="black", alpha=0.7
            )
        axs[0, 1].set_title(
            f"Test Distribution of {feature_name}", fontsize=14
            )

        # Boxplots
        axs[1, 0].boxplot(train_series, orientation="horizontal")
        axs[1, 1].boxplot(test_series, orientation="horizontal")

        # Set labels
        axs[1, 0].set_xlabel(f"{feature_name}", fontsize=12)
        axs[1, 1].set_xlabel(f"{feature_name}", fontsize=12)

        # Save the plot
        plot_path = os.path.join(output_dir, f"{feature_name}_hist_box.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def _plot_correlation_matrix(self, output_dir: str) -> None:
        """Plot and save the correlation matrix for continuous features.

        Args:
            output_dir (str): The directory where the correlation matrix plot 
            will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        correlation_matrix = self.X_train[self.continuous_features].corr()

        # Calculate figure size based on number of features
        size_per_feature = 0.5
        plot_width = max(
            12, size_per_feature * len(self.continuous_features)
            )
        plot_height = max(
            8, size_per_feature * len(self.continuous_features) * 0.75
            )

        plt.figure(figsize=(plot_width, plot_height))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5
            )
        plt.title("Correlation Matrix of Continuous Features", fontsize=14)

        plot_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def _plot_categorical_pie(self, feature_name: str, output_dir: str) -> None:
        """
        Creates a pie chart to visualize the frequency distribution of a 
        categorical feature.

        Args:
            feature_name (str): The name of the feature for the chart title.
            output_dir (str): The directory where the pie chart will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        train_series = self.X_train[feature_name].dropna()
        test_series = self.X_test[feature_name].dropna()

        train_value_counts = train_series.value_counts()
        test_value_counts = test_series.value_counts()

        _, axs = plt.subplots(1, 2, figsize=(14, 8))

        axs[0].pie(
            train_value_counts, labels=train_value_counts.index,
            autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors
        )
        axs[0].set_title(f"Train {feature_name} Distribution")

        axs[1].pie(
            test_value_counts, labels=test_value_counts.index,
            autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors
        )
        axs[1].set_title(f"Test {feature_name} Distribution")

        plot_path = os.path.join(output_dir, f"{feature_name}_pie_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def get_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the training features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the training 
            features and training labels.
        """
        if self.scaler and self.continuous_features:
            categorical_data = (
                self.X_train[self.categorical_features].copy()
                if self.categorical_features
                else pd.DataFrame(index=self.X_train.index)
                )

            continuous_scaled = pd.DataFrame(
                self.scaler.transform(self.X_train[self.continuous_features]),
                columns=self.continuous_features,
                index=self.X_train.index
                )

            X_train_scaled = pd.concat( # pylint: disable=C0103
                [categorical_data, continuous_scaled], axis=1
                )
            X_train_scaled = X_train_scaled[self.X_train.columns] # pylint: disable=C0103
            return X_train_scaled, self.y_train
        return self.X_train, self.y_train

    def get_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the testing features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the testing 
            features and testing labels.
        """
        if self.scaler and self.continuous_features:
            categorical_data = (
                self.X_test[self.categorical_features].copy()
                if self.categorical_features
                else pd.DataFrame(index=self.X_test.index)
                )

            continuous_scaled = pd.DataFrame(
                self.scaler.transform(self.X_test[self.continuous_features]),
                columns=self.continuous_features,
                index=self.X_test.index
                )

            X_test_scaled = pd.concat( # pylint: disable=C0103
                [categorical_data, continuous_scaled], axis=1
                )
            X_test_scaled = X_test_scaled[self.X_test.columns] # pylint: disable=C0103
            return X_test_scaled, self.y_test
        return self.X_test, self.y_test

    def get_train_test(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Returns both the training and testing split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple 
            containing the training features, testing features, training 
            labels, and testing labels.
        """
        X_train, y_train = self.get_train() # pylint: disable=C0103
        X_test, y_test = self.get_test() # pylint: disable=C0103
        return X_train, X_test, y_train, y_test

    def save_distribution(self, dataset_dir: str) -> None:
        """Save the continuous and categorical statistics to JSON files.

        Args:
            dataset_dir (str): The directory where the statistics JSON files 
            and visualizations will be saved.
        """
        os.makedirs(dataset_dir, exist_ok=True)

        if self.continuous_stats:
            continuous_stats_path = os.path.join(
                dataset_dir, "continuous_stats.json"
                )
            with open(continuous_stats_path, "w", encoding="utf-8") as f:
                json.dump(self.continuous_stats, f, indent=4)

            for feature in self.continuous_features:
                self._plot_histogram_boxplot(
                    feature, os.path.join(dataset_dir, "hist_box_plot")
                )
            self._plot_correlation_matrix(dataset_dir)

        if self.categorical_stats:
            categorical_stats_path = os.path.join(
                dataset_dir, "categorical_stats.json"
                )
            with open(categorical_stats_path, "w", encoding="utf-8") as f:
                json.dump(self.categorical_stats, f, indent=4)

            for feature in self.categorical_features:
                self._plot_categorical_pie(
                    feature, os.path.join(dataset_dir, "pie_plot")
                    )

    def get_split_metadata(self) -> Dict[str, Any]:
        """Returns the split metadata used in certain metric calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the split metadata.
        """
        return {
            "num_features": len(self.X_train.columns),
            "num_samples": len(self.X_train) + len(self.X_test)
        }
