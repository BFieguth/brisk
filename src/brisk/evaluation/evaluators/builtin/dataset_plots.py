"""Implement evaluators to plot datasets."""
from typing import Any, List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from brisk.evaluation.evaluators import dataset_plot_evaluator

class HistogramBoxplot(dataset_plot_evaluator.DatasetPlotEvaluator):
    """Plot a histogram and boxplot for a dataset."""
    def plot(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        """Plot a histogram and boxplot for a dataset.
        
        Parameters
        ----------
        train_data (pd.Series): 
            The training data
        test_data (pd.Series): 
            The test data
        feature_name (str): 
            The name of the feature to plot
        filename (str): 
            The name of the file to save the plot to
        dataset_name (str): 
            The name of the dataset
        group_name (str): 
            The name of the experiment group

        Returns
        -------
        None
        """
        plot_data = self._generate_plot_data(
            train_data, test_data, feature_name
        )
        self._create_plot(plot_data)
        metadata = self._generate_metadata(
            dataset_name, group_name, feature_name
        )
        self._save_plot(filename, metadata)
        self._log_results(self.method_name, filename)

    def _generate_plot_data(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Generate the plot data.
        
        Parameters
        ----------
        train_data (pd.Series): 
            The training data
        test_data (pd.Series): 
            The test data
        feature_name (str): 
            The name of the feature to plot

        Returns
        -------
        Dict[str, Any]
            The plot data
        """
        plot_data = {
            "train_series": train_data,
            "test_series": test_data,
            "feature_name": feature_name
        }
        return plot_data

    def _create_plot(self, plot_data: Dict[str, Any]) -> None:
        """Create a histogram and boxplot for a dataset.
        
        Parameters
        ----------
        plot_data (Dict[str, Any]): 
            The plot data

        Returns
        -------
        None
        """
        _, axs = plt.subplots(
            nrows=2, ncols=2, sharex="col",
            gridspec_kw={"height_ratios": (3, 1)}, figsize=(12, 6)
        )

        bins_train = self._get_bin_number(plot_data["train_series"])
        bins_test = self._get_bin_number(plot_data["test_series"])

        axs[0, 0].hist(
            plot_data["train_series"], bins=bins_train, edgecolor="black",
            alpha=0.7
            )
        axs[0, 0].set_title(
            f"Train Distribution of {plot_data['feature_name']}",fontsize=14 # pylint: disable=W1405
        )
        axs[0, 0].set_ylabel("Frequency", fontsize=12)

        axs[0, 1].hist(
            plot_data["test_series"], bins=bins_test, edgecolor="black",
            alpha=0.7
        )
        axs[0, 1].set_title(
            f"Test Distribution of {plot_data['feature_name']}", fontsize=14 # pylint: disable=W1405
        )

        axs[1, 0].boxplot(plot_data["train_series"], orientation="horizontal")
        axs[1, 1].boxplot(plot_data["test_series"], orientation="horizontal")

        axs[1, 0].set_xlabel(plot_data["feature_name"], fontsize=12)
        axs[1, 1].set_xlabel(plot_data["feature_name"], fontsize=12)

        plt.tight_layout()

    def _get_bin_number(self, feature_series: pd.Series) -> int:
        """Get the number of bins for a given feature using Sturges' rule.
        
        Parameters
        ----------
        feature_series (pd.Series): 
            The series of feature values.

        Returns
        -------
        int
            The number of bins
        """
        return int(np.ceil(np.log2(len(feature_series)) + 1)) # Sturges' rule

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for output.
        
        Parameters
        ----------
        dataset_name (str): 
            The name of the dataset
        group_name (str): 
            The name of the experiment group
        feature_name (str): 
            The name of the feature

        Returns
        -------
        Dict[str, Any]
            The metadata
        """
        method = f"{self.method_name}_{feature_name}"
        return self.metadata.get_dataset(
            method, dataset_name, group_name
        )


class PiePlot(dataset_plot_evaluator.DatasetPlotEvaluator):
    """Plot a pie chart for a dataset."""
    def plot(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        """Plot a pie chart for a feature.
        
        Parameters
        ----------
        train_data (pd.Series): 
            The training data for the feature
        test_data (pd.Series): 
            The test data for the feature
        feature_name (str): 
            The name of the feature
        filename (str): 
            The name of the file to save the plot to
        dataset_name (str): 
            The name of the dataset
        group_name (str): 
            The name of the experiment group
        
        Returns
        -------
        None
        """
        plot_data = self._generate_plot_data(
            train_data, test_data, feature_name
        )
        self._create_plot(plot_data)
        metadata = self._generate_metadata(
            dataset_name, group_name, feature_name
        )
        self._save_plot(filename, metadata)
        self._log_results(self.method_name, filename)

    def _generate_plot_data(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Generate the plot data.
        
        Parameters
        ----------
        train_data (pd.Series): 
            The training data for the feature
        test_data (pd.Series): 
            The test data for the feature
        feature_name (str): 
            The name of the feature
        
        Returns
        -------
        Dict[str, Any]
            The plot data
        """
        plot_data = {
            "train_value_counts": train_data.value_counts(),
            "test_value_counts": test_data.value_counts(),
            "feature_name": feature_name
        }
        return plot_data

    def _create_plot(self, plot_data: Dict[str, Any]) -> None:
        """Create a pie chart for a feature.
        
        Parameters
        ----------
        plot_data (Dict[str, Any]): 
            The plot data

        Returns
        -------
        None
        """
        _, axs = plt.subplots(1, 2, figsize=(14, 8))

        axs[0].pie(
            plot_data["train_value_counts"],
            labels=plot_data["train_value_counts"].index,
            autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors
        )
        axs[0].set_title(f"Train {plot_data['feature_name']} Distribution") # pylint: disable=W1405

        axs[1].pie(
            plot_data["test_value_counts"],
            labels=plot_data["test_value_counts"].index,
            autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors
        )
        axs[1].set_title(f"Test {plot_data['feature_name']} Distribution") # pylint: disable=W1405

        plt.tight_layout()

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for output.
        
        Parameters
        ----------
        dataset_name (str): 
            The name of the dataset
        group_name (str): 
            The name of the experiment group
        feature_name (str): 
            The name of the feature
        
        Returns
        -------
        Dict[str, Any]
            The metadata
        """
        method = f"{self.method_name}_{feature_name}"
        return self.metadata.get_dataset(
            method, dataset_name, group_name
        )


class CorrelationMatrix(dataset_plot_evaluator.DatasetPlotEvaluator):
    """Plot a correlation matrix for a dataset."""
    def plot(
        self,
        train_data: pd.DataFrame,
        continuous_features: List[str],
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        """Plot a correlation matrix for a dataset.
        
        Parameters
        ----------
        train_data (pd.DataFrame): 
            The training data
        continuous_features (List[str]): 
            The names of the continuous features
        filename (str): 
            The name of the file to save the plot to
        dataset_name (str): 
            The name of the dataset
        group_name (str): 
            The name of the experiment group
        
        Returns
        -------
        None
        """
        plot_data = self._generate_plot_data(train_data, continuous_features)
        self._create_plot(plot_data)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_plot(filename, metadata)
        self._log_results(self.method_name, filename)

    def _generate_plot_data(
        self,
        train_data: pd.DataFrame,
        continuous_features: List[str],
    ) -> Dict[str, Any]:
        """Generate the plot data.
        
        Parameters
        ----------
        train_data (pd.DataFrame): 
            The training data
        continuous_features (List[str]): 
            The names of the continuous features
        
        Returns
        -------
        Dict[str, Any]
            The plot data
        """
        size_per_feature = 0.5
        plot_data = {
            "correlation_matrix": train_data[continuous_features].corr(),
            "width": max(12, size_per_feature * len(continuous_features)),
            "height": max(
                8, size_per_feature * len(continuous_features) * 0.75
            ),
        }
        return plot_data

    def _create_plot(self, plot_data: Dict[str, Any]) -> None:
        """Create a correlation matrix for a dataset.
        
        Parameters
        ----------
        plot_data (Dict[str, Any]): 
            The plot data

        Returns
        -------
        None
        """
        plt.figure(figsize=(plot_data["width"], plot_data["height"]))
        sns.heatmap(
            plot_data["correlation_matrix"], annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5
            )
        plt.title("Correlation Matrix of Continuous Features", fontsize=14)
        plt.tight_layout()
