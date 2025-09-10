"""Evaluators that create plots for datasets.

This module provides built-in plot evaluators for creating various
visualizations of datasets. These evaluators help understand data
distributions, feature relationships, and dataset characteristics
through visual analysis.
"""
from typing import Any, List, Dict

import pandas as pd
import numpy as np
import plotnine as pn

from brisk.evaluation.evaluators import dataset_plot_evaluator

class Histogram(dataset_plot_evaluator.DatasetPlotEvaluator):
    """Plot histogram and boxplot visualizations for dataset features.
    
    This evaluator creates side-by-side histogram plots comparing the
    distribution of features between training and test datasets. It uses
    Sturges" rule to determine optimal bin counts and applies consistent
    styling across all visualizations.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to "histogram"
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize the Histogram evaluator."""
        super().__init__(method_name, description, plot_settings)
        self.name = "histogram"

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
        
        Creates side-by-side histogram visualizations comparing the distribution
        of a feature between training and test datasets. The plot includes
        faceted histograms with consistent binning and styling.

        Parameters
        ----------
        train_data : pd.Series
            The training data for the feature to plot
        test_data : pd.Series
            The test data for the feature to plot
        feature_name : str
            The name of the feature being plotted
        filename : str
            The filename to save the plot to
        dataset_name : str
            The name of the dataset being analyzed
        group_name : str
            The name of the experiment group

        Returns
        -------
        None
            The plot is saved to the specified filename
        """
        plot_data = self._generate_plot_data(
            train_data, test_data, feature_name
        )
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(
            dataset_name, group_name, feature_name
        )
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

    def _generate_plot_data(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Generate the plot data for histogram visualization.
        
        Prepares the data needed for creating histogram plots by organizing
        training and test data along with feature metadata.

        Parameters
        ----------
        train_data : pd.Series
            The training data for the feature
        test_data : pd.Series
            The test data for the feature
        feature_name : str
            The name of the feature being plotted

        Returns
        -------
        Dict[str, Any]
            Dictionary containing plot data with keys:
            - train_series: training data series
            - test_series: test data series
            - feature_name: name of the feature
        """
        plot_data = {
            "train_series": train_data,
            "test_series": test_data,
            "feature_name": feature_name
        }
        return plot_data

    def _create_plot(self, plot_data: Dict[str, Any]):
        """Create side-by-side histograms for train and test datasets.
        
        Generates a plotnine-based histogram visualization with faceted
        subplots comparing training and test data distributions.

        Parameters
        ----------
        plot_data : Dict[str, Any]
            Dictionary containing the data needed for plotting

        Returns
        -------
        plotnine.ggplot
            The generated histogram plot object
        """
        train_df = pd.DataFrame({
            "value": plot_data["train_series"],
            "dataset": pd.Categorical(
                ["Train"] * len(plot_data["train_series"]),
                categories=["Train", "Test"]
            )
        })
        test_df = pd.DataFrame({
            "value": plot_data["test_series"],
            "dataset": pd.Categorical(
                ["Test"] * len(plot_data["test_series"]),
                categories=["Train", "Test"]
            )
        })
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        bins_train = self._get_bin_number(plot_data["train_series"])
        bins_test = self._get_bin_number(plot_data["test_series"])
        hist_plot = (
            pn.ggplot(combined_df, pn.aes(x="value", fill="dataset")) +
            pn.geom_histogram(
                alpha=0.7,
                position="identity",
                bins=max(bins_train, bins_test),
                color="black"
            ) +
            pn.facet_wrap("~dataset", ncol=2, scales="free_y") +
            pn.labs(fill="Data Split") +
            pn.labs(
                title=f"Distribution of {plot_data['feature_name']}",
                x=plot_data["feature_name"],
                y="Frequency"
            ) +
            pn.scale_fill_manual(
                values=[self.primary_color, self.accent_color]
            ) +
            self.theme
        )

        return hist_plot

    def _get_bin_number(self, feature_series: pd.Series) -> int:
        """Get the number of bins for a given feature using Sturges" rule.
        
        Calculates the optimal number of histogram bins using Sturges" rule,
        which is based on the logarithm of the data size.

        Parameters
        ----------
        feature_series : pd.Series
            The series of feature values to determine bin count for

        Returns
        -------
        int
            The calculated number of bins using Sturges" rule
        """
        return int(np.ceil(np.log2(len(feature_series)) + 1)) # Sturges" rule

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for output.
        
        Creates metadata dictionary for the histogram plot including
        dataset and group information.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset being analyzed
        group_name : str
            The name of the experiment group
        feature_name : str
            The name of the feature being plotted

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata for the plot
        """
        method = f"{self.method_name}_{feature_name}"
        return self.metadata.get_dataset(
            method, dataset_name, group_name
        )


class BarPlot(dataset_plot_evaluator.DatasetPlotEvaluator):
    """Plot bar chart visualizations for categorical dataset features.
    
    This evaluator creates grouped bar charts comparing the proportions
    of categorical values between training and test datasets. It provides
    a clear visual comparison of class distributions across data splits.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to "bar_plot"
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize the BarPlot evaluator."""
        super().__init__(method_name, description, plot_settings)
        self.name = "bar_plot"

    def plot(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        """Plot a bar chart for categorical feature proportions.
        
        Creates a grouped bar chart comparing the proportions of categorical
        values between training and test datasets. The chart shows both
        absolute counts and normalized proportions.

        Parameters
        ----------
        train_data : pd.Series
            The training data for the categorical feature
        test_data : pd.Series
            The test data for the categorical feature
        feature_name : str
            The name of the categorical feature being plotted
        filename : str
            The filename to save the plot to
        dataset_name : str
            The name of the dataset being analyzed
        group_name : str
            The name of the experiment group

        Returns
        -------
        None
            The plot is saved to the specified filename
        """
        plot_data = self._generate_plot_data(
            train_data, test_data, feature_name
        )
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(
            dataset_name, group_name, feature_name
        )
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

    def _generate_plot_data(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Generate the plot data for bar chart visualization.
        
        Prepares the data needed for creating bar chart plots by calculating
        value counts and proportions for both training and test datasets.

        Parameters
        ----------
        train_data : pd.Series
            The training data for the categorical feature
        test_data : pd.Series
            The test data for the categorical feature
        feature_name : str
            The name of the feature being plotted

        Returns
        -------
        Dict[str, Any]
            Dictionary containing plot data with keys:
            - train_value_counts: value counts for training data
            - test_value_counts: value counts for test data
            - feature_name: name of the feature
        """
        plot_data = {
            "train_value_counts": train_data.value_counts(),
            "test_value_counts": test_data.value_counts(),
            "feature_name": feature_name
        }
        return plot_data

    def _create_plot(self, plot_data: Dict[str, Any]):
        """Create a grouped bar chart comparing categorical proportions between
        train and test.
        
        Generates a plotnine-based bar chart visualization with grouped bars
        showing proportions of categorical values for both training and test
        datasets.

        Parameters
        ----------
        plot_data : Dict[str, Any]
            Dictionary containing the data needed for plotting

        Returns
        -------
        plotnine.ggplot
            The generated bar chart plot object
        """
        train_df = pd.DataFrame({
            "category": plot_data["train_value_counts"].index,
            "count": plot_data["train_value_counts"].values,
            "proportion": (plot_data["train_value_counts"].values / 
                        plot_data["train_value_counts"].sum()),
            "dataset": pd.Categorical(
                ["Train"] * len(plot_data["train_value_counts"]),
                categories=["Train", "Test"]
            )
        })
        test_df = pd.DataFrame({
            "category": plot_data["test_value_counts"].index,
            "count": plot_data["test_value_counts"].values,
            "proportion": (plot_data["test_value_counts"].values / 
                        plot_data["test_value_counts"].sum()),
            "dataset": pd.Categorical(
                ["Test"] * len(plot_data["test_value_counts"]),
                categories=["Train", "Test"]
            )
        })

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        plot = (
            pn.ggplot(
                combined_df,
                pn.aes(x="category", y="proportion", fill="dataset")
            ) +
            pn.geom_col(position="dodge", alpha=0.7, color="black", width=0.7) +
            pn.labs(
                title=f"Proportion Comparison: {plot_data['feature_name']}",
                x=plot_data["feature_name"],
                y="Proportion",
                fill="Data Split"
            ) +
            pn.scale_y_continuous(
                labels=lambda x: [f"{val:.1%}" for val in x]
            ) +
            pn.scale_fill_manual(
                values=[self.primary_color, self.accent_color]
            ) +
            self.theme
        )
        return plot

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for output.
        
        Creates metadata dictionary for the bar chart plot including
        dataset and group information.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset being analyzed
        group_name : str
            The name of the experiment group
        feature_name : str
            The name of the feature being plotted

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata for the plot
        """
        method = f"{self.method_name}_{feature_name}"
        return self.metadata.get_dataset(
            method, dataset_name, group_name
        )


class CorrelationMatrix(dataset_plot_evaluator.DatasetPlotEvaluator):
    """Plot correlation matrix heatmaps for continuous features.
    
    This evaluator creates correlation matrix heatmaps showing the
    relationships between continuous features in the dataset. The heatmap
    uses a color gradient to represent correlation strength and includes
    correlation values as text annotations.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to "correlation_matrix"
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize the CorrelationMatrix evaluator."""
        super().__init__(method_name, description, plot_settings)
        self.name = "correlation_matrix"

    def plot(
        self,
        train_data: pd.DataFrame,
        continuous_features: List[str],
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        """Plot a correlation matrix for continuous features.
        
        Creates a correlation matrix heatmap showing the relationships
        between all continuous features in the dataset. The plot includes
        correlation values as text annotations and uses a color gradient
        to represent correlation strength.

        Parameters
        ----------
        train_data : pd.DataFrame
            The training data containing continuous features
        continuous_features : List[str]
            List of continuous feature names to include in the correlation
            matrix
        filename : str
            The filename to save the plot to
        dataset_name : str
            The name of the dataset being analyzed
        group_name : str
            The name of the experiment group

        Returns
        -------
        None
            The plot is saved to the specified filename
        """
        plot_data = self._generate_plot_data(train_data, continuous_features)
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

    def _generate_plot_data(
        self,
        train_data: pd.DataFrame,
        continuous_features: List[str],
    ) -> Dict[str, Any]:
        """Generate the plot data for correlation matrix visualization.
        
        Prepares the data needed for creating correlation matrix plots by
        calculating correlations and determining appropriate plot dimensions.

        Parameters
        ----------
        train_data : pd.DataFrame
            The training data containing continuous features
        continuous_features : List[str]
            List of continuous feature names to include in the matrix

        Returns
        -------
        Dict[str, Any]
            Dictionary containing plot data with keys:
            - correlation_matrix: pandas correlation matrix
            - width: calculated plot width based on number of features
            - height: calculated plot height based on number of features
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

    def _create_plot(self, plot_data: Dict[str, Any]):
        """Create a correlation matrix heatmap using plotnine.
        
        Generates a plotnine-based correlation matrix heatmap with
        color-coded tiles and correlation values as text annotations.

        Parameters
        ----------
        plot_data : Dict[str, Any]
            Dictionary containing the data needed for plotting

        Returns
        -------
        plotnine.ggplot
            The generated correlation matrix heatmap plot object
        """
        corr_matrix = plot_data["correlation_matrix"]

        corr_df = corr_matrix.reset_index().melt(
            id_vars="index",
            var_name="variable2",
            value_name="correlation"
        )
        corr_df.rename(columns={"index": "variable1"}, inplace=True)
        plot = (
            pn.ggplot(
                corr_df,
                pn.aes(x="variable1", y="variable2", fill="correlation")
            ) +
            pn.geom_tile(color="white", size=0.5) +
            pn.geom_text(
                pn.aes(label="correlation"),
                format_string="{:.2f}",
                size=8,
                color="black"
            ) +
            pn.scale_fill_gradient2(
                low=self.primary_color,
                high=self.accent_color,
                midpoint=0,
                name="Correlation",
                limits=(-1, 1)
            ) +
            pn.labs(
                title="Correlation Matrix of Continuous Features",
                x="",
                y=""
            ) +
            pn.theme(
                figure_size=(plot_data["width"], plot_data["height"]),
                axis_text_x=pn.element_text(angle=45, hjust=1)
            ) +
            self.theme
        )
        return plot
