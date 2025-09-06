"""Implement evaluators to plot datasets."""
from typing import Any, List, Dict

import pandas as pd
import numpy as np
import plotnine as pn

from brisk.evaluation.evaluators import dataset_plot_evaluator

class Histogram(dataset_plot_evaluator.DatasetPlotEvaluator):
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

    def _create_plot(self, plot_data: Dict[str, Any]):
        """Create side-by-side histograms for train and test datasets.
        
        Parameters
        ----------
        plot_data (Dict[str, Any]): 
            The plot data

        Returns
        -------
        plotnine plot object
        """
        train_df = pd.DataFrame({
            'value': plot_data["train_series"],
            'dataset': pd.Categorical(['Train'] * len(plot_data["train_series"]), 
                                    categories=['Train', 'Test'])
        })
        test_df = pd.DataFrame({
            'value': plot_data["test_series"],
            'dataset': pd.Categorical(['Test'] * len(plot_data["test_series"]), 
                                    categories=['Train', 'Test'])
        })
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        bins_train = self._get_bin_number(plot_data["train_series"])
        bins_test = self._get_bin_number(plot_data["test_series"])
        
        hist_plot = (
            pn.ggplot(combined_df, pn.aes(x='value', fill='dataset')) +
            pn.geom_histogram(alpha=0.7, position='identity', 
                            bins=max(bins_train, bins_test), color='black') +
            pn.facet_wrap('~dataset', ncol=2, scales='free_y') +
            pn.labs(fill='Data Split') +
            pn.labs(
                title=f"Distribution of {plot_data['feature_name']}",
                x=plot_data['feature_name'],
                y='Frequency'
            ) +
            pn.scale_fill_manual(values=[self.primary_color, self.accent_color]) +
            self.theme
        )

        return hist_plot

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


class BarPlot(dataset_plot_evaluator.DatasetPlotEvaluator):
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

    def _create_plot(self, plot_data: Dict[str, Any]):
        """Create a grouped bar chart comparing categorical proportions between train and test.
        
        Parameters
        ----------
        plot_data (Dict[str, Any]): 
            The plot data

        Returns
        -------
        plotnine plot object
        """
        train_df = pd.DataFrame({
            'category': plot_data["train_value_counts"].index,
            'count': plot_data["train_value_counts"].values,
            'proportion': (plot_data["train_value_counts"].values / 
                        plot_data["train_value_counts"].sum()),
            'dataset': pd.Categorical(['Train'] * len(plot_data["train_value_counts"]), 
                                    categories=['Train', 'Test'])
        })
        
        test_df = pd.DataFrame({
            'category': plot_data["test_value_counts"].index,
            'count': plot_data["test_value_counts"].values,
            'proportion': (plot_data["test_value_counts"].values / 
                        plot_data["test_value_counts"].sum()),
            'dataset': pd.Categorical(['Test'] * len(plot_data["test_value_counts"]), 
                                    categories=['Train', 'Test'])
        })
        
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        plot = (
            pn.ggplot(combined_df, pn.aes(x='category', y='proportion', fill='dataset')) +
            pn.geom_col(position='dodge', alpha=0.7, color='black', width=0.7) +
            pn.labs(
                title=f"Proportion Comparison: {plot_data['feature_name']}",
                x=plot_data['feature_name'],
                y='Proportion',
                fill='Data Split'
            ) +
            pn.scale_y_continuous(labels=lambda x: [f"{val:.1%}" for val in x]) +
            pn.scale_fill_manual(values=[self.primary_color, self.accent_color]) +
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
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_plot(filename, metadata, plot=plot)
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

    def _create_plot(self, plot_data: Dict[str, Any]):
        """Create a correlation matrix heatmap using plotnine.
        
        Parameters
        ----------
        plot_data (Dict[str, Any]): 
            The plot data

        Returns
        -------
        plotnine plot object
        """
        corr_matrix = plot_data["correlation_matrix"]
        
        corr_df = corr_matrix.reset_index().melt(
            id_vars='index',
            var_name='variable2',
            value_name='correlation'
        )
        corr_df.rename(columns={'index': 'variable1'}, inplace=True)
        
        plot = (
            pn.ggplot(corr_df, pn.aes(x='variable1', y='variable2', fill='correlation')) +
            pn.geom_tile(color='white', size=0.5) +
            pn.geom_text(pn.aes(label='correlation'), 
                    format_string='{:.2f}', 
                    size=8, 
                    color='black') +
            pn.scale_fill_gradient2(
                low=self.primary_color, 
                high=self.accent_color, 
                midpoint=0,
                name='Correlation',
                limits=(-1, 1)
            ) +
            pn.labs(
                title='Correlation Matrix of Continuous Features',
                x='',
                y=''
            ) +
            pn.theme(
                figure_size=(plot_data["width"], plot_data["height"]),
                axis_text_x=pn.element_text(angle=45, hjust=1)
            ) +
            self.theme
        )
        
        return plot
